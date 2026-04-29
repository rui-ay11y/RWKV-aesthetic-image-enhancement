"""Efficiency benchmarking: GFLOPs, latency, memory, parameters."""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from src.models.bottlenecks.base import BaseBottleneck


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: Any PyTorch module.

    Returns:
        Tuple (total_params, trainable_params).
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_gflops(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """Estimate GFLOPs for one forward pass using fvcore.

    Falls back to -1.0 if fvcore is not installed or the model is not
    supported.

    Args:
        model:        Module to profile.
        input_tensor: Example input (must match model's expected shape).

    Returns:
        GFLOPs (float), or -1.0 on failure.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        return flops.total() / 1e9
    except Exception:
        return -1.0


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 50,
    num_runs: int = 200,
    device: str = "cuda",
) -> float:
    """Measure average inference latency in milliseconds.

    Args:
        model:        Module to benchmark (moved to device inside).
        input_tensor: Input tensor (moved to device inside).
        num_warmup:   Warm-up iterations (not timed).
        num_runs:     Timed iterations.
        device:       Target device string.

    Returns:
        Mean latency in ms.
    """
    model = model.to(device).eval()
    x = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (elapsed / num_runs) * 1000.0  # → ms


def benchmark_bottleneck(
    bottleneck: BaseBottleneck,
    resolutions: list[str] | None = None,
    patch_size: int = 14,
    device: str = "cuda",
    num_warmup: int = 50,
    num_runs: int = 200,
) -> dict[str, Any]:
    """Benchmark a bottleneck module across multiple image resolutions.

    For each resolution HxW the token count is N = (H/patch_size)*(W/patch_size)
    and a random (1, N, input_dim) tensor is used as input.

    Args:
        bottleneck:   Bottleneck module to benchmark.
        resolutions:  List of 'HxW' strings. Defaults to three standard sizes.
        patch_size:   Encoder patch size.
        device:       Device to run on.
        num_warmup:   Warm-up iterations per resolution.
        num_runs:     Timed iterations per resolution.

    Returns:
        Dict with keys 'params_total', 'params_trainable',
        'gflops', 'latency_ms', 'peak_memory_mb' — the last three
        are nested dicts keyed by resolution string.
    """
    if resolutions is None:
        resolutions = ["256x256", "480x480", "1080x1920"]

    total_params, trainable_params = count_parameters(bottleneck)
    results: dict[str, Any] = {
        "params_total":     total_params,
        "params_trainable": trainable_params,
        "gflops":           {},
        "latency_ms":       {},
        "peak_memory_mb":   {},
    }

    # Retrieve input_dim from the bottleneck's proj_in layer
    proj_in = getattr(bottleneck, "proj_in", None)
    input_dim = proj_in.in_features if proj_in is not None else 768

    bottleneck.eval()
    bottleneck = bottleneck.to(device)

    for res_str in resolutions:
        h_s, w_s = res_str.split("x")
        n_tokens = (int(h_s) // patch_size) * (int(w_s) // patch_size)
        dummy = torch.randn(1, n_tokens, input_dim, device=device)

        results["gflops"][res_str]     = compute_gflops(bottleneck, dummy)
        results["latency_ms"][res_str] = measure_latency(
            bottleneck, dummy, num_warmup=num_warmup, num_runs=num_runs, device=device
        )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = bottleneck(dummy)
            results["peak_memory_mb"][res_str] = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            results["peak_memory_mb"][res_str] = -1.0

    return results
