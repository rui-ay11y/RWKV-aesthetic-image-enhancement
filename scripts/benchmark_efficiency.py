"""Efficiency benchmarking for bottleneck modules.

Measures GFLOPs, inference latency, peak memory, and parameter counts
across three resolution tiers (256x256, 480x480, 1080x1920).

This script benchmarks the bottleneck in isolation, not the full
encoder-bottleneck-decoder pipeline.

Usage::

    # Benchmark the supported CNN bottleneck baseline
    python scripts/benchmark_efficiency.py --bottleneck cnn

    # Benchmark the currently implemented bottlenecks in sequence
    for bn in none cnn rwkv; do
        python scripts/benchmark_efficiency.py --bottleneck $bn
    done
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.efficiency import benchmark_bottleneck
from src.utils.logging_utils import export_results_json
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark bottleneck efficiency")
    p.add_argument(
        "--bottleneck",
        default="cnn",
        choices=["none", "cnn", "rwkv"],
        help="Bottleneck config name to benchmark",
    )
    p.add_argument(
        "--encoder-dim", type=int, default=768,
        help="Encoder output dimension fed into the bottleneck (default: DINOv2-B = 768)",
    )
    p.add_argument(
        "--patch-size", type=int, default=14,
        help="Encoder patch size (default: DINOv2-B = 14)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--num-warmup", type=int, default=50)
    p.add_argument("--num-runs",   type=int, default=200)
    p.add_argument(
        "--output", default=None,
        help="JSON output path (default: results/efficiency_{bottleneck}.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────
    device = str(get_device()) if args.device == "auto" else args.device

    # ── Instantiate bottleneck from config ────────────────────────────
    cfg_path = Path("configs/bottleneck") / f"{args.bottleneck}.yaml"
    cfg = OmegaConf.load(cfg_path)
    bottleneck = hydra.utils.instantiate(cfg, input_dim=args.encoder_dim)

    print(f"\n{'='*56}")
    print(f"  Bottleneck : {args.bottleneck}")
    print(f"  Device     : {device}")
    print(f"  Input dim  : {args.encoder_dim}  |  Patch size: {args.patch_size}")
    print(f"{'='*56}")

    # ── Benchmark ─────────────────────────────────────────────────────
    results = benchmark_bottleneck(
        bottleneck,
        resolutions=["256x256", "480x480", "1080x1920"],
        patch_size=args.patch_size,
        device=device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )
    results["bottleneck"] = args.bottleneck
    results["device"] = device

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  Params total     : {results['params_total']:>12,}")
    print(f"  Params trainable : {results['params_trainable']:>12,}")
    for res in ["256x256", "480x480", "1080x1920"]:
        print(f"\n  [{res}]")
        print(f"    GFLOPs  : {results['gflops'].get(res, -1):.3f}")
        print(f"    Latency : {results['latency_ms'].get(res, -1):.2f} ms")
        print(f"    Memory  : {results['peak_memory_mb'].get(res, -1):.1f} MB")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.output or f"results/efficiency_{args.bottleneck}.json"
    export_results_json(results, out_path)


if __name__ == "__main__":
    main()
