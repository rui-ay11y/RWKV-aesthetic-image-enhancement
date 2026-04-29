"""Image quality metrics: PSNR, SSIM, LPIPS."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_psnr(pred: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """Compute mean Peak Signal-to-Noise Ratio across a batch.

    Args:
        pred:       Predicted images (B, 3, H, W) in [0, 1].
        target:     Ground truth images (B, 3, H, W) in [0, 1].
        data_range: Pixel value range (1.0 for normalised images).

    Returns:
        Mean PSNR scalar (dB).
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])  # (B,)
    psnr = 10.0 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return psnr.mean()


def compute_ssim(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Compute mean SSIM across a batch.

    Reuses the Gaussian kernel from SSIMLoss to avoid code duplication.

    Args:
        pred:        Predicted images (B, 3, H, W) in [0, 1].
        target:      Ground truth images (B, 3, H, W) in [0, 1].
        window_size: Gaussian window size.

    Returns:
        Mean SSIM scalar.
    """
    from src.losses.ssim_loss import _gaussian_kernel, _ssim
    window = _gaussian_kernel(window_size, 1.5, 3).to(pred.device).to(pred.dtype)
    return _ssim(pred, target, window, window_size, 3)


def compute_lpips(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean LPIPS perceptual distance across a batch.

    Requires the ``lpips`` package. Returns 0 if unavailable.

    Args:
        pred:   Predicted images (B, 3, H, W) in [0, 1].
        target: Ground truth images (B, 3, H, W) in [0, 1].

    Returns:
        Mean LPIPS distance scalar (lower = more similar).
    """
    try:
        import lpips as _lpips_lib
        # Cache LPIPS model per device to avoid repeated loading
        device_key = str(pred.device)
        cache = compute_lpips.__dict__.setdefault("_cache", {})
        if device_key not in cache:
            cache[device_key] = _lpips_lib.LPIPS(net="alex").to(pred.device)
        lpips_fn = cache[device_key]

        # LPIPS expects inputs in [-1, 1]
        with torch.no_grad():
            return lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
    except ImportError:
        return pred.new_zeros(1).squeeze()


class MetricCollection:
    """Convenience wrapper that computes PSNR, SSIM, and LPIPS together.

    Args:
        cfg: Hydra config (reserved for future per-experiment metric flags).
    """

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg

    def _lpips_enabled(self) -> bool:
        """Return whether LPIPS should be computed for this run."""
        if self.cfg is None:
            return True

        evaluation_cfg = self.cfg.get("evaluation")
        if evaluation_cfg is None:
            return True

        return bool(evaluation_cfg.get("enable_lpips", True))

    def __call__(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute all metrics for a batch.

        Args:
            pred:   Predicted images (B, 3, H, W) in [0, 1].
            target: Ground truth images (B, 3, H, W) in [0, 1].

        Returns:
            Dict with keys 'psnr', 'ssim', 'lpips'.
        """
        lpips_value = (
            compute_lpips(pred, target)
            if self._lpips_enabled()
            else pred.new_zeros(())
        )
        return {
            "psnr":  compute_psnr(pred, target),
            "ssim":  compute_ssim(pred, target),
            "lpips": lpips_value,
        }
