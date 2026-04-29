"""SSIM structural similarity loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> Tensor:
    """Create a normalised Gaussian convolution kernel.

    Args:
        window_size: Kernel spatial size (square).
        sigma: Gaussian standard deviation.
        channels: Number of input channels (kernel is repeated per channel).

    Returns:
        Kernel tensor of shape (channels, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)                                   # (W, W)
    return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def _ssim(
    pred: Tensor,
    target: Tensor,
    window: Tensor,
    window_size: int,
    channels: int,
    data_range: float = 1.0,
) -> Tensor:
    """Compute mean SSIM between two image tensors.

    Args:
        pred, target: Image tensors (B, C, H, W).
        window: Pre-computed Gaussian kernel.
        window_size: Spatial size of the kernel.
        channels: Number of image channels.
        data_range: Dynamic range of pixel values.

    Returns:
        Scalar mean SSIM value.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu1 = F.conv2d(pred,   window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(pred   * target, window, padding=pad, groups=channels) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (numerator / denominator).mean()


class SSIMLoss(nn.Module):
    """Structural Similarity loss: weight × (1 − SSIM).

    Args:
        weight: Scalar multiplier.
        window_size: Gaussian window size.
        sigma: Gaussian standard deviation.
    """

    def __init__(
        self,
        weight: float = 0.1,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.window_size = window_size
        self.register_buffer("window", _gaussian_kernel(window_size, sigma, 3))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute weighted SSIM loss.

        Args:
            pred:   Predicted image (B, 3, H, W) in [0, 1].
            target: Ground truth image (B, 3, H, W) in [0, 1].

        Returns:
            Scalar loss = weight × (1 − SSIM).
        """
        ssim_val = _ssim(pred, target, self.window, self.window_size, 3)
        return self.weight * (1.0 - ssim_val)
