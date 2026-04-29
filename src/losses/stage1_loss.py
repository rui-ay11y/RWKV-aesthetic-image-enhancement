"""Stage-1 loss: L1 + 0.1 * (1 - SSIM) + 0.01 * Perceptual."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .perceptual_loss import PerceptualLoss


def ssim_index(pred: Tensor, target: Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> Tensor:
    """Compute mean SSIM per sample."""
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (numerator / (denominator + 1e-8)).mean(dim=(1, 2, 3))


class Stage1Loss(nn.Module):
    """Exact formula requested by user:

    total = L1 + 0.1 * (1 - SSIM) + 0.01 * Perceptual
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.01,
        perceptual_layers: Sequence[str] | None = None,
        perceptual_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual = PerceptualLoss(
            weight=1.0,
            layers=list(perceptual_layers) if perceptual_layers is not None else None,
            pretrained=perceptual_pretrained,
        )

    def forward(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim_index(pred, target).mean()
        ssim_loss = 1.0 - ssim_val
        perceptual = self.perceptual(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss + self.perceptual_weight * perceptual
        return {
            "total": total,
            "l1": l1,
            "ssim": ssim_val,
            "ssim_loss": ssim_loss,
            "perceptual": perceptual,
        }
