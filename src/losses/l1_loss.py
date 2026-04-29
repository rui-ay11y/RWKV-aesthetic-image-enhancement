"""L1 pixel-wise reconstruction loss."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class L1Loss(nn.Module):
    """Weighted mean absolute error (L1) loss.

    Args:
        weight: Scalar multiplier applied to the raw L1 value.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute weighted L1 loss.

        Args:
            pred:   Predicted image (B, 3, H, W).
            target: Ground truth image (B, 3, H, W).

        Returns:
            Scalar loss = weight × mean(|pred − target|).
        """
        return self.weight * F.l1_loss(pred, target)
