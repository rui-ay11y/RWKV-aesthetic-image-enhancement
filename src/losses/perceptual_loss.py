"""VGG16-based perceptual loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PerceptualLoss(nn.Module):
    """Perceptual loss using intermediate VGG16 feature maps.

    Computes L1 distance between pred and target in the feature space of
    a frozen VGG16 at the specified layer checkpoints.

    Args:
        weight: Scalar multiplier for the total perceptual loss.
        layers: List of layer names to use. Supported values:
                'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'.
    """

    # Maps human-readable layer name → VGG16 feature index (exclusive end)
    _LAYER_IDX: dict[str, int] = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_3": 16,
        "relu4_3": 23,
    }

    def __init__(
        self,
        weight: float = 0.01,
        layers: list[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.weight = weight
        layer_names = layers or ["relu1_2", "relu2_2", "relu3_3"]

        import torchvision.models as models
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg_feats = models.vgg16(weights=weights).features

        # Build sequential slices between consecutive target layers
        indices = sorted(self._LAYER_IDX[n] for n in layer_names)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in indices:
            self.slices.append(nn.Sequential(*list(vgg_feats.children())[prev:idx]))
            prev = idx

        # Freeze VGG — no gradient, no optimizer entry
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: Tensor) -> Tensor:
        """Map [0, 1] input to ImageNet-normalised space."""
        return (x - self.mean) / self.std

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute weighted perceptual loss.

        Args:
            pred:   Predicted image (B, 3, H, W) in [0, 1].
            target: Ground truth image (B, 3, H, W) in [0, 1].

        Returns:
            Scalar loss = weight × Σ L1(feat_pred_i, feat_target_i).
        """
        p = self._normalize(pred)
        t = self._normalize(target)

        loss = pred.new_zeros(1).squeeze()
        for slice_net in self.slices:
            p = slice_net(p)
            t = slice_net(t)
            loss = loss + F.l1_loss(p, t.detach())

        return self.weight * loss
