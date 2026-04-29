"""Frozen DINOv2 encoder wrapper (E-2, default encoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders.base import BaseEncoder


class DINOv2Encoder(BaseEncoder):
    """Frozen DINOv2 ViT encoder.

    Loads a pretrained DINOv2 model via timm and extracts patch token
    features (CLS token is discarded). Always kept in eval mode.

    Args:
        name: timm model name, e.g. 'vit_base_patch14_dinov2'.
        pretrained: Whether to load pretrained ImageNet weights.
        frozen: Whether to freeze all parameters (should always be True).
        patch_size: Patch size of the ViT backbone (14 for DINOv2-B).
        embed_dim: Output feature dimension (768 for DINOv2-B).
    """

    def __init__(
        self,
        name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        frozen: bool = True,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim

        self.backbone = self._load_backbone(name, pretrained, img_size)

        if frozen:
            self.freeze()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(name: str, pretrained: bool, img_size: int) -> nn.Module:
        """Load DINOv2 backbone, trying timm first then torch.hub."""
        timm_name = {
            "dinov2_vitb14": "vit_base_patch14_dinov2",
            "dinov2_vitl14": "vit_large_patch14_dinov2",
            "dinov2_vitg14": "vit_giant_patch14_dinov2",
            "dinov2_vits14": "vit_small_patch14_dinov2",
        }.get(name, name)

        try:
            import timm
            # timm >= 0.9 ships dinov2 models
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
            )
            return model
        except Exception:
            pass

        # Fallback: Facebook's official torch.hub release
        hub_name = {
            "vit_base_patch14_dinov2": "dinov2_vitb14",
            "vit_large_patch14_dinov2": "dinov2_vitl14",
            "vit_giant_patch14_dinov2": "dinov2_vitg14",
            "vit_small_patch14_dinov2": "dinov2_vits14",
        }.get(timm_name, timm_name)
        model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=pretrained,
            verbose=False,
        )
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Extract patch token features (no CLS token).

        The forward pass is always wrapped in torch.no_grad() because
        this encoder is frozen and should never accumulate gradients.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Patch token features (B, N, C) where N = (H/patch_size)*(W/patch_size).
        """
        with torch.no_grad():
            features = self.backbone.forward_features(x)

        # timm returns either a Tensor or a dict depending on version
        if isinstance(features, dict):
            # timm >= 0.9.x with DINOv2: keys vary by version
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"]          # (B, N, C)
            if "patch_tokens" in features:
                return features["patch_tokens"]
            # Generic fallback: remove CLS from first tensor found
            tokens = next(v for v in features.values() if isinstance(v, Tensor))
            return tokens[:, 1:, :] if tokens.dim() == 3 else tokens
        else:
            # Shape: (B, N+1, C) — first token is CLS
            return features[:, 1:, :]                          # (B, N, C)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size
