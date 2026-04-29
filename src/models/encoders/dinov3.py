"""Frozen DINOv3 ViT-B encoder wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.models.encoders.base import BaseEncoder


class DINOv3Encoder(BaseEncoder):
    """Frozen DINOv3 ViT-B image encoder.

    Loads a pretrained DINOv3 ViT-B model via HuggingFace Transformers and
    extracts patch token features. CLS token and register tokens are discarded.
    Always kept in eval mode. Drop-in replacement inside ImageEnhancementPipeline.

    Args:
        model_name: HuggingFace model ID.
        frozen:     Whether to freeze all parameters (default True).
        patch_size: Patch size of the backbone (16 for DINOv3 ViT-B).
        embed_dim:  Output feature dimension (768 for DINOv3 ViT-B).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frozen: bool = True,
        patch_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim  = embed_dim

        self.backbone = AutoModel.from_pretrained(model_name)
        self.num_register_tokens = self.backbone.config.num_register_tokens
        
        # [MOD] DINOv3 expects ImageNet-style normalized inputs.
        # Keep buffers non-persistent to avoid polluting checkpoints.
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        if frozen:
            self.freeze()

    def forward(self, x: Tensor) -> Tensor:
        """Extract patch token features (no CLS or register tokens).

        Args:
            x: Input image (B, 3, H, W), values in [0, 1].
               Note: DINOv3 expects ImageNet-normalised input. If your
               pipeline already normalises, set normalise=False in the
               data transforms.

        Returns:
            Patch token features (B, N, C) where N = (H/patch_size)*(W/patch_size).
        """
        # [MOD] Convert to the exact input distribution expected by DINOv3.
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)

        # last_hidden_state: (B, 1 + num_register_tokens + N, C)
        # index 0          → CLS token
        # index 1 : 1+R    → register tokens
        # index 1+R :      → patch tokens
        R = self.num_register_tokens
        return outputs.last_hidden_state[:, 1 + R:, :]   # (B, N, C)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size