"""Lightweight CNN decoder with PixelShuffle upsampling (NAFNet-style, multi-scale img skip)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.decoders.base import BaseDecoder


class ChanLayerNorm(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W).

    Permutes to (B, H, W, C), applies LayerNorm(C), permutes back.
    Unlike BatchNorm, statistics are computed per-sample with no batch
    dependency, which is critical for image-to-image restoration tasks.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1).float()).to(x.dtype).permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    """Two-layer residual conv block with GELU activation (no BN)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ChanLayerNorm(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class UpsampleBlock(nn.Module):
    """2× upsample via PixelShuffle + multi-scale img skip fusion.

    After upsampling, concatenates projected original-image features at the
    same spatial resolution and fuses them via a 1×1 conv, giving every
    scale direct access to pixel-level information from the input image.
    """

    def __init__(self, in_channels: int, out_channels: int, img_proj_channels: int = 3) -> None:
        super().__init__()
        self.norm = ChanLayerNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True)
        self.ps = nn.PixelShuffle(2)                     # (B, C*4, H, W) → (B, C, 2H, 2W)
        self.act = nn.GELU()
        # 1×1 fusion: merge upsampled features with projected img features
        self.img_fuse = nn.Conv2d(out_channels + img_proj_channels, out_channels, 1, bias=True)
        self.res = ResidualBlock(out_channels)

    def forward(self, x: Tensor, img_feat: Tensor) -> Tensor:
        x = self.act(self.ps(self.conv(self.norm(x))))   # (B, out_channels, 2H, 2W)
        x = self.img_fuse(torch.cat([x, img_feat], dim=1))
        return self.res(x)


class CNNDecoder(BaseDecoder):
    """Lightweight CNN decoder: token features → residual delta → enhanced image.

    Architecture:
      1. LayerNorm + Linear projection: (B, N, C) → (B, N, base_channels)
      2. Reshape to 2D feature map: (B, base_channels, h, w)
      3. Progressive 2× upsampling (PixelShuffle) with multi-scale image guidance:
           each UpsampleBlock fuses projected original-image features at its
           output resolution (image-guided, not encoder feature skip).
      4. Output head → residual delta via tanh
      5. Residual addition handled by the pipeline: enhanced = clamp(img + delta, 0, 1)

    Uses channel-wise LayerNorm throughout (no BatchNorm), following
    NAFNet/restoration conventions.

    Args:
        input_dim: Bottleneck output dimension (injected by pipeline).
        patch_size: Encoder patch size (stored for reference).
        num_upsample_blocks: Number of 2× PixelShuffle stages.
        base_channels: Channel count after initial projection.
        residual_scale: Maximum absolute residual magnitude (tanh output scale).
        img_proj_channels: Projected img channels fused at each upsample stage.
    """

    def __init__(
        self,
        input_dim: int,
        patch_size: int = 14,
        num_upsample_blocks: int = 4,
        base_channels: int = 64,
        residual_scale: float = 0.5,
        img_proj_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.residual_scale = residual_scale
        # Exposed as attribute so pipeline can detect residual mode
        self.predict_residual = True

        # Token projection
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, base_channels),
            nn.GELU(),
        )

        # Shared img projection applied at every scale
        self.img_proj = nn.Sequential(
            nn.Conv2d(3, img_proj_channels, 1, bias=True),
            ChanLayerNorm(img_proj_channels),
        )

        # Channel schedule: gentle decay (keep high-res stages thicker)
        # Default with base_channels=64: [64, 64, 48, 32, 24]
        _ratios = (1.0, 0.75, 0.5, 0.375)
        ch = [base_channels] + [
            max(round(base_channels * _ratios[min(i - 1, len(_ratios) - 1)]), 24)
            for i in range(1, num_upsample_blocks + 1)
        ]
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(ch[i], ch[i + 1], img_proj_channels)
            for i in range(num_upsample_blocks)
        ])

        # Output head → delta (intermediate dim tracks ch[-1], not a fixed 16)
        self.head = nn.Sequential(
            ChanLayerNorm(ch[-1]),
            nn.Conv2d(ch[-1], ch[-1], 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(ch[-1], 3, 1, bias=True),
        )
        # Zero-init: start from identity mapping (delta = 0)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # Learnable alignment refinement: corrects the mismatch when
        # num_upsample_blocks 2× stages don't evenly cover patch_size
        # (e.g. DINOv2 patch_size=14 → 4 stages → 16× vs 14× needed).
        # Operates on ch[-1]-dim feature map (before head), not on 3-channel
        # output, so correction has full feature-space capacity.
        # Applied as a residual; zero-init so training starts from bilinear.
        self.align_refine = nn.Conv2d(ch[-1], ch[-1], 3, padding=1, bias=True)
        nn.init.zeros_(self.align_refine.weight)
        nn.init.zeros_(self.align_refine.bias)

    def forward(self, x: Tensor, h: int, w: int, img: Tensor | None = None) -> Tensor:
        """Decode token features to a residual delta.

        Args:
            x: Token features (B, N, C) where N = h * w.
            h: Spatial grid height (H_img / patch_size).
            w: Spatial grid width  (W_img / patch_size).
            img: Original input image (B, 3, H, W) for multi-scale skip. Required.

        Returns:
            Residual delta (B, 3, H, W), strictly in [-residual_scale, residual_scale]
            (tanh is always the final operation, after any alignment step).
            The pipeline adds this to the original image and clamps to [0, 1].
        """
        assert img is not None, (
            "CNNDecoder requires the original image for multi-scale skip connections"
        )
        B = x.shape[0]
        _, _, H, W = img.shape

        # Project tokens and reshape to 2D feature map
        x = self.proj(x)                                            # (B, N, base_channels)
        x = x.transpose(1, 2).contiguous().reshape(B, -1, h, w)    # (B, base_channels, h, w)

        # Progressive upsampling with multi-scale img skip
        for i, block in enumerate(self.upsample_blocks):
            out_h = h * (2 ** (i + 1))
            out_w = w * (2 ** (i + 1))
            img_resized = F.interpolate(img, size=(out_h, out_w), mode="bilinear", align_corners=False)
            img_feat = self.img_proj(img_resized)
            x = block(x, img_feat)

        # Align feature map to original size before head, so align_refine
        # operates in full feature-space rather than 3-channel output space.
        # tanh is always the last operation → output is always strictly bounded.
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            x = x + self.align_refine(x)

        return torch.tanh(self.head(x)) * self.residual_scale
