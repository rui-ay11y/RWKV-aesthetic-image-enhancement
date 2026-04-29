"""ImageEnhancementPipeline: assembles Encoder → Bottleneck → Decoder."""
from __future__ import annotations

import inspect
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from src.models.encoders.base import BaseEncoder
from src.models.bottlenecks.base import BaseBottleneck
from src.models.decoders.base import BaseDecoder


class ImageEnhancementPipeline(nn.Module):
    """Full image color enhancement pipeline.

    Connects three components in a fixed order:
        [Frozen Encoder] → [Trainable Bottleneck] → [Trainable Decoder]

    The encoder backbone is typically frozen, but some encoder variants may
    expose small trainable fusion/adaptation heads. The optimizer therefore
    includes any encoder parameters that still have ``requires_grad=True``.

    Args:
        encoder: Pretrained, frozen encoder. Must implement BaseEncoder.
        bottleneck: Trainable context-modeling bottleneck. Must implement BaseBottleneck.
        decoder: Trainable image reconstruction decoder. Must implement BaseDecoder.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        bottleneck: BaseBottleneck,
        decoder: BaseDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        # Ensure encoder is frozen at construction time
        self.encoder.freeze()
        # ******** MOD START: Detect whether bottleneck forward accepts h,w ********
        self._bottleneck_accepts_hw = False
        try:
            sig = inspect.signature(self.bottleneck.forward)
            self._bottleneck_accepts_hw = ("h" in sig.parameters and "w" in sig.parameters)
        except (TypeError, ValueError):
            self._bottleneck_accepts_hw = False
        # ******** MOD END: Detect whether bottleneck forward accepts h,w ********

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Enhance input image.

        Args:
            x: Input image of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Enhanced image of shape (B, 3, H, W), values in [0, 1].
        """
        _, _, H, W = x.shape

        # Spatial grid dimensions used by the decoder
        h = H // self.encoder.patch_size
        w = W // self.encoder.patch_size

        # ── Encoder ──
        # [E1 MOD] Do not wrap the whole encoder in no_grad(). Some encoder
        # variants keep the backbone frozen internally but expose trainable
        # fusion/adaptation heads that must receive gradients.
        features = self.encoder(x)              # (B, N, C_enc)

        # ── Bottleneck (trainable) ──
        # ******** MOD START: Pass h,w to bottleneck only when supported ********
        if self._bottleneck_accepts_hw:
            features = self.bottleneck(features, h=h, w=w)  # (B, N, C_bot)
        else:
            features = self.bottleneck(features)            # (B, N, C_bot)
        # ******** MOD END: Pass h,w to bottleneck only when supported ********

        # ── Decoder (trainable) ──
        # Decoder is responsible for emitting output at (H, W); alignment is
        # handled internally (learnable align_refine after bilinear if needed).
        output = self.decoder(features, h, w, img=x)

        # Residual enhancement: add delta to original image.
        return (x + output).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only bottleneck and decoder parameters.

        Use this method to build the optimizer parameter group. Fully frozen
        encoders contribute zero parameters, while encoder-side trainable
        heads (for example, E1 fusion layers) are included automatically:

            optimizer = AdamW(pipeline.trainable_parameters(), lr=1e-4)
        """
        # [E1 MOD] Include any encoder parameters that remain trainable
        # (for example, E1's multi-layer fusion head). Fully frozen encoders
        # contribute zero parameters here, so baseline behavior is unchanged.
        yield from (p for p in self.encoder.parameters() if p.requires_grad)
        yield from self.bottleneck.parameters()
        yield from self.decoder.parameters()

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts for each component.

        Returns:
            Dict with keys 'encoder', 'bottleneck', 'decoder', 'trainable'.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "encoder":    _count(self.encoder),
            "bottleneck": _count(self.bottleneck),
            "decoder":    _count(self.decoder),
            "trainable":  trainable,
        }
