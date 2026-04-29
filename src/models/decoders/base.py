"""Abstract base class for all decoder modules."""
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for image decoders.

    Decoders take (B, N, C) token features and the spatial grid dimensions,
    then produce (B, 3, H, W) RGB outputs.

    The concrete output semantics depend on the decoder:
    - a full enhanced image in [0, 1], or
    - a residual delta that the pipeline adds back to the input image
    """

    @abstractmethod
    def forward(self, x: Tensor, h: int, w: int, img: Tensor | None = None) -> Tensor:
        """Decode token features into an output image.

        Args:
            x: Token features of shape (B, N, C).
            h: Spatial grid height — H_img / patch_size.
            w: Spatial grid width  — W_img / patch_size.
            img: Optional original input image (B, 3, H, W) for pixel-level skip connection.

        Returns:
            Decoder output tensor (B, 3, H_img, W_img).
        """
        ...
