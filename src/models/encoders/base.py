"""Abstract base class for all image encoders."""
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    """Abstract base class defining the encoder interface.

    All encoders receive (B, 3, H, W) images and output (B, N, C) token
    features where N = (H / patch_size) * (W / patch_size).
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode input image into patch token features.

        Args:
            x: Input image tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Token features of shape (B, N, C).
        """
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Output feature dimension C."""
        ...

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Patch size; determines decoder upsample ratio."""
        ...

    def freeze(self) -> None:
        """Freeze all parameters — no gradient will be computed."""
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "BaseEncoder":
        """Keep frozen encoder permanently in eval mode regardless of mode flag."""
        return super().train(False)
