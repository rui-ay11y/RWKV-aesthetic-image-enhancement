"""Abstract base class for all bottleneck modules."""
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseBottleneck(nn.Module, ABC):
    """Abstract base class for all bottleneck modules.

    All bottlenecks receive (B, N, C_in) token features from the encoder
    and return (B, N, C_out) features for the decoder. The token count N
    must be preserved.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Process token features through the bottleneck.

        Args:
            x: Encoder output token features of shape (B, N, C_in).

        Returns:
            Processed token features of shape (B, N, C_out).
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output feature dimension C_out."""
        ...