from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.bottlenecks.base import BaseBottleneck
from src.models.decoders.cnn_decoder import CNNDecoder
from src.models.decoders.base import BaseDecoder
from src.models.encoders.base import BaseEncoder
from src.models.pipeline import ImageEnhancementPipeline


class DummyEncoder(BaseEncoder):
    def __init__(self, patch_size: int = 14, embed_dim: int = 8) -> None:
        super().__init__()
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.proj(x)
        return feat.flatten(2).transpose(1, 2)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size


class DummyBottleneck(BaseBottleneck):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self._output_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class DummyDecoder(BaseDecoder):
    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        batch = x.shape[0]
        return torch.zeros(batch, 3, h * 16, w * 16, device=x.device, dtype=x.dtype)


class DummyResidualDecoder(BaseDecoder):
    def __init__(self, residual_value: float = 0.3) -> None:
        super().__init__()
        self.predict_residual = True
        self.residual_value = residual_value

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        batch = x.shape[0]
        return torch.full(
            (batch, 3, h * 16, w * 16),
            fill_value=self.residual_value,
            device=x.device,
            dtype=x.dtype,
        )


def test_pipeline_resizes_decoder_output_to_match_input() -> None:
    pipeline = ImageEnhancementPipeline(
        DummyEncoder(patch_size=14),
        DummyBottleneck(),
        DummyDecoder(),
    )

    inputs = torch.rand(2, 3, 256, 256)
    outputs = pipeline(inputs)

    assert tuple(outputs.shape) == tuple(inputs.shape)


def test_pipeline_adds_decoder_residual_to_input() -> None:
    pipeline = ImageEnhancementPipeline(
        DummyEncoder(patch_size=14),
        DummyBottleneck(),
        DummyResidualDecoder(residual_value=0.3),
    )

    inputs = torch.full((2, 3, 256, 256), 0.8)
    outputs = pipeline(inputs)

    assert tuple(outputs.shape) == tuple(inputs.shape)
    assert torch.allclose(outputs, torch.ones_like(outputs))


def test_cnn_decoder_zero_initializes_residual_head() -> None:
    decoder = CNNDecoder(
        input_dim=8,
        patch_size=14,
        num_upsample_blocks=4,
        base_channels=32,
        predict_residual=True,
    )

    tokens = torch.randn(2, 18 * 18, 8)
    outputs = decoder(tokens, 18, 18)

    assert torch.allclose(outputs, torch.zeros_like(outputs), atol=1e-6)
