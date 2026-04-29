from __future__ import annotations

import torch

from src.models.bottlenecks.rwkv_bottleneck import RWKVBottleneckV1


def test_rwkv_bottleneck_forward_and_backward() -> None:
    model = RWKVBottleneckV1(
        input_dim=768,
        hidden_dim=384,
        num_layers=2,
        drop_rate=0.0,
    )
    inputs = torch.randn(2, 37, 768, requires_grad=True)

    outputs = model(inputs)
    loss = outputs.mean()
    loss.backward()

    assert tuple(outputs.shape) == (2, 37, 384)
    assert model.proj_in.weight.grad is not None
    assert model.proj_out.weight.grad is not None
