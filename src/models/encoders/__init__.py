__all__ = [
    "BaseEncoder",
    "DINOv2Encoder",
    "DINOv3Encoder",
    "DINOv3MultiLayerEncoder",
    "VRWKVEncoder",
]


def __getattr__(name: str):
    if name == "BaseEncoder":
        from .base import BaseEncoder
        return BaseEncoder
    if name == "DINOv2Encoder":
        from .dinov2 import DINOv2Encoder
        return DINOv2Encoder
    if name == "DINOv3Encoder":
        from .dinov3 import DINOv3Encoder
        return DINOv3Encoder
    if name == "DINOv3MultiLayerEncoder":
        # [E1 MOD] Export the multi-layer fusion encoder used in E1 experiments.
        from .dinov3_multilayer import DINOv3MultiLayerEncoder
        return DINOv3MultiLayerEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
