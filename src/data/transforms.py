"""Training and validation image transforms."""
from __future__ import annotations

import torch
import torchvision.transforms as T
from torch import Tensor


_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


def get_train_transforms(
    image_size: int = 480,
    crop_size: int = 256,
    normalize_to_neg_one_one: bool = True,
) -> T.Compose:
    """Build training-time transform pipeline with augmentation.

    Pipeline: Resize → RandomCrop → RandomHorizontalFlip → ToTensor → Normalize.
    Normalisation maps [0, 1] → [-1, 1] (mean=0.5, std=0.5 per channel).

    Args:
        image_size: Resize shorter edge to this size before cropping.
        crop_size: Random crop size applied after resize.
        normalize_to_neg_one_one: Map tensors from [0, 1] to [-1, 1].

    Returns:
        Composed torchvision transform.
    """
    transforms = [
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ]
    if normalize_to_neg_one_one:
        transforms.append(T.Normalize(mean=_NORM_MEAN, std=_NORM_STD))
    return T.Compose(transforms)


def get_val_transforms(
    image_size: int = 256,
    normalize_to_neg_one_one: bool = True,
) -> T.Compose:
    """Build validation/test transform pipeline (no augmentation).

    Pipeline: Resize to square → ToTensor → Normalize.

    Args:
        image_size: Target square resolution.
        normalize_to_neg_one_one: Map tensors from [0, 1] to [-1, 1].

    Returns:
        Composed torchvision transform.
    """
    transforms = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ]
    if normalize_to_neg_one_one:
        transforms.append(T.Normalize(mean=_NORM_MEAN, std=_NORM_STD))
    return T.Compose(transforms)


def denormalize(x: Tensor) -> Tensor:
    """Reverse the [-1, 1] normalisation back to [0, 1].

    Args:
        x: Normalised tensor (any shape).

    Returns:
        Tensor with values clamped to [0, 1].
    """
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def to_image_range(x: Tensor, normalized_to_neg_one_one: bool = True) -> Tensor:
    """Convert a loader tensor to [0, 1] using explicit config, not heuristics."""
    return denormalize(x) if normalized_to_neg_one_one else x.clamp(0.0, 1.0)