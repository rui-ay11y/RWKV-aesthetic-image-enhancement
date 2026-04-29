"""Qualitative comparison visualisation: input | pred | GT grid."""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision.utils as vutils
from torch import Tensor


def make_comparison_grid(
    inputs: Tensor,
    preds: Tensor,
    targets: Tensor,
    nrow: int = 4,
) -> Tensor:
    """Create a side-by-side comparison grid: [input | pred | GT].

    Each triplet occupies one column-group of 3 images. Images are
    laid out left-to-right, nrow triplets per visual row.

    Args:
        inputs:  Input images  (B, 3, H, W) in [0, 1].
        preds:   Predicted images (B, 3, H, W) in [0, 1].
        targets: Ground-truth images (B, 3, H, W) in [0, 1].
        nrow:    Number of triplets shown per row.

    Returns:
        Grid tensor (3, H_grid, W_grid) in [0, 1].
    """
    n = min(inputs.shape[0], nrow)
    tiles = []
    for i in range(n):
        tiles.extend([inputs[i], preds[i], targets[i]])
    return vutils.make_grid(torch.stack(tiles), nrow=3, padding=2, normalize=False)


def save_comparison_grid(
    inputs: Tensor,
    preds: Tensor,
    targets: Tensor,
    save_path: str | Path,
    nrow: int = 4,
) -> None:
    """Save a comparison grid image to disk.

    Parent directories are created automatically.

    Args:
        inputs:    Input images  (B, 3, H, W) in [0, 1].
        preds:     Predicted images (B, 3, H, W) in [0, 1].
        targets:   Ground-truth images (B, 3, H, W) in [0, 1].
        save_path: Output file path (.png recommended).
        nrow:      Number of triplets per row.
    """
    from PIL import Image

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_comparison_grid(inputs, preds, targets, nrow=nrow)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(save_path)
