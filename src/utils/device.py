"""Cross-platform device detection: CUDA > MPS > CPU."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority order: NVIDIA CUDA > Apple MPS > CPU.

    Returns:
        Best available torch.device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_accelerator() -> str:
    """Return the PyTorch Lightning Trainer accelerator string.

    Returns:
        One of 'gpu', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
