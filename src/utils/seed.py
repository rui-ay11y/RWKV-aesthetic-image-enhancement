"""Global random seed utilities for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for deterministic training.

    Covers Python built-in random, NumPy, PyTorch CPU and all CUDA devices,
    and PYTHONHASHSEED environment variable.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
