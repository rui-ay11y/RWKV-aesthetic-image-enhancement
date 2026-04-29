"""Checkpoint loading and saving utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def allow_trusted_checkpoint_loading() -> None:
    """Allow local Lightning checkpoints to load on PyTorch >= 2.6.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Lightning checkpoints in this project store an OmegaConf config inside
    ``hyper_parameters``, so best-checkpoint reloads during ``trainer.test``
    can fail unless we opt back into trusted full checkpoint loading.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights from a checkpoint file.

    Handles both raw state-dict files (.pth) and PyTorch Lightning
    checkpoint files (.ckpt) that nest weights under 'state_dict'.
    Automatically strips the 'model.' prefix added by Lightning.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        strict: If True, requires exact key match.

    Returns:
        Full checkpoint dict (contains 'state_dict', 'epoch', etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    allow_trusted_checkpoint_loading()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip 'pipeline.' or 'model.' prefix added by LightningModule
    stripped: dict[str, Any] = {}
    for k, v in state_dict.items():
        for prefix in ("pipeline.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        stripped[k] = v

    model.load_state_dict(stripped, strict=strict)
    return ckpt


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights to a checkpoint file.

    Args:
        model: Model whose state_dict to save.
        path: Output file path (.pth).
        extra: Optional extra metadata merged into the checkpoint dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
