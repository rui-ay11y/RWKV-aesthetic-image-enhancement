"""Config-driven combined weighted loss."""
from __future__ import annotations

import hydra
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss components.

    Components are fully specified in Hydra config. Each component must
    implement ``forward(pred, target) -> scalar Tensor`` and carry a
    ``weight`` attribute (used internally by its own forward).

    Example config (loss/stage1.yaml)::

        _target_: src.losses.combined_loss.CombinedLoss
        components:
          l1:
            _target_: src.losses.l1_loss.L1Loss
            weight: 1.0
          ssim:
            _target_: src.losses.ssim_loss.SSIMLoss
            weight: 0.1

    Args:
        components: OmegaConf DictConfig mapping names → instantiable
                    loss configs.
    """

    def __init__(self, components: DictConfig) -> None:
        super().__init__()
        self.loss_fns = nn.ModuleDict({
            name: cfg if isinstance(cfg, nn.Module) else hydra.utils.instantiate(cfg)
            for name, cfg in components.items()
            if cfg is not None and not str(name).startswith("_")
        })

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute sum of all weighted loss components.

        Args:
            pred:   Predicted image (B, 3, H, W).
            target: Ground truth image (B, 3, H, W).

        Returns:
            Total scalar loss.
        """
        total = pred.new_zeros(1).squeeze()
        for fn in self.loss_fns.values():
            weight = float(getattr(fn, "weight", 1.0))
            if weight <= 0.0:
                continue
            total = total + fn(pred, target)
        return total

    def component_losses(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute each component separately for logging.

        Args:
            pred:   Predicted image.
            target: Ground truth image.

        Returns:
            Dict mapping component names to scalar Tensors.
        """
        out: dict[str, Tensor] = {}
        for name, fn in self.loss_fns.items():
            weight = float(getattr(fn, "weight", 1.0))
            if weight <= 0.0:
                out[name] = pred.new_zeros(())
            else:
                out[name] = fn(pred, target)
        return out
