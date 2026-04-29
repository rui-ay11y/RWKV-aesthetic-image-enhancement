"""PyTorch Lightning Module — wraps pipeline, loss, optimiser, metrics."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor

from src.models.pipeline import ImageEnhancementPipeline
from src.evaluation.metrics import MetricCollection
from src.evaluation.visualization import save_comparison_grid
from src.data.transforms import to_image_range


class ColorEnhanceLitModule(pl.LightningModule):
    """LightningModule for image colour enhancement training.

    Responsibilities:
    - Assemble pipeline (encoder + bottleneck + decoder) via Hydra instantiate
    - Compute combined loss on each batch
    - Track PSNR / SSIM / LPIPS metrics
    - Save periodic visualisation grids to outputs/figures/
    - Export JSON result summary to results/ on test end
    - Configure AdamW optimizer + cosine/step/plateau LR scheduler

    Args:
        cfg: Fully resolved Hydra DictConfig for the experiment.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        # ── Pipeline ──────────────────────────────────────────────────
        encoder_kwargs = {}
        if "dinov2" in str(cfg.encoder.get("_target_", "")).lower():
            encoder_kwargs["img_size"] = int(cfg.data.crop_size)

        encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
        bottleneck = hydra.utils.instantiate(
            cfg.bottleneck, input_dim=encoder.embed_dim
        )
        decoder = hydra.utils.instantiate(
            cfg.decoder,
            input_dim=bottleneck.output_dim,
            patch_size=encoder.patch_size,
        )
        self.pipeline = ImageEnhancementPipeline(encoder, bottleneck, decoder)

        # ── Loss ──────────────────────────────────────────────────────
        self.criterion = hydra.utils.instantiate(cfg.loss)

        # ── Metrics ───────────────────────────────────────────────────
        self.metrics = MetricCollection(cfg)
        self.inputs_are_normalized = bool(
            cfg.data.get("normalize_to_neg_one_one", True)
        )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _to_01(self, x: Tensor) -> Tensor:
        """Convert loader tensors to [0, 1] using explicit config."""
        return to_image_range(
            x,
            normalized_to_neg_one_one=self.inputs_are_normalized,
        )

    def _shared_step(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Run forward pass + loss for train/val/test steps.

        Returns:
            Tuple ``(loss, pred_01, target_01, component_losses)`` where all
            image tensors live in [0, 1].
        """
        input_img, target_img = batch
        input_01  = self._to_01(input_img)
        target_01 = self._to_01(target_img)
        pred      = self.pipeline(input_01)          # (B, 3, H, W) in [0, 1]

        component_losses: dict[str, Tensor] = {}
        if hasattr(self.criterion, "component_losses"):
            component_losses = self.criterion.component_losses(pred, target_01)
            loss = pred.new_zeros(())
            for value in component_losses.values():
                loss = loss + value
        else:
            loss = self.criterion(pred, target_01)

        return loss, pred, target_01, component_losses

    def _log_component_losses(
        self,
        prefix: str,
        component_losses: dict[str, Tensor],
        *,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        """Log individual weighted loss terms with a shared naming scheme."""
        if not component_losses:
            return

        self.log_dict(
            {
                f"{prefix}/loss_{name}": value
                for name, value in component_losses.items()
            },
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            sync_dist=True,
        )

    # ------------------------------------------------------------------
    # Lightning step hooks
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        loss, _, _, component_losses = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self._log_component_losses("train", component_losses, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        loss, pred, target, component_losses = self._shared_step(batch)
        m = self.metrics(pred, target)

        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        self._log_component_losses("val", component_losses, on_epoch=True)
        self.log_dict(
            {f"val/{k}": v for k, v in m.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Periodic qualitative visualisation (first batch only)
        save_every = self.cfg.evaluation.save_samples_every_n_epochs
        if batch_idx == 0 and (self.current_epoch % save_every == 0):
            n = min(
                self.cfg.evaluation.num_visualization_samples,
                pred.shape[0],
            )
            fig_dir = Path("outputs/figures") / self.cfg.experiment_name
            save_comparison_grid(
                self._to_01(batch[0])[:n].cpu(),
                pred[:n].cpu(),
                target[:n].cpu(),
                save_path=fig_dir / f"epoch_{self.current_epoch:04d}.png",
            )

    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        loss, pred, target, component_losses = self._shared_step(batch)
        m = self.metrics(pred, target)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self._log_component_losses("test", component_losses, on_epoch=True)
        self.log_dict(
            {f"test/{k}": v for k, v in m.items()},
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_end(self) -> None:
        """Export JSON result summary to results/ after test completes."""
        logged = self.trainer.logged_metrics
        results = {
            "experiment_name": self.cfg.experiment_name,
            "encoder":    self.cfg.encoder.get("name", "unknown"),
            "bottleneck": str(self.cfg.bottleneck.get("_target_", "unknown")),
            "metrics": {
                k: float(v)
                for k, v in logged.items()
                if k.startswith("test/")
            },
        }
        out_dir = Path(self.cfg.logging.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.cfg.experiment_name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Build AdamW optimizer + LR scheduler.

        Frozen encoders contribute zero trainable params. Encoder variants with
        small trainable heads (for example, E1 multi-layer fusion) are grouped
        separately when configured.
        """
        # [E1 MOD] Support separate learning rates for trainable encoder heads
        # vs. the inherited RWKV/decoder stack. This helps preserve the strong
        # baseline while letting the new E1 fusion layers adapt more quickly.
        base_lr = float(self.cfg.training.learning_rate)
        main_lr = self.cfg.training.get("main_learning_rate")
        encoder_lr = self.cfg.training.get("encoder_learning_rate")
        main_lr = float(main_lr) if main_lr is not None else base_lr
        encoder_lr = float(encoder_lr) if encoder_lr is not None else base_lr

        encoder_trainable = [
            p for p in self.pipeline.encoder.parameters() if p.requires_grad
        ]
        main_trainable = [
            *self.pipeline.bottleneck.parameters(),
            *self.pipeline.decoder.parameters(),
        ]

        if encoder_trainable:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": encoder_trainable,
                        "lr": encoder_lr,
                    },
                    {
                        "params": main_trainable,
                        "lr": main_lr,
                    },
                ],
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            trainable = list(self.pipeline.trainable_parameters())
            optimizer = torch.optim.AdamW(
                trainable,
                lr=base_lr,
                weight_decay=self.cfg.training.weight_decay,
            )

        scheduler_name = self.cfg.training.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.max_epochs,
                eta_min=1e-6,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.5
            )
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/psnr",
                "interval":  "epoch",
                "frequency": 1,
            },
        }
