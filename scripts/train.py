"""Hydra main training entry point.

Usage examples::

    # Official Stage 5 baseline: DINOv2 encoder + CNN bottleneck + CNN decoder
    python scripts/train.py experiment_name=dino_cnn_cnn

    # Ablation: remove perceptual / SSIM terms
    python scripts/train.py ablation=a_loss_l1_only experiment_name=a_loss_l1

    # Debug run (2 epochs, tiny data, no early stop)
    python scripts/train.py training.max_epochs=2 data.batch_size=2 \\
        data.train_split=10 data.test_split=5 experiment_name=debug

    # Resume full training state from a saved checkpoint
    python scripts/train.py training.max_epochs=30 \\
        training.resume_from_checkpoint=outputs/checkpoints/model.ckpt

    # Load model weights from a checkpoint and continue with fresh optimiser
    python scripts/train.py training.max_epochs=10 \\
        training.init_from_checkpoint=outputs/checkpoints/model.ckpt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lit_module import ColorEnhanceLitModule
from src.utils.checkpoint import allow_trusted_checkpoint_loading, load_checkpoint
from src.utils.seed import set_seed
from src.utils.config_snapshot import save_config_snapshot
from src.utils.device import get_accelerator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Required for deterministic CUDA GEMM paths when Lightning sets
    # torch.use_deterministic_algorithms(True).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    allow_trusted_checkpoint_loading()

    # 0. Runtime device checks (fail-fast when GPU is required)
    accelerator = get_accelerator()
    require_gpu = bool(cfg.training.get("require_gpu", True))
    print(
        f"[device] python={sys.executable} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()} accelerator={accelerator}"
    )
    if require_gpu and accelerator != "gpu":
        raise RuntimeError(
            "GPU is required but not available. "
            "Select a CUDA-enabled interpreter (e.g., E:\\Anaconda\\python.exe) "
            "and verify torch.cuda.is_available() is True."
        )

    # 1. Reproducibility
    set_seed(cfg.seed)

    # 2. Persist full config to outputs/logs/{experiment_name}/config/
    save_config_snapshot(cfg)

    # 3. Model
    model = ColorEnhanceLitModule(cfg)
    init_ckpt = cfg.training.get("init_from_checkpoint")
    if init_ckpt:
        load_checkpoint(init_ckpt, model.pipeline, strict=False)

    # 4. Data
    datamodule = hydra.utils.instantiate(cfg.data)

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            filename=f"{cfg.experiment_name}_{{epoch:03d}}-{{val/psnr:.2f}}",
            verbose=True,
        ),
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 6. Logger (pure local CSV — no cloud dependency)
    csv_logger = CSVLogger(
        save_dir=cfg.logging.log_dir,
        name=cfg.experiment_name,
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # 8. Fit
    resume_ckpt = cfg.training.get("resume_from_checkpoint")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # 9. Test with the best checkpoint (also triggers on_test_end → JSON export)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
