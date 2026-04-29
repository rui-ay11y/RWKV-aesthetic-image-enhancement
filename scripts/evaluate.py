"""Standalone evaluation script for a trained checkpoint.

Usage::

    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/dino_cnn_cnn_epoch050-24.30.ckpt \\
        --config     outputs/logs/dino_cnn_cnn/config/config.yaml \\
        --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lit_module import ColorEnhanceLitModule
from src.evaluation.metrics import MetricCollection
from src.evaluation.visualization import save_comparison_grid
from src.data.transforms import to_image_range
from src.utils.checkpoint import allow_trusted_checkpoint_loading
from src.utils.device import get_device
from src.utils.logging_utils import print_metrics_table, export_results_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config", required=True, help="Path to config.yaml snapshot")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--output-dir", default="outputs/eval")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-vis", type=int, default=8,
                   help="Number of images in the comparison grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    # ── Config ────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir

    # ── Device ────────────────────────────────────────────────────────
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available in this interpreter.")

    # ── Model ─────────────────────────────────────────────────────────
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location=device
    )
    model.eval().to(device)

    # ── Data ──────────────────────────────────────────────────────────
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()

    # ── Evaluation loop ───────────────────────────────────────────────
    metric_fn = MetricCollection(cfg)
    accum: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    out_dir = Path(args.output_dir) / cfg.experiment_name
    vis_saved = False
    inputs_are_normalized = bool(cfg.data.get("normalize_to_neg_one_one", True))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs_01 = to_image_range(
                inputs,
                normalized_to_neg_one_one=inputs_are_normalized,
            )
            targets_01 = to_image_range(
                targets,
                normalized_to_neg_one_one=inputs_are_normalized,
            )

            preds = model.pipeline(inputs_01)
            m = metric_fn(preds, targets_01)
            for k, v in m.items():
                accum[k].append(v.item())

            # Save comparison grid for the first batch
            if not vis_saved:
                n = min(args.num_vis, inputs_01.shape[0])
                save_comparison_grid(
                    inputs_01[:n].cpu(),
                    preds[:n].cpu(),
                    targets_01[:n].cpu(),
                    save_path=out_dir / "comparison.png",
                )
                vis_saved = True

    # ── Aggregate & report ────────────────────────────────────────────
    results = {k: sum(v) / len(v) for k, v in accum.items()}
    print_metrics_table(results, title=f"Evaluation — {cfg.experiment_name}")

    export_results_json(
        {"experiment": cfg.experiment_name, "metrics": results},
        out_dir / "eval_results.json",
    )


if __name__ == "__main__":
    main()
