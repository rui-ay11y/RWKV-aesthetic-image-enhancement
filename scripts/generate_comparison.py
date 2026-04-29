"""Generate qualitative comparison figures from a trained checkpoint.

Produces input | pred | GT side-by-side grids saved as PNG files.

Usage::

    python scripts/generate_comparison.py \\
        --checkpoint outputs/checkpoints/b3_rwkv_epoch050-24.30.ckpt \\
        --config     outputs/logs/b3_rwkv/version_0/config/config.yaml \\
        --num-batches 4 --num-samples 8
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
from src.evaluation.visualization import save_comparison_grid
from src.data.transforms import to_image_range
from src.utils.device import get_device
from src.utils.checkpoint import allow_trusted_checkpoint_loading


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparison figures")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default="outputs/figures")
    p.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of test batches to visualise",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Images per batch grid",
    )
    p.add_argument("--device", default="auto", help="Device: auto/cuda/cpu/mps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    allow_trusted_checkpoint_loading()

    cfg = OmegaConf.load(args.config)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available in this interpreter.")

    # ── Load model ────────────────────────────────────────────────────
    model = ColorEnhanceLitModule.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        map_location=device,
    )
    model.eval().to(device)

    # ── Data ──────────────────────────────────────────────────────────
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    loader = datamodule.test_dataloader()
    inputs_are_normalized = bool(cfg.data.get("normalize_to_neg_one_one", True))

    out_dir = Path(args.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

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
            n = min(args.num_samples, inputs_01.shape[0])

            save_comparison_grid(
                inputs_01[:n].cpu(),
                preds[:n].cpu(),
                targets_01[:n].cpu(),
                save_path=out_dir / f"batch_{batch_idx:04d}.png",
                nrow=n,
            )
            print(f"Saved batch {batch_idx:04d} -> {out_dir}")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
