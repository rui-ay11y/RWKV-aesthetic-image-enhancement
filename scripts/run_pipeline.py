"""Convenience runner for A/B/C/D workflow in VSCode.

Stages:
  A: train
  B: evaluate (latest checkpoint)
  C: generate qualitative comparisons
  D: export JSON results to LaTeX tables

Examples:
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage evaluate --experiment-name final_rwkv_2k
    python scripts/run_pipeline.py --stage all --experiment-name final_rwkv_2k
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _latest_checkpoint(experiment_name: str) -> Path:
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Search recursively because Lightning filenames that include monitored
    # metrics like "val/psnr" can create nested directories on Windows.
    all_ckpts = sorted(
        ckpt_dir.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_ckpts:
        raise FileNotFoundError(
            "No checkpoint found under outputs/checkpoints. Run training first."
        )

    if experiment_name:
        exp_ckpts = [p for p in all_ckpts if experiment_name in str(p)]
        if exp_ckpts:
            return exp_ckpts[0]

    return all_ckpts[0]

def _default_config_snapshot(experiment_name: str) -> Path:
    cfg = PROJECT_ROOT / "outputs" / "logs" / experiment_name / "config" / "config.yaml"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Config snapshot not found: {cfg}. Run training first."
        )
    return cfg


def stage_train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/train.py",
        f"experiment_name={args.experiment_name}",
        "bottleneck=rwkv",
        f"data.data_dir={args.data_dir.as_posix()}",
        "training.require_gpu=true",
        f"loss.components.l1.weight={args.w_l1}",
        f"loss.components.ssim.weight={args.w_ssim}",
        f"loss.components.perceptual.weight={args.w_perceptual}",
    ]
    if args.quick:
        cmd.extend(
            [
                "training.max_epochs=2",
                "data.train_subset_size=64",
                "data.val_subset_size=16",
                "data.test_subset_size=16",
                "data.batch_size=2",
            ]
        )
    _run(cmd)


def stage_evaluate(args: argparse.Namespace) -> None:
    ckpt = args.checkpoint or _latest_checkpoint(args.experiment_name)
    cfg = args.config or _default_config_snapshot(args.experiment_name)
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        str(ckpt),
        "--config",
        str(cfg),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.eval_output_dir),
        "--device",
        args.device,
    ]
    _run(cmd)


def stage_compare(args: argparse.Namespace) -> None:
    ckpt = args.checkpoint or _latest_checkpoint(args.experiment_name)
    cfg = args.config or _default_config_snapshot(args.experiment_name)
    cmd = [
        sys.executable,
        "scripts/generate_comparison.py",
        "--checkpoint",
        str(ckpt),
        "--config",
        str(cfg),
        "--output-dir",
        str(args.fig_output_dir),
        "--num-batches",
        str(args.num_batches),
        "--num-samples",
        str(args.num_samples),
        "--device",
        args.device,
    ]
    _run(cmd)


def stage_export(args: argparse.Namespace) -> None:
    _run(
        [
            sys.executable,
            "scripts/export_results.py",
            "--input",
            "results",
            "--output",
            str(args.table_output_dir),
        ]
    )

    eval_dir = args.eval_output_dir / args.experiment_name
    if eval_dir.exists():
        _run(
            [
                sys.executable,
                "scripts/export_results.py",
                "--input",
                str(eval_dir),
                "--output",
                str(args.table_output_dir),
            ]
        )
    else:
        print(f"[warn] Eval directory not found, skip export: {eval_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run train/eval/compare/export workflow")
    p.add_argument(
        "--stage",
        choices=["train", "evaluate", "compare", "export", "all"],
        default="all",
    )
    p.add_argument(
        "--experiment-name",
        default="final_rwkv_2k",
        help="Experiment name used for checkpoints/log snapshots.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:/USYD/usyd-26s1/5703_capstone/datasets/archive"),
        help="Dataset root directory (must contain raw/, c/, splits/).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config snapshot path for evaluation/comparison.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path for evaluation/comparison.",
    )
    p.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Evaluation output root.",
    )
    p.add_argument(
        "--fig-output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Comparison figure output root.",
    )
    p.add_argument(
        "--table-output-dir",
        type=Path,
        default=Path("paper/tables"),
        help="LaTeX table export output directory.",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "auto", "cpu", "mps"],
        default="cuda",
        help="Execution device for evaluate/compare; train is forced to GPU.",
    )
    p.add_argument("--num-batches", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--w-l1", type=float, default=1.0, help="L1 loss weight.")
    p.add_argument("--w-ssim", type=float, default=0.1, help="SSIM loss weight.")
    p.add_argument(
        "--w-perceptual",
        type=float,
        default=0.01,
        help="Perceptual loss weight.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick sanity train mode (small subsets, 2 epochs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"train", "all"}:
        stage_train(args)
    if args.stage in {"evaluate", "all"}:
        stage_evaluate(args)
    if args.stage in {"compare", "all"}:
        stage_compare(args)
    if args.stage in {"export", "all"}:
        stage_export(args)


if __name__ == "__main__":
    main()

