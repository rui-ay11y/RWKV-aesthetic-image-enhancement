"""CSV training logger and Rich console output utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


class CSVTrainLogger:
    """Append-mode CSV logger for per-epoch training metrics.

    Columns written every epoch:
        epoch, train_loss, val_loss, psnr, ssim, lpips, lr

    Args:
        path: Path to the output CSV file.
    """

    FIELDNAMES = ["epoch", "train_loss", "val_loss", "psnr", "ssim", "lpips", "lr"]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists()

    def log(self, row: dict[str, Any]) -> None:
        """Append one row to the CSV.

        Args:
            row: Dict mapping column names to values.
                 Missing keys are written as empty strings.
        """
        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.FIELDNAMES, extrasaction="ignore"
            )
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def export_results_json(results: dict[str, Any], path: str | Path) -> None:
    """Export experiment results dict to a JSON file.

    Args:
        results: Arbitrary results dict (must be JSON-serialisable).
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"[green]Results saved → {path}[/green]")


def print_metrics_table(metrics: dict[str, float], title: str = "Metrics") -> None:
    """Print a Rich table of metric name → value pairs.

    Args:
        metrics: Dict mapping metric names to float values.
        title: Table title shown above the table.
    """
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for name, value in metrics.items():
        fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(name, fmt)
    console.print(table)
