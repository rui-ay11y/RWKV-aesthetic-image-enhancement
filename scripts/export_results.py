"""Export JSON result files to LaTeX booktabs tables.

Reads all JSON files in the results/ directory and writes a .tex table
file for each into paper/tables/.

Usage::

    python scripts/export_results.py
    python scripts/export_results.py --input results/ --output paper/tables/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export JSON results to LaTeX tables")
    p.add_argument("--input",  default="results/",      help="Input results directory")
    p.add_argument("--output", default="paper/tables/", help="Output tables directory")
    return p.parse_args()


def results_to_latex(
    rows: list[dict],
    caption: str,
    label: str,
) -> str:
    """Convert a list of result dicts to a LaTeX booktabs table string.

    Each dict must have an 'experiment' key; remaining keys become columns.

    Args:
        rows:    List of result dicts.
        caption: Table caption.
        label:   LaTeX \\label value.

    Returns:
        LaTeX table string.
    """
    if not rows:
        return f"% No results yet for: {caption}\n"

    metric_keys = [k for k in rows[0] if k != "experiment"]
    col_spec = "l" + "r" * len(metric_keys)
    header = " & ".join(["Experiment"] + [k.upper() for k in metric_keys]) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for row in rows:
        cells = [row.get("experiment", "—")]
        for k in metric_keys:
            v = row.get(k, "—")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {in_dir}")
        return

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        rows = data.get("results", [])
        # Also handle single-experiment result files (from on_test_end)
        if not rows and "metrics" in data:
            row = {"experiment": data.get("experiment_name", json_path.stem)}
            row.update(data["metrics"])
            rows = [row]

        table = results_to_latex(
            rows,
            caption=data.get("description", json_path.stem.replace("_", " ").title()),
            label=f"tab:{json_path.stem}",
        )

        out_path = out_dir / f"{json_path.stem}.tex"
        out_path.write_text(table)
        print(f"  {json_path.name} → {out_path}")

    print(f"\nExported {len(json_files)} table(s) to {out_dir}")


if __name__ == "__main__":
    main()
