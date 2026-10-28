#!/usr/bin/env python3
"""Export training curves for one experiment from its Ultralytics ``results.csv``.

Reads only the file written by the training run. Columns that are absent are skipped, and
a missing file gives a clear error instead of an empty figure.

Usage:
    python evaluation/plot_training.py --run results/experiments/baseline_yolo11n
Writes:
    <run>/figures/training_curves.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

LOSS_COLUMNS = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
METRIC_COLUMNS = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


class PlotError(RuntimeError):
    pass


def read_results_csv(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        raise PlotError(f"results.csv not found: {path}. Run training first (it is written by Ultralytics).")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise PlotError(f"results.csv has no rows: {path}")
    data: dict[str, list[float]] = {"epoch": []}
    for row in rows:
        data["epoch"].append(float(row.get("epoch", len(data["epoch"]))))
        for col, value in row.items():
            if col is None:
                continue
            col = col.strip()
            if col == "epoch" or value in (None, ""):
                continue
            try:
                data.setdefault(col, []).append(float(value))
            except ValueError:
                continue
    return data


def plot(run_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_results_csv(run_dir / "results.csv")
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))
    for col in LOSS_COLUMNS:
        if col in data and len(data[col]) == len(data["epoch"]):
            ax_loss.plot(data["epoch"], data[col], label=col)
    for col in METRIC_COLUMNS:
        if col in data and len(data[col]) == len(data["epoch"]):
            ax_metric.plot(data["epoch"], data[col], label=col)
    ax_loss.set(title="Losses", xlabel="epoch", ylabel="loss")
    ax_metric.set(title="Validation metrics (training-time)", xlabel="epoch", ylabel="value", ylim=(0, 1))
    for ax in (ax_loss, ax_metric):
        if ax.get_lines():
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{run_dir.name}: training curves (source: results.csv)")
    fig.tight_layout()
    out = run_dir / "figures" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        out = plot(args.run)
    except PlotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
