#!/usr/bin/env python3
"""Figures generated from result artifacts. A figure is written only when its source exists.

Currently:
  results/figures/dataset_distribution.png  boxes per class per split, from results/dataset_validation_clean.json
Detector comparison and per-class AP figures are produced from results/detector_comparison.csv and
results/per_class_metrics.csv once detector runs are measured (not yet).

Usage:
    python evaluation/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dataset_distribution(validation_json: Path, out: Path) -> Path | None:
    if not validation_json.is_file():
        print(f"skipped dataset_distribution: {validation_json} not found")
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(validation_json.read_text())
    splits = list(report["splits"])
    classes = report["data_yaml_names"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        counts = [report["splits"][split]["boxes_per_class"].get(c, 0) for c in classes]
        xs = [j + i * width for j in range(len(classes))]
        ax.bar(xs, counts, width=width, label=f"{split} ({report['splits'][split]['images']} images)")
    ax.set_xticks([j + width * (len(splits) - 1) / 2 for j in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.set_ylabel("labelled boxes")
    ax.set_title(f"Class distribution: {report['dataset_root'].split('/')[-1]} (source: label files)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    results = PROJECT_ROOT / "results"
    made = dataset_distribution(results / "dataset_validation_clean.json",
                                results / "figures" / "dataset_distribution.png")
    if made:
        print(f"wrote {made.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
