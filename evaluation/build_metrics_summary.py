#!/usr/bin/env python3
"""Build ``results/metrics_summary.json``, the single source of truth for reported metrics.

Every number in the output is copied from an artifact that exists on disk. If an artifact
is missing, the field is written as ``NOT MEASURED`` with the path that was searched.
Nothing is estimated, defaulted, or copied from documentation.

Artifacts read:
  results/experiments/<name>/experiment_summary.json   detector run summary (training launcher)
  results/experiments/<name>/results.csv                per-epoch training curves (Ultralytics)
  results/experiments/<name>/confusion_matrix.png      (Ultralytics plots, if produced)
  results/ablation/summary.json                        ablation run summary
  annotations/<module>.json                            human ground truth (see docs/EVALUATION.md)

Usage:
    python evaluation/build_metrics_summary.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
EXPERIMENTS = ("baseline_yolo11n", "yolo11n_scratch", "icsd_restored_scratch")
NOT_MEASURED = "NOT MEASURED"

# Ultralytics results_dict keys -> summary keys.
DETECTOR_KEYS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP@0.5",
    "metrics/mAP50-95(B)": "mAP@0.5:0.95",
}


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def detector_entry(name: str, project_root: Path) -> dict[str, Any]:
    run_dir = project_root / "results" / "experiments" / name
    summary_path = run_dir / "experiment_summary.json"
    artifacts = {
        "experiment_summary": str(summary_path.relative_to(project_root)) if summary_path.is_file() else NOT_MEASURED,
        "training_curves": str((run_dir / "results.csv").relative_to(project_root))
        if (run_dir / "results.csv").is_file() else NOT_MEASURED,
        "confusion_matrix": str((run_dir / "confusion_matrix.png").relative_to(project_root))
        if (run_dir / "confusion_matrix.png").is_file() else NOT_MEASURED,
    }
    summary = _read_json(summary_path)
    if summary is None or not summary.get("metrics"):
        return {
            "status": NOT_MEASURED,
            "reason": "no completed training run summary found" if summary is None
            else "summary contains no validation metrics",
            "metrics": {k: NOT_MEASURED for k in DETECTOR_KEYS.values()},
            "per_class_mAP50_95": NOT_MEASURED,
            "artifacts": artifacts,
        }
    raw = summary["metrics"]
    metrics = {out: float(raw[src]) if src in raw else NOT_MEASURED for src, out in DETECTOR_KEYS.items()}
    per_class = summary.get("per_class_map50_95") or NOT_MEASURED
    return {
        "status": "MEASURED",
        "config": summary.get("config_file"),
        "dataset_yaml": summary.get("resolved_dataset_yaml") or summary.get("dataset_yaml"),
        "effective_train": summary.get("effective_train"),
        "wall_seconds": summary.get("wall_seconds"),
        "metrics": metrics,
        "per_class_mAP50_95": per_class,
        "artifacts": artifacts,
    }


def ablation_entry(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "ablation" / "summary.json"
    data = _read_json(path)
    if data is None:
        return {
            "status": NOT_MEASURED,
            "reason": "no ablation run found; execute evaluation/run_ablation.py with a trained detector",
            "source": str(path.relative_to(project_root)),
            "configurations": NOT_MEASURED,
        }
    return {
        "status": "OPERATIONAL COUNTS ONLY (no ground-truth metrics)",
        "source": str(path.relative_to(project_root)),
        "run": data.get("run"),
        "configurations": data.get("configurations"),
    }


def ground_truth_entry(module: str, project_root: Path) -> dict[str, Any]:
    path = project_root / "annotations" / f"{module}.json"
    if not path.is_file():
        return {
            "status": NOT_MEASURED,
            "reason": f"human ground truth not found at annotations/{module}.json",
            "metrics": NOT_MEASURED,
        }
    return {
        "status": "GROUND TRUTH PRESENT; metrics require an evaluation run (not yet executed)",
        "ground_truth": f"annotations/{module}.json",
        "metrics": NOT_MEASURED,
    }


def build(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "_note": (
            "Single source of truth. Each value is copied from an artifact on disk; "
            "missing artifacts are NOT MEASURED. Validation status: NOT EXPERIMENTALLY VALIDATED."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": {name: detector_entry(name, project_root) for name in EXPERIMENTS},
        "ablation": ablation_entry(project_root),
        "wpam": ground_truth_entry("wpam", project_root),
        "tracking": ground_truth_entry("tracking", project_root),
        "tsmm": ground_truth_entry("tsmm", project_root),
        "scrm": ground_truth_entry("scrm", project_root),
        "cram": {
            "status": "QUALITATIVE CASE STUDIES ONLY",
            "reason": "rule-based prototype; no ground-truth risk labels exist",
            "metrics": NOT_MEASURED,
        },
    }


def main() -> int:
    summary = build()
    out = RESULTS / "metrics_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    measured = [k for k, v in summary["detector"].items() if v["status"] == "MEASURED"]
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")
    print(f"detector runs measured: {measured or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
