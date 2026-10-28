#!/usr/bin/env python3
"""Collect experiment artifacts into the results CSV tables.

Every value comes from a file on disk. A missing artifact gives the cell ``NOT MEASURED``.
No value is estimated, defaulted or copied from documentation.

Outputs (under results/):
  detector_comparison.csv   one row per detector run (test split is the reported result)
  per_class_metrics.csv     one row per detector run and class (test split)
  wpam_metrics.csv          association metrics from results/eval/wpam.json
  tracking_metrics.csv      tracking metrics from results/eval/tracking.json
  tsmm_metrics.csv          temporal metrics from results/eval/tsmm.json
  scrm_metrics.csv          compliance metrics from results/eval/scrm.json
  cram_results.csv          qualitative scenario results from results/eval/cram_cases.json
  ablation_metrics.csv      ablation configurations (operational counts; no ground-truth metrics)
  figures/                  figures generated from the artifacts above (only when data exists)

The detector "best" flag follows a stated rule and is never a claim beyond the measured data:
the highest test mAP@0.5:0.95 among runs whose test metrics are MEASURED; ties are broken by
test mAP@0.5. If no run is measured, no run is flagged.

Usage:
    python evaluation/collect_results.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_metrics_summary import EXPERIMENTS, NOT_MEASURED  # noqa: E402

DETECTOR_FIELDS = [
    "experiment", "status", "initialization", "epochs_run", "image_size", "batch_size",
    "precision", "recall", "F1", "mAP@0.5", "mAP@0.5:0.95", "best_by_test_mAP50_95",
    "model_size_mb", "training_time_hours", "inference_latency_ms", "source",
]


def _read(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _cell(value: Any) -> Any:
    return NOT_MEASURED if value is None else value


def _rel(root: Path, path: Path) -> str:
    """Path relative to the project root that was passed in, or NOT MEASURED if absent."""
    return str(path.relative_to(root)) if path.exists() else NOT_MEASURED


def detector_rows(root: Path) -> tuple[list[dict], list[dict]]:
    """Return (comparison rows, per-class rows). Test split is the reported split."""
    comparison, per_class = [], []
    for name in EXPERIMENTS:
        run = root / "results" / "experiments" / name
        summary = _read(run / "experiment_summary.json") or {}
        test = _read(run / "test_metrics.json")
        manifest = _read(run / "manifest.json") or {}
        weights = run / "weights" / "best.pt"
        timings = summary.get("timings") or {}
        train_cfg = summary.get("effective_train") or {}
        row = {
            "experiment": name,
            "status": "MEASURED" if test else NOT_MEASURED,
            "initialization": manifest.get("initialization") or NOT_MEASURED,
            "epochs_run": NOT_MEASURED,  # recorded by Ultralytics results.csv, not the summary
            "image_size": _cell(train_cfg.get("imgsz")),
            "batch_size": _cell(train_cfg.get("batch")),
            "precision": _cell(test and test.get("precision")),
            "recall": _cell(test and test.get("recall")),
            "F1": _cell(test and test.get("F1")),
            "mAP@0.5": _cell(test and test.get("mAP@0.5")),
            "mAP@0.5:0.95": _cell(test and test.get("mAP@0.5:0.95")),
            "best_by_test_mAP50_95": "no",
            "model_size_mb": round(weights.stat().st_size / 1e6, 2) if weights.is_file() else NOT_MEASURED,
            "training_time_hours": (round(timings["train_wall_seconds"] / 3600, 3)
                                    if "train_wall_seconds" in timings else NOT_MEASURED),
            "inference_latency_ms": NOT_MEASURED,  # requires the documented benchmark (not run)
            "source": _rel(root, run / "test_metrics.json"),
        }
        comparison.append(row)
        if test:
            for cls, ap in (test.get("per_class_AP50-95") or {}).items():
                per_class.append({
                    "experiment": name, "class": cls,
                    "precision": _cell((test.get("per_class_precision") or {}).get(cls)),
                    "recall": _cell((test.get("per_class_recall") or {}).get(cls)),
                    "F1": _cell((test.get("per_class_f1") or {}).get(cls)),
                    "AP50": _cell((test.get("per_class_AP50") or {}).get(cls)),
                    "AP50-95": _cell(ap),
                    "source": _rel(root, run / "test_metrics.json"),
                })
    flag_best(comparison)
    return comparison, per_class


def flag_best(rows: list[dict]) -> None:
    measured = [r for r in rows if r["status"] == "MEASURED" and isinstance(r["mAP@0.5:0.95"], float)]
    if not measured:
        return
    best = max(measured, key=lambda r: (r["mAP@0.5:0.95"], r["mAP@0.5"] if isinstance(r["mAP@0.5"], float) else -1))
    best["best_by_test_mAP50_95"] = "yes"


def module_rows(root: Path, module: str, metric_names: list[str]) -> list[dict]:
    """One row per metric. Values come from results/eval/<module>.json if it exists."""
    data = _read(root / "results" / "eval" / f"{module}.json") or {}
    source = _rel(root, root / "results" / "eval" / f"{module}.json")
    return [{"metric": m, "value": _cell(data.get(m)), "source": source} for m in metric_names]


WPAM_METRICS = ["association_precision", "association_recall", "association_f1", "correct_association_rate",
                "ground_truth_images", "ground_truth_links"]
TRACKING_METRICS = ["id_consistency", "id_switches_total", "gt_workers", "gt_worker_observations"]
TSMM_METRICS = ["raw_transitions", "smoothed_transitions", "transition_reduction",
                "false_alerts_without_tsmm", "false_alerts_with_tsmm",
                "missed_violations_without_tsmm", "missed_violations_with_tsmm"]
SCRM_METRICS = ["accuracy", "precision", "recall", "f1", "coverage", "unknown_count", "evaluated_workers"]


def cram_rows(root: Path) -> list[dict]:
    data = _read(root / "results" / "eval" / "cram_cases.json")
    source = _rel(root, root / "results" / "eval" / "cram_cases.json")
    if not data or not data.get("cases"):
        return [{"scenario": NOT_MEASURED, "observed_condition": NOT_MEASURED, "risk_score": NOT_MEASURED,
                 "risk_level": NOT_MEASURED, "interpretation": NOT_MEASURED, "source": source,
                 "note": "rule-based prototype; qualitative case studies not yet run"}]
    return [{**c, "source": source, "note": "rule-based prototype; not a validated risk model"}
            for c in data["cases"]]


def ablation_rows(root: Path) -> list[dict]:
    data = _read(root / "results" / "ablation" / "summary.json")
    source = _rel(root, root / "results" / "ablation" / "summary.json")
    names = ["detector_only", "detector_wpam", "detector_wpam_tracking", "detector_wpam_tracking_tsmm",
             "detector_wpam_tracking_tsmm_scrm", "full_csa_yolo"]
    rows = []
    for name in names:
        cfg = ((data or {}).get("configurations") or {}).get(name)
        rows.append({
            "configuration": name,
            "status": "COUNTS ONLY" if cfg else NOT_MEASURED,
            "enabled_modules": ";".join(cfg["enabled_modules"]) if cfg else NOT_MEASURED,
            "images_processed": _cell(cfg and cfg.get("images_processed")),
            "detections_total": _cell(cfg and cfg.get("detections_total")),
            "associations_total": _cell(cfg and cfg.get("associations_total")),
            "distinct_track_ids": _cell(cfg and cfg.get("distinct_track_ids")),
            "mAP@0.5": NOT_MEASURED,  # per-configuration detector metric is not defined in this ablation
            "association_f1": NOT_MEASURED,
            "compliance_accuracy": NOT_MEASURED,
            "source": source,
        })
    return rows


def write_csv(path: Path, rows: list[dict], fields: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow({fields[0]: NOT_MEASURED})


def build(root: Path = PROJECT_ROOT) -> dict[str, Path]:
    results = root / "results"
    comparison, per_class = detector_rows(root)
    outputs = {
        "detector_comparison.csv": (comparison, DETECTOR_FIELDS),
        "per_class_metrics.csv": (per_class, ["experiment", "class", "precision", "recall", "F1",
                                              "AP50", "AP50-95", "source"]),
        "wpam_metrics.csv": (module_rows(root, "wpam", WPAM_METRICS), ["metric", "value", "source"]),
        "tracking_metrics.csv": (module_rows(root, "tracking", TRACKING_METRICS), ["metric", "value", "source"]),
        "tsmm_metrics.csv": (module_rows(root, "tsmm", TSMM_METRICS), ["metric", "value", "source"]),
        "scrm_metrics.csv": (module_rows(root, "scrm", SCRM_METRICS), ["metric", "value", "source"]),
        "cram_results.csv": (cram_rows(root), ["scenario", "observed_condition", "risk_score", "risk_level",
                                               "interpretation", "source", "note"]),
        "ablation_metrics.csv": (ablation_rows(root), None),
    }
    written = {}
    for name, (rows, fields) in outputs.items():
        path = results / name
        write_csv(path, rows, fields)
        written[name] = path
    return written


def main() -> int:
    written = build()
    for name, path in written.items():
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
