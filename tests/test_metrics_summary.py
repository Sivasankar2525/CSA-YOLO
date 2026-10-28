"""Metrics summary builder: values come only from artifacts; missing artifacts stay NOT MEASURED.

These tests use synthetic files in a temporary project root. They verify the copying and
the NOT MEASURED logic. They are not experimental results.
"""

import json
from pathlib import Path

from evaluation.build_metrics_summary import EXPERIMENTS, NOT_MEASURED, build


def _run_dir(root: Path, name: str) -> Path:
    d = root / "results" / "experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _split_json(precision=0.1, recall=0.2, map50=0.3, map5095=0.4) -> dict:
    return {
        "precision": precision, "recall": recall, "mAP@0.5": map50, "mAP@0.5:0.95": map5095,
        "F1": None, "per_class_precision": {"person": 0.5}, "per_class_recall": None,
        "per_class_f1": None, "per_class_AP50": {"person": 0.6}, "per_class_AP50-95": {"person": 0.4},
        "unavailable": ["F1"], "evaluation_settings": {"conf": 0.001, "iou": 0.7},
    }


def test_empty_project_reports_everything_not_measured(tmp_path: Path) -> None:
    summary = build(tmp_path)
    for name in EXPERIMENTS:
        entry = summary["detector"][name]
        assert entry["status"] == NOT_MEASURED
        assert entry["test"]["metrics"]["mAP@0.5"] == NOT_MEASURED
        assert entry["validation"]["metrics"]["mAP@0.5"] == NOT_MEASURED
    assert summary["ablation"]["status"] == NOT_MEASURED
    for module in ("wpam", "tracking", "tsmm", "scrm"):
        assert summary[module]["metrics"] == NOT_MEASURED


def test_cram_is_marked_qualitative_not_validated(tmp_path: Path) -> None:
    cram = build(tmp_path)["cram"]
    assert cram["status"] == "QUALITATIVE CASE STUDIES ONLY"
    assert cram["metrics"] == NOT_MEASURED


def test_test_split_metrics_are_copied_verbatim(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "baseline_yolo11n")
    (run / "test_metrics.json").write_text(json.dumps(_split_json(0.123, 0.456, 0.789, 0.321)))
    entry = build(tmp_path)["detector"]["baseline_yolo11n"]
    assert entry["status"] == "MEASURED"
    assert entry["reported_split"] == "test"
    assert entry["test"]["metrics"]["precision"] == 0.123
    assert entry["test"]["metrics"]["mAP@0.5:0.95"] == 0.321
    assert entry["test"]["per_class"]["AP50"] == {"person": 0.6}
    assert entry["test"]["per_class"]["F1"] == NOT_MEASURED  # absent in the source artifact


def test_training_time_validation_is_never_reported_as_test(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "baseline_yolo11n")
    (run / "experiment_summary.json").write_text(json.dumps({
        "config_file": "x", "metrics": {"metrics/mAP50(B)": 0.99},
    }))
    entry = build(tmp_path)["detector"]["baseline_yolo11n"]
    assert entry["status"] == NOT_MEASURED
    assert entry["test"]["metrics"]["mAP@0.5"] == NOT_MEASURED
    assert "no test-split evaluation" in entry["reason"]


def test_validation_is_reported_separately_from_test(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "yolo11n_scratch")
    (run / "val_metrics.json").write_text(json.dumps(_split_json(map50=0.11)))
    entry = build(tmp_path)["detector"]["yolo11n_scratch"]
    assert entry["validation"]["status"] == "MEASURED"
    assert entry["validation"]["metrics"]["mAP@0.5"] == 0.11
    assert entry["test"]["status"] == NOT_MEASURED


def test_missing_metric_key_is_not_measured_not_zero(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "baseline_yolo11n")
    partial = _split_json()
    partial["precision"] = None
    (run / "test_metrics.json").write_text(json.dumps(partial))
    metrics = build(tmp_path)["detector"]["baseline_yolo11n"]["test"]["metrics"]
    assert metrics["precision"] == NOT_MEASURED
    assert metrics["recall"] == 0.2


def test_ground_truth_presence_does_not_produce_metrics(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "wpam.json").write_text("{}")
    entry = build(tmp_path)["wpam"]
    assert entry["ground_truth"] == "annotations/wpam.json"
    assert entry["metrics"] == NOT_MEASURED


def test_ablation_summary_is_reported_as_counts_only(tmp_path: Path) -> None:
    abl = tmp_path / "results" / "ablation"
    abl.mkdir(parents=True)
    (abl / "summary.json").write_text(json.dumps({"run": {"seed": 42}, "configurations": {"detector_only": {}}}))
    entry = build(tmp_path)["ablation"]
    assert "COUNTS ONLY" in entry["status"]
    assert entry["configurations"] == {"detector_only": {}}
