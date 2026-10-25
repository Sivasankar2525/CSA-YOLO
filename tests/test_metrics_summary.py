"""Metrics summary builder: values come only from artifacts; missing artifacts stay NOT MEASURED.

These tests use synthetic files in a temporary project root. They verify the copying and
the NOT MEASURED logic. They are not experimental results.
"""

import json
from pathlib import Path

from evaluation.build_metrics_summary import EXPERIMENTS, NOT_MEASURED, build


def test_empty_project_reports_everything_not_measured(tmp_path: Path) -> None:
    summary = build(tmp_path)
    for name in EXPERIMENTS:
        entry = summary["detector"][name]
        assert entry["status"] == NOT_MEASURED
        assert all(v == NOT_MEASURED for v in entry["metrics"].values())
    assert summary["ablation"]["status"] == NOT_MEASURED
    for module in ("wpam", "tracking", "tsmm", "scrm"):
        assert summary[module]["metrics"] == NOT_MEASURED


def test_cram_is_marked_qualitative_not_validated(tmp_path: Path) -> None:
    cram = build(tmp_path)["cram"]
    assert cram["status"] == "QUALITATIVE CASE STUDIES ONLY"
    assert cram["metrics"] == NOT_MEASURED


def test_completed_run_summary_is_copied_verbatim(tmp_path: Path) -> None:
    run = tmp_path / "results" / "experiments" / "baseline_yolo11n"
    run.mkdir(parents=True)
    (run / "experiment_summary.json").write_text(json.dumps({
        "config_file": "configs/experiments/baseline_yolo11n.yaml",
        "effective_train": {"epochs": 1, "device": "cpu"},
        "metrics": {
            "metrics/precision(B)": 0.123,
            "metrics/recall(B)": 0.456,
            "metrics/mAP50(B)": 0.789,
            "metrics/mAP50-95(B)": 0.321,
        },
        "per_class_map50_95": {"person": 0.5},
    }))
    entry = build(tmp_path)["detector"]["baseline_yolo11n"]
    assert entry["status"] == "MEASURED"
    assert entry["metrics"] == {"precision": 0.123, "recall": 0.456, "mAP@0.5": 0.789, "mAP@0.5:0.95": 0.321}
    assert entry["per_class_mAP50_95"] == {"person": 0.5}
    assert entry["effective_train"] == {"epochs": 1, "device": "cpu"}


def test_summary_without_metrics_stays_not_measured(tmp_path: Path) -> None:
    run = tmp_path / "results" / "experiments" / "yolo11n_scratch"
    run.mkdir(parents=True)
    (run / "experiment_summary.json").write_text(json.dumps({"config_file": "x", "metrics": {}}))
    entry = build(tmp_path)["detector"]["yolo11n_scratch"]
    assert entry["status"] == NOT_MEASURED
    assert entry["metrics"]["mAP@0.5"] == NOT_MEASURED


def test_missing_metric_key_is_not_measured_not_zero(tmp_path: Path) -> None:
    run = tmp_path / "results" / "experiments" / "baseline_yolo11n"
    run.mkdir(parents=True)
    (run / "experiment_summary.json").write_text(json.dumps({
        "config_file": "x", "metrics": {"metrics/precision(B)": 0.5},
    }))
    metrics = build(tmp_path)["detector"]["baseline_yolo11n"]["metrics"]
    assert metrics["precision"] == 0.5
    assert metrics["mAP@0.5"] == NOT_MEASURED


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
