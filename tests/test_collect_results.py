"""Results collector: every missing artifact becomes NOT MEASURED; the best-run rule is explicit.

Synthetic artifacts in a temporary project root. These test the table logic, not any result.
"""

import csv
import json
from pathlib import Path

from evaluation.build_metrics_summary import NOT_MEASURED
from evaluation.collect_results import build


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _test_metrics(map50_95: float, map50: float = 0.5) -> dict:
    return {"precision": 0.6, "recall": 0.5, "F1": None, "mAP@0.5": map50, "mAP@0.5:0.95": map50_95,
            "per_class_AP50-95": {"person": 0.7, "helmet": 0.3}, "per_class_AP50": {"person": 0.8, "helmet": 0.4},
            "per_class_precision": None, "per_class_recall": None, "per_class_f1": None}


def _run(root: Path, name: str, test: dict | None) -> None:
    run = root / "results" / "experiments" / name
    run.mkdir(parents=True, exist_ok=True)
    if test is not None:
        (run / "test_metrics.json").write_text(json.dumps(test))


def test_empty_project_is_all_not_measured(tmp_path: Path) -> None:
    written = build(tmp_path)
    rows = _read_csv(written["detector_comparison.csv"])
    assert len(rows) == 3
    assert all(r["status"] == NOT_MEASURED and r["mAP@0.5:0.95"] == NOT_MEASURED for r in rows)
    assert all(r["best_by_test_mAP50_95"] == "no" for r in rows)
    assert all(r["inference_latency_ms"] == NOT_MEASURED for r in rows)
    assert _read_csv(written["per_class_metrics.csv"])[0]["experiment"] == NOT_MEASURED
    assert _read_csv(written["wpam_metrics.csv"])[0]["value"] == NOT_MEASURED
    assert all(r["value"] == NOT_MEASURED for r in _read_csv(written["tracking_metrics.csv"]))
    assert _read_csv(written["cram_results.csv"])[0]["scenario"] == NOT_MEASURED


def test_best_is_flagged_only_among_measured_runs(tmp_path: Path) -> None:
    _run(tmp_path, "baseline_yolo11n", _test_metrics(0.40))
    _run(tmp_path, "yolo11n_scratch", _test_metrics(0.55))
    _run(tmp_path, "icsd_restored_scratch", None)
    rows = {r["experiment"]: r for r in _read_csv(build(tmp_path)["detector_comparison.csv"])}
    assert rows["yolo11n_scratch"]["best_by_test_mAP50_95"] == "yes"
    assert rows["baseline_yolo11n"]["best_by_test_mAP50_95"] == "no"
    assert rows["icsd_restored_scratch"]["status"] == NOT_MEASURED


def test_tie_on_map50_95_is_broken_by_map50(tmp_path: Path) -> None:
    _run(tmp_path, "baseline_yolo11n", _test_metrics(0.5, map50=0.7))
    _run(tmp_path, "yolo11n_scratch", _test_metrics(0.5, map50=0.6))
    rows = {r["experiment"]: r for r in _read_csv(build(tmp_path)["detector_comparison.csv"])}
    assert rows["baseline_yolo11n"]["best_by_test_mAP50_95"] == "yes"
    assert rows["yolo11n_scratch"]["best_by_test_mAP50_95"] == "no"


def test_per_class_rows_come_from_measured_runs_only(tmp_path: Path) -> None:
    _run(tmp_path, "baseline_yolo11n", _test_metrics(0.4))
    rows = _read_csv(build(tmp_path)["per_class_metrics.csv"])
    assert {r["class"] for r in rows} == {"person", "helmet"}
    person = next(r for r in rows if r["class"] == "person")
    assert person["AP50-95"] == "0.7"
    assert person["precision"] == NOT_MEASURED  # per-class precision was absent


def test_module_values_come_from_eval_files_only(tmp_path: Path) -> None:
    (tmp_path / "results" / "eval").mkdir(parents=True)
    (tmp_path / "results" / "eval" / "wpam.json").write_text(json.dumps({"association_precision": 0.9}))
    rows = {r["metric"]: r["value"] for r in _read_csv(build(tmp_path)["wpam_metrics.csv"])}
    assert rows["association_precision"] == "0.9"
    assert rows["association_recall"] == NOT_MEASURED


def test_cram_cases_are_written_with_prototype_note(tmp_path: Path) -> None:
    (tmp_path / "results" / "eval").mkdir(parents=True)
    (tmp_path / "results" / "eval" / "cram_cases.json").write_text(json.dumps({"cases": [{
        "scenario": "missing helmet", "observed_condition": "no helmet for 6 frames",
        "risk_score": 25.0, "risk_level": "MEDIUM", "interpretation": "single PPE item missing"}]}))
    rows = _read_csv(build(tmp_path)["cram_results.csv"])
    assert rows[0]["scenario"] == "missing helmet"
    assert "prototype" in rows[0]["note"]


def test_ablation_counts_are_reported_without_accuracy(tmp_path: Path) -> None:
    (tmp_path / "results" / "ablation").mkdir(parents=True)
    (tmp_path / "results" / "ablation" / "summary.json").write_text(json.dumps({"configurations": {
        "detector_only": {"enabled_modules": [], "images_processed": 5, "detections_total": 22,
                          "associations_total": 0, "distinct_track_ids": None}}}))
    rows = {r["configuration"]: r for r in _read_csv(build(tmp_path)["ablation_metrics.csv"])}
    assert rows["detector_only"]["images_processed"] == "5"
    assert rows["detector_only"]["association_f1"] == NOT_MEASURED
    assert rows["full_csa_yolo"]["status"] == NOT_MEASURED
