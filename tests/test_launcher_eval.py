"""Launcher evaluation and failure-message tests, using fake model objects (no GPU, no training)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("train_detector", PROJECT_ROOT / "training" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load()


class FakeBox:
    def __init__(self):
        self.mp, self.mr, self.map50, self.map = 0.5, 0.4, 0.3, 0.2
        self.ap_class_index = [0, 1]
        self.p = [0.6, 0.4]
        self.r = [0.5, 0.3]
        self.f1 = [0.55, 0.34]
        self.ap50 = [0.7, 0.2]
        self.maps = [0.5, 0.1]


class FakeModel:
    def __init__(self):
        self.calls = []

    def val(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(box=FakeBox(), names={0: "person", 1: "helmet"})


def test_split_outputs_are_named_by_split_and_never_overwrite(tmp_path: Path) -> None:
    model = FakeModel()
    run = tmp_path / "run"
    launcher.evaluate_split(model, tmp_path / "d.yaml", "val", 640, "cpu", run)
    launcher.evaluate_split(model, tmp_path / "d.yaml", "test", 640, "cpu", run)
    assert (run / "val_metrics.json").is_file()
    assert (run / "test_metrics.json").is_file()
    names = [c["name"] for c in model.calls]
    assert names == ["eval_val", "eval_test"]  # separate plot folders


def test_evaluation_settings_are_fixed_and_identical_across_splits(tmp_path: Path) -> None:
    model = FakeModel()
    launcher.evaluate_split(model, tmp_path / "d.yaml", "test", 640, "cpu", tmp_path)
    call = model.calls[0]
    assert call["conf"] == launcher.EVALUATION_SETTINGS["conf"] == 0.001
    assert call["iou"] == launcher.EVALUATION_SETTINGS["iou"] == 0.7
    assert call["imgsz"] == 640 and call["split"] == "test"


def test_extracted_metrics_include_per_class_and_unavailable_list(tmp_path: Path) -> None:
    result = launcher.evaluate_split(FakeModel(), tmp_path / "d.yaml", "test", 640, "cpu", tmp_path)
    assert result["precision"] == 0.5
    assert result["per_class_AP50"] == {"person": 0.7, "helmet": 0.2}
    assert result["F1"] is None and "F1" in result["unavailable"]
    assert result["evaluation_settings"]["iou"] == 0.7


def test_missing_split_folders_give_actionable_error(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train",
                                    "val": "images/valid", "test": "images/test", "nc": 1, "names": ["p"]}))
    with pytest.raises(launcher.LauncherError, match="split folder missing.*csa_ppe_v1_clean"):
        launcher.check_dataset_folders(data)


def test_present_split_folders_pass(tmp_path: Path) -> None:
    for sub in ("images/train", "images/valid", "images/test"):
        (tmp_path / sub).mkdir(parents=True)
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train",
                                    "val": "images/valid", "test": "images/test", "nc": 1, "names": ["p"]}))
    launcher.check_dataset_folders(data)  # no exception


def test_optimizer_record_without_trainer_is_explicit_not_invented() -> None:
    record = launcher.optimizer_record(SimpleNamespace())
    assert record["name"] is None and record["lr0"] is None
    assert "not available" in record["note"]


def test_optimizer_record_reads_real_param_groups() -> None:
    class Opt:
        param_groups = [{"lr": 0.01, "momentum": 0.937, "weight_decay": 0.0005}]

    model = SimpleNamespace(trainer=SimpleNamespace(optimizer=Opt()))
    record = launcher.optimizer_record(model)
    assert record["name"] == "Opt" and record["lr0"] == 0.01 and record["momentum"] == 0.937


def test_manifest_records_environment_and_marks_unavailable(tmp_path: Path) -> None:
    from csa_yolo.utils.manifest import build_manifest

    manifest = build_manifest(
        experiment={"name": "x", "model": {"family": "ultralytics", "weights": "yolo11n.pt"}},
        effective_train={"epochs": 1}, dataset_yaml=tmp_path / "d.yaml",
        resolved_dataset_yaml=tmp_path / "r.yaml", output_dir=tmp_path, seed=42,
        timings={}, optimizer={"name": None}, evaluation={},
    )
    json.dumps(manifest)  # serialisable
    assert manifest["seed"] == 42
    assert manifest["initialization"] == "pretrained weights: yolo11n.pt"
    assert manifest["environment"]["python"]
    assert "commit" in manifest["git"]


def test_random_init_is_described_correctly() -> None:
    from csa_yolo.utils.manifest import _initialization

    assert _initialization({"family": "ultralytics", "weights": "yolo11n.yaml"}).startswith("random")
    assert _initialization({"family": "icsd"}).startswith("random")
