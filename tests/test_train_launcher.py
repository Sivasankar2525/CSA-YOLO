"""Training launcher tests: config validation, overrides, dataset path resolution (no training run)."""

import importlib.util
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "train_detector", PROJECT_ROOT / "training" / "train_detector.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()
EXPERIMENTS = sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))


def test_three_required_experiments_exist() -> None:
    names = {p.stem for p in EXPERIMENTS}
    assert {"baseline_yolo11n", "yolo11n_scratch", "icsd_restored_scratch"} <= names


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_every_experiment_loads_and_uses_clean_dataset(path: Path) -> None:
    cfg = launcher.load_experiment(path)
    assert cfg["data"] == "datasets/csa_ppe_v1_clean/data.yaml"
    assert cfg["seed"] == 42
    assert cfg["train"]["imgsz"] == 640
    assert cfg["train"]["batch"] == 16


def test_experiment_families_are_known() -> None:
    families = {p.stem: launcher.load_experiment(p)["model"]["family"] for p in EXPERIMENTS}
    assert families["baseline_yolo11n"] == "ultralytics"
    assert families["yolo11n_scratch"] == "ultralytics"
    assert families["icsd_restored_scratch"] == "icsd"


def test_missing_section_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "x", "seed": 1, "model": {"family": "ultralytics"}, "data": "d"}))
    with pytest.raises(KeyError, match="train"):
        launcher.load_experiment(bad)


def test_unknown_model_family_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "name": "x", "seed": 1, "model": {"family": "rcnn"}, "data": "d", "train": {"epochs": 1},
    }))
    with pytest.raises(ValueError, match="family"):
        launcher.load_experiment(bad)


def test_overrides_keep_yaml_values_when_not_given() -> None:
    train = {"epochs": 100, "device": "mps", "batch": 16}
    assert launcher.apply_overrides(train, None, None) == train


def test_device_override_for_colab() -> None:
    effective = launcher.apply_overrides({"epochs": 100, "device": "mps"}, "cuda", None)
    assert effective == {"epochs": 100, "device": "cuda"}


def test_epochs_override_for_smoke_tests() -> None:
    effective = launcher.apply_overrides({"epochs": 100, "device": "mps"}, None, 1)
    assert effective["epochs"] == 1 and effective["device"] == "mps"


def test_overrides_do_not_mutate_input() -> None:
    train = {"epochs": 100, "device": "mps"}
    launcher.apply_overrides(train, "cuda", 2)
    assert train == {"epochs": 100, "device": "mps"}


def test_non_positive_epochs_override_is_rejected() -> None:
    with pytest.raises(ValueError):
        launcher.apply_overrides({"epochs": 100}, None, 0)


def test_resolve_data_yaml_points_to_dataset_folder(tmp_path: Path) -> None:
    dataset = tmp_path / "some_dataset"
    dataset.mkdir()
    source = dataset / "data.yaml"
    source.write_text(yaml.safe_dump({
        "path": "/Users/someone/elsewhere/dataset",  # stale absolute path from another machine
        "train": "images/train", "val": "images/valid", "test": "images/test",
        "nc": 1, "names": ["person"],
    }))
    out = tmp_path / "run"
    resolved = launcher.resolve_data_yaml(source, out)
    data = yaml.safe_load(resolved.read_text())
    assert data["path"] == str(dataset.resolve())
    assert data["names"] == ["person"]
    assert resolved == out / "data_resolved.yaml"
    # the source file is not modified
    assert yaml.safe_load(source.read_text())["path"] == "/Users/someone/elsewhere/dataset"


def test_real_clean_dataset_yaml_resolves_to_local_folder() -> None:
    source = PROJECT_ROOT / "datasets" / "csa_ppe_v1_clean" / "data.yaml"
    if not source.exists():
        pytest.skip("csa_ppe_v1_clean not built on this machine")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        resolved = launcher.resolve_data_yaml(source, Path(tmp))
        data = yaml.safe_load(resolved.read_text())
    assert data["path"] == str(source.resolve().parent)
    assert data["nc"] == 5
