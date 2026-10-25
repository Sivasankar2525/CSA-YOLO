"""Leakage-cleanup tests: split policy on synthetic data, plus the real clean dataset if present."""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = PROJECT_ROOT / "datasets" / "csa_ppe_v1_clean"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_clean_dataset", PROJECT_ROOT / "scripts" / "build_clean_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _make_dataset(root: Path, files: dict[str, bytes]) -> None:
    """files: {'split/name.jpg': content}. Each image gets one label line."""
    for split in ("train", "valid", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for key, content in files.items():
        split, name = key.split("/")
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / name).write_bytes(content)
        (root / "labels" / split / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text("names: [person, helmet]\n")


def test_duplicate_is_removed_from_train_when_test_copy_exists(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/a.jpg": b"same", "test/b.jpg": b"same", "valid/c.jpg": b"other"})
    removals, groups = builder.plan_removals(tmp_path)
    assert [(r["split"], r["file"]) for r in removals] == [("train", "a.jpg")]
    assert len(groups) == 1
    assert groups[next(iter(groups))]["kept"] == "test/b.jpg"


def test_valid_copy_is_kept_over_train_copy(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/a.jpg": b"same", "valid/b.jpg": b"same"})
    removals, _ = builder.plan_removals(tmp_path)
    assert [(r["split"], r["file"]) for r in removals] == [("train", "a.jpg")]


def test_kept_copy_is_first_test_file_by_name(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"test/a.jpg": b"same", "test/b.jpg": b"same", "train/c.jpg": b"same"})
    removals, groups = builder.plan_removals(tmp_path)
    assert groups[next(iter(groups))]["kept"] == "test/a.jpg"
    assert sorted((r["split"], r["file"]) for r in removals) == [("test", "b.jpg"), ("train", "c.jpg")]


def test_within_train_duplicates_keep_smallest_name(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/b.jpg": b"same", "train/a.jpg": b"same", "train/c.jpg": b"same"})
    removals, _ = builder.plan_removals(tmp_path)
    assert sorted(r["file"] for r in removals) == ["b.jpg", "c.jpg"]


def test_no_duplicates_means_no_removals(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/a.jpg": b"1", "valid/b.jpg": b"2", "test/c.jpg": b"3"})
    removals, groups = builder.plan_removals(tmp_path)
    assert removals == [] and groups == {}


def test_verify_detects_cross_split_leakage(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/a.jpg": b"same", "test/b.jpg": b"same"})
    report = builder.verify(tmp_path, ["person", "helmet"])
    assert report["exact_image_leakage"] == 1


def test_verify_reports_zero_on_leakage_free_split(tmp_path: Path) -> None:
    _make_dataset(tmp_path, {"train/a.jpg": b"1", "valid/b.jpg": b"2", "test/c.jpg": b"3"})
    report = builder.verify(tmp_path, ["person", "helmet"])
    assert report["exact_image_leakage"] == 0
    assert report["within_split_duplicate_files"] == 0
    assert report["images_without_labels"] == 0
    assert report["labels_without_images"] == 0


@pytest.mark.skipif(not (CLEAN / "data.yaml").exists(), reason="csa_ppe_v1_clean not built")
def test_real_clean_dataset_has_zero_exact_overlap() -> None:
    names = ["person", "helmet", "no_helmet", "vest", "no_vest"]
    report = builder.verify(CLEAN, names)
    assert report["exact_image_leakage"] == 0
    assert report["within_split_duplicate_files"] == 0
    assert report["images_without_labels"] == 0
    assert report["labels_without_images"] == 0
