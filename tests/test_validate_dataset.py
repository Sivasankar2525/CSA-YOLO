"""Dataset validator tests on small synthetic YOLO datasets (each check must fire)."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("validate_dataset", PROJECT_ROOT / "scripts" / "validate_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vd = _load()
NAMES = ["person", "helmet"]


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (20, 30, 3), dtype=np.uint8))


def _dataset(root: Path, items: dict[str, list[str]], images_seed: int = 0) -> Path:
    """items: {'train/a': ['0 0.5 0.5 0.2 0.2', ...]} -> image + label files."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES, "nc": len(NAMES)}))
    for i, (key, lines) in enumerate(items.items()):
        split, stem = key.split("/")
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        _image(root / "images" / split / f"{stem}.jpg", seed=images_seed + i)
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    for split in ("train", "valid", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    return root


def _check(root: Path, expect=None) -> dict:
    return vd.validate_dataset(root, expect or {})


def test_clean_synthetic_dataset_passes(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"], "valid/b": ["1 0.3 0.3 0.1 0.1"],
                                     "test/c": ["0 0.5 0.5 0.4 0.4"]})
    report = _check(root)
    assert report["verdict"] == "PASS"
    assert report["total_valid_boxes"] == 3
    assert report["splits"]["train"]["boxes_per_class"] == {"person": 1, "helmet": 0}


def test_image_without_label_is_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"]})
    _image(root / "images" / "train" / "orphan.jpg", seed=99)
    report = _check(root)
    assert report["splits"]["train"]["problems"]["image_without_label"] == 1


def test_label_without_image_is_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"]})
    (root / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert _check(root)["splits"]["train"]["problems"]["label_without_image"] == 1


def test_malformed_lines_are_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2", "0 x 0.5 0.2 0.2"]})
    problems = _check(root)["splits"]["train"]["problems"]
    assert problems["malformed_line_field_count"] == 1
    assert problems["malformed_line_non_numeric"] == 1


def test_class_id_out_of_range_is_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["7 0.5 0.5 0.2 0.2"]})
    assert _check(root)["splits"]["train"]["problems"]["class_id_out_of_range"] == 1


def test_box_problems_are_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": [
        "0 0.5 0.5 0.0 0.2",    # zero width
        "0 1.5 0.5 0.2 0.2",    # coordinate outside [0,1]
        "0 0.98 0.5 0.4 0.2",   # overhangs the right edge
    ]})
    problems = _check(root)["splits"]["train"]["problems"]
    assert problems["box_non_positive_size"] == 1
    assert problems["box_coordinate_outside_unit_interval"] == 1
    assert problems["box_overhangs_image"] == 1


def test_unreadable_image_is_flagged(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"]})
    (root / "images" / "train" / "a.jpg").write_bytes(b"not an image")
    assert _check(root)["splits"]["train"]["problems"]["unreadable_image"] == 1


def test_cross_split_exact_duplicate_is_leakage(tmp_path: Path) -> None:
    root = tmp_path / "d"
    _dataset(root, {"train/a": ["0 0.5 0.5 0.2 0.2"], "test/b": ["0 0.5 0.5 0.2 0.2"]})
    # make the test image byte-identical to the train image
    (root / "images" / "test" / "b.jpg").write_bytes((root / "images" / "train" / "a.jpg").read_bytes())
    overlap = _check(root)["overlap"]
    assert overlap["exact_image_leakage"] == 1
    assert overlap["near_duplicate_leakage"].startswith("NOT MEASURED")


def test_within_split_duplicate_is_reported_separately(tmp_path: Path) -> None:
    root = tmp_path / "d"
    _dataset(root, {"train/a": ["0 0.5 0.5 0.2 0.2"], "train/b": ["0 0.5 0.5 0.2 0.2"]})
    (root / "images" / "train" / "b.jpg").write_bytes((root / "images" / "train" / "a.jpg").read_bytes())
    overlap = _check(root)["overlap"]
    assert overlap["exact_image_leakage"] == 0
    assert overlap["within_split_duplicate_files"] == 1


def test_expected_counts_are_compared_not_forced(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"]})
    report = _check(root, {"train": 2})
    assert report["splits"]["train"]["expected_images"] == 2
    assert report["splits"]["train"]["count_matches_expected"] is False
    assert report["splits"]["train"]["images"] == 1  # the data was not changed


def test_empty_split_does_not_crash(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train/a": ["0 0.5 0.5 0.2 0.2"]})
    report = _check(root)
    assert report["splits"]["test"]["images"] == 0


def test_missing_data_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vd.validate_dataset(tmp_path, {})


def test_expect_parser_rejects_bad_input() -> None:
    assert vd._parse_expect(["train=5", "valid=2"]) == {"train": 5, "valid": 2}
    with pytest.raises(Exception):
        vd._parse_expect(["bogus=1"])
