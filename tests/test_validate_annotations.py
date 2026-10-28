"""Annotation validator: each schema rule fires, well-formed files pass (synthetic examples only)."""

import json
from pathlib import Path

from evaluation.validate_annotations import validate_file

META = {"annotator": "ROLE", "annotation_date": "2026-01-01"}


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_valid_wpam_passes(tmp_path: Path) -> None:
    data = {**META, "images": {"img1": {"workers": {"w1": {"helmet": "h1", "vest": None}}}}}
    assert validate_file(_write(tmp_path, "wpam.json", data)) == []


def test_wpam_missing_class_key_is_reported(tmp_path: Path) -> None:
    data = {**META, "images": {"img1": {"workers": {"w1": {"helmet": "h1"}}}}}
    errors = validate_file(_write(tmp_path, "wpam.json", data))
    assert any("missing 'vest'" in e for e in errors)


def test_scrm_rejects_unknown_as_human_label(tmp_path: Path) -> None:
    data = {**META, "images": {"img1": {"w1": "UNKNOWN"}}}
    errors = validate_file(_write(tmp_path, "scrm.json", data))
    assert any("UNKNOWN is a system output" in e for e in errors)


def test_scrm_valid_labels_pass(tmp_path: Path) -> None:
    data = {**META, "images": {"img1": {"w1": "COMPLIANT", "w2": "NON_COMPLIANT"}}}
    assert validate_file(_write(tmp_path, "scrm.json", data)) == []


def test_tracking_requires_time_ordered_frames(tmp_path: Path) -> None:
    assert validate_file(_write(tmp_path, "tracking.json", {**META, "frames": []}))
    good = {**META, "frames": [{"w1": "5"}, {"w1": 5}]}
    assert validate_file(_write(tmp_path, "tracking.json", good)) == []


def test_tsmm_checks_status_values_and_intervals(tmp_path: Path) -> None:
    bad = {**META, "clips": {"c1": {"frame_status": {"w1": ["COMPLIANT", "MAYBE"]},
                                    "occlusion_intervals": [[10, 5]]}}}
    errors = validate_file(_write(tmp_path, "tsmm.json", bad))
    assert any("must be one of" in e for e in errors)
    assert any("start<=end" in e for e in errors)


def test_metadata_is_required(tmp_path: Path) -> None:
    data = {"images": {"img1": {"w1": "COMPLIANT"}}}
    errors = validate_file(_write(tmp_path, "scrm.json", data))
    assert any("missing metadata 'annotator'" in e for e in errors)


def test_unreadable_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "scrm.json"
    path.write_text("{not json")
    assert validate_file(path)[0].startswith("cannot read JSON")


def test_unknown_module_name_requires_flag(tmp_path: Path) -> None:
    path = _write(tmp_path, "notes.json", {**META, "images": {}})
    try:
        validate_file(path)
    except ValueError as exc:
        assert "--module" in str(exc)
    else:
        raise AssertionError("expected ValueError")
