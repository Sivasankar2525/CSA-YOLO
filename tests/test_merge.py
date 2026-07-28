"""Unit tests for dataset merging: class mapping, YOLO writing, splits."""

import json
from pathlib import Path

import pytest

from csa_yolo.data.coco import parse_coco_file
from csa_yolo.data.merge import map_annotation, random_split, to_yolo_lines
from csa_yolo.data.types import AnnotatedObject, ImageAnnotation

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]


def _ann(objects: list[AnnotatedObject]) -> ImageAnnotation:
    return ImageAnnotation("img.jpg", 200, 100, tuple(objects))


def test_map_annotation_remaps_and_drops() -> None:
    ann = _ann([
        AnnotatedObject("hardhat", (0, 0, 10, 10)),
        AnnotatedObject("excavators", (5, 5, 50, 50)),   # unmapped → dropped
        AnnotatedObject("person", (0, 0, 80, 90)),
    ])
    mapped = map_annotation(ann, {"hardhat": "helmet", "person": "person"}, CLASSES)
    assert mapped is not None
    assert [o.class_name for o in mapped.objects] == ["helmet", "person"]


def test_map_annotation_none_when_empty() -> None:
    ann = _ann([AnnotatedObject("truck", (0, 0, 10, 10))])
    assert map_annotation(ann, {"hardhat": "helmet"}, CLASSES) is None


def test_map_annotation_invalid_target_raises() -> None:
    with pytest.raises(ValueError):
        map_annotation(_ann([]), {"hardhat": "helmetz"}, CLASSES)


def test_to_yolo_lines_normalizes_and_clamps() -> None:
    ann = _ann([
        AnnotatedObject("person", (50.0, 25.0, 150.0, 75.0)),
        AnnotatedObject("helmet", (-10.0, -5.0, 20.0, 10.0)),   # clamped to bounds
        AnnotatedObject("vest", (0.0, 0.0, 0.5, 40.0)),          # degenerate → dropped
    ])
    lines = to_yolo_lines(ann, CLASSES)
    assert len(lines) == 2
    assert lines[0] == "0 0.500000 0.500000 0.500000 0.500000"
    cls, cx, cy, w, h = lines[1].split()
    assert cls == "1" and float(cx) == pytest.approx(0.05) and float(w) == pytest.approx(0.1)


def test_random_split_deterministic_and_sized() -> None:
    a = random_split(100, (0.8, 0.1, 0.1), seed=7)
    b = random_split(100, (0.8, 0.1, 0.1), seed=7)
    assert a == b
    assert a.count("train") == 80 and a.count("valid") == 10 and a.count("test") == 10
    assert random_split(100, (0.8, 0.1, 0.1), seed=8) != a


def test_random_split_bad_ratios() -> None:
    with pytest.raises(ValueError):
        random_split(10, (0.5, 0.2, 0.2), seed=1)


def test_parse_coco_file(tmp_path: Path) -> None:
    coco = {
        "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "hardhat"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 50},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 50},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [10, 10, 30, 20]},
            {"image_id": 1, "category_id": 1, "bbox": [15, 5, 10, 8]},
        ],
    }
    path = tmp_path / "_annotations.coco.json"
    path.write_text(json.dumps(coco))
    parsed = parse_coco_file(path)
    assert len(parsed) == 2
    first = next(p for p in parsed if p.image_filename == "a.jpg")
    assert len(first.objects) == 2
    assert first.objects[0].xyxy == (10, 10, 40, 30)   # xywh → xyxy
    empty = next(p for p in parsed if p.image_filename == "b.jpg")
    assert empty.objects == ()
