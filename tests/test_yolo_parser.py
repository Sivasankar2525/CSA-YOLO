"""Unit tests for YOLO-format label parsing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from csa_yolo.data.yolo import parse_yolo_directory, parse_yolo_label

CLASSES = ["person", "vest", "helmet_blue", "helmet_red", "helmet_white", "helmet_yellow"]


@pytest.fixture()
def yolo_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A 200x100 image with one centered person box and one vest box."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "img1.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    (labels / "img1.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\n")
    return labels, images


def test_parse_yolo_label_denormalizes(yolo_pair: tuple[Path, Path]) -> None:
    labels, images = yolo_pair
    ann = parse_yolo_label(labels / "img1.txt", images / "img1.jpg", CLASSES)
    assert (ann.image_width, ann.image_height) == (200, 100)
    person, vest = ann.objects
    assert person.class_name == "person"
    # cx=0.5*200=100, w=0.2*200=40 → x1=80, x2=120; cy=0.5*100=50, h=0.4*100=40 → y1=30, y2=70
    assert person.xyxy == (80.0, 30.0, 120.0, 70.0)
    assert vest.class_name == "vest"


def test_out_of_range_class_raises(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "x.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    label = tmp_path / "x.txt"
    label.write_text("99 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        parse_yolo_label(label, images / "x.jpg", CLASSES)


def test_parse_directory_skips_orphan_labels(yolo_pair: tuple[Path, Path]) -> None:
    labels, images = yolo_pair
    (labels / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    parsed = parse_yolo_directory(labels, images, CLASSES)
    assert len(parsed) == 1
    assert parsed[0].image_filename == "img1.jpg"
