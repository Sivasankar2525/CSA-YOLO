"""Unit tests for VOC parsing and dataset auditing."""

from pathlib import Path

import pytest

from csa_yolo.data.audit import audit_annotations
from csa_yolo.data.types import AnnotatedObject, ImageAnnotation
from csa_yolo.data.voc import parse_voc_annotation

VOC_XML = """<annotation>
  <filename>img_001.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>person</name><difficult>0</difficult>
    <bndbox><xmin>100</xmin><ymin>50</ymin><xmax>300</xmax><ymax>450</ymax></bndbox>
  </object>
  <object>
    <name>hat</name><difficult>1</difficult>
    <bndbox><xmin>150</xmin><ymin>30</ymin><xmax>250</xmax><ymax>90</ymax></bndbox>
  </object>
</annotation>"""


@pytest.fixture()
def voc_file(tmp_path: Path) -> Path:
    path = tmp_path / "img_001.xml"
    path.write_text(VOC_XML)
    return path


def test_parse_voc_annotation(voc_file: Path) -> None:
    ann = parse_voc_annotation(voc_file)
    assert ann.image_filename == "img_001.jpg"
    assert (ann.image_width, ann.image_height) == (640, 480)
    assert len(ann.objects) == 2
    person, hat = ann.objects
    assert person.class_name == "person" and not person.difficult
    assert hat.class_name == "hat" and hat.difficult
    assert person.xyxy == (100.0, 50.0, 300.0, 450.0)


def test_parse_malformed_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<annotation><object>")
    with pytest.raises(ValueError):
        parse_voc_annotation(bad)


def _ann(objects: list[AnnotatedObject], name: str = "a.jpg") -> ImageAnnotation:
    return ImageAnnotation(image_filename=name, image_width=640, image_height=480,
                           objects=tuple(objects))


def test_audit_counts_and_bins() -> None:
    annotations = [
        _ann([
            AnnotatedObject("person", (0, 0, 200, 400)),    # large (80000 px^2)
            AnnotatedObject("hat", (0, 0, 20, 20)),         # small (400 px^2)
            AnnotatedObject("hat", (0, 0, 90, 90)),         # medium (8100 px^2)
        ]),
        _ann([AnnotatedObject("person", (0, 0, 100, 100), difficult=True)], name="b.jpg"),
    ]
    audit = audit_annotations(annotations, "unit")
    assert audit.num_images == 2
    assert audit.num_instances == 4
    assert audit.class_counts == {"person": 2, "hat": 2}
    assert audit.images_per_class == {"person": 2, "hat": 1}
    assert audit.bbox_size_bins["hat"] == {"small": 1, "medium": 1}
    assert audit.co_occurrence == {"hat+person": 1}
    assert audit.difficult_instances == 1
    assert audit.objects_per_image["max"] == 3.0


def test_audit_empty() -> None:
    audit = audit_annotations([], "empty")
    assert audit.num_images == 0
    assert audit.class_counts == {}
