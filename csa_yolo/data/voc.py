"""Pascal VOC annotation parsing.

SHWD (and several other candidate PPE datasets) ship Pascal VOC XML
annotations. This parser converts them into lightweight typed records used by
the audit and (later) the VOC→YOLO conversion in dataset preparation.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from csa_yolo.data.types import AnnotatedObject, ImageAnnotation

logger = logging.getLogger(__name__)


def parse_voc_annotation(xml_path: Path | str) -> ImageAnnotation:
    """Parse a single Pascal VOC XML annotation file.

    Args:
        xml_path: Path to the ``.xml`` annotation.

    Returns:
        The parsed annotation.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or malformed.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"Annotation not found: {xml_path}")
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed VOC XML {xml_path}: {exc}") from exc

    filename = root.findtext("filename", default=xml_path.stem)
    size = root.find("size")
    width = int(float(size.findtext("width"))) if size is not None else 0
    height = int(float(size.findtext("height"))) if size is not None else 0

    objects: list[AnnotatedObject] = []
    for obj in root.iter("object"):
        name = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if name is None or bndbox is None:
            logger.warning("Skipping malformed <object> in %s", xml_path)
            continue
        try:
            xyxy = tuple(
                float(bndbox.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError):
            logger.warning("Skipping object with malformed bndbox in %s", xml_path)
            continue
        difficult = (obj.findtext("difficult", default="0").strip() == "1")
        objects.append(AnnotatedObject(class_name=name.strip(), xyxy=xyxy, difficult=difficult))

    return ImageAnnotation(
        image_filename=filename,
        image_width=width,
        image_height=height,
        objects=tuple(objects),
    )


def parse_voc_directory(annotations_dir: Path | str) -> list[ImageAnnotation]:
    """Parse every ``.xml`` file in a directory, skipping malformed files.

    Malformed files are logged and skipped rather than aborting the audit,
    but their count is reported so data-quality problems stay visible.
    """
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    parsed: list[ImageAnnotation] = []
    skipped = 0
    for xml_file in sorted(annotations_dir.glob("*.xml")):
        try:
            parsed.append(parse_voc_annotation(xml_file))
        except ValueError:
            skipped += 1
    if skipped:
        logger.warning("Skipped %d malformed annotation files in %s", skipped, annotations_dir)
    logger.info("Parsed %d VOC annotations from %s", len(parsed), annotations_dir)
    return parsed
