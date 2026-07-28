"""COCO-format annotation parsing (Roboflow exports use this)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from csa_yolo.data.types import AnnotatedObject, ImageAnnotation

logger = logging.getLogger(__name__)


def parse_coco_file(annotations_json: Path | str) -> list[ImageAnnotation]:
    """Parse a COCO ``_annotations.coco.json`` file into per-image annotations.

    Args:
        annotations_json: Path to the COCO JSON (images assumed to live in the
            same directory, as in Roboflow exports).

    Returns:
        One :class:`ImageAnnotation` per image entry (images without
        annotations are included with zero objects, so counts stay honest).
    """
    annotations_json = Path(annotations_json)
    if not annotations_json.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_json}")
    data = json.loads(annotations_json.read_text())

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}
    objects_by_image: dict[int, list[AnnotatedObject]] = {img_id: [] for img_id in images}

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in images:
            logger.warning("Annotation references unknown image id %s", image_id)
            continue
        x, y, w, h = ann["bbox"]  # COCO: [x_min, y_min, width, height]
        class_name = categories.get(ann["category_id"], str(ann["category_id"]))
        objects_by_image[image_id].append(
            AnnotatedObject(class_name=class_name, xyxy=(x, y, x + w, y + h))
        )

    parsed = [
        ImageAnnotation(
            image_filename=img["file_name"],
            image_width=int(img["width"]),
            image_height=int(img["height"]),
            objects=tuple(objects_by_image[img_id]),
        )
        for img_id, img in images.items()
    ]
    logger.info("Parsed %d COCO image annotations from %s", len(parsed), annotations_json)
    return parsed
