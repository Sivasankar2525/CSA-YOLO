"""YOLO-format annotation parsing.

CHV (and Roboflow exports) use YOLO labels: one ``.txt`` per image with lines
``class_id cx cy w h`` in coordinates normalized to [0, 1]. Converting to
absolute pixels requires the image dimensions, which are read from the image
files themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2

from csa_yolo.data.types import AnnotatedObject, ImageAnnotation

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def parse_yolo_label(
    label_path: Path | str,
    image_path: Path | str,
    class_names: Sequence[str],
) -> ImageAnnotation:
    """Parse one YOLO label file into absolute-pixel annotations.

    Args:
        label_path: The ``.txt`` label file.
        image_path: The corresponding image (needed for pixel dimensions).
        class_names: Ordered class names; index = YOLO class id.

    Returns:
        The parsed annotation.

    Raises:
        FileNotFoundError: If label or image is missing/unreadable.
        ValueError: If a class id is outside ``class_names``.
    """
    label_path, image_path = Path(label_path), Path(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image unreadable: {image_path}")
    height, width = image.shape[:2]

    objects: list[AnnotatedObject] = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            logger.warning("Skipping malformed line %d in %s", line_no, label_path)
            continue
        class_id = int(float(parts[0]))
        if not 0 <= class_id < len(class_names):
            raise ValueError(
                f"Class id {class_id} out of range for {len(class_names)} classes "
                f"({label_path}:{line_no})"
            )
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        x2 = (cx + w / 2.0) * width
        y2 = (cy + h / 2.0) * height
        objects.append(AnnotatedObject(class_name=class_names[class_id], xyxy=(x1, y1, x2, y2)))

    return ImageAnnotation(
        image_filename=image_path.name,
        image_width=width,
        image_height=height,
        objects=tuple(objects),
    )


def parse_yolo_directory(
    labels_dir: Path | str,
    images_dir: Path | str,
    class_names: Sequence[str],
) -> list[ImageAnnotation]:
    """Parse every YOLO label in a directory, pairing labels with images.

    Labels without a readable image (and macOS ``._`` resource-fork files)
    are skipped with a logged count so quality issues stay visible.
    """
    labels_dir, images_dir = Path(labels_dir), Path(images_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    parsed: list[ImageAnnotation] = []
    skipped = 0
    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.name.startswith("._"):
            continue
        image_path = _find_image(images_dir, label_file.stem)
        if image_path is None:
            skipped += 1
            continue
        try:
            parsed.append(parse_yolo_label(label_file, image_path, class_names))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s: %s", label_file.name, exc)
            skipped += 1
    if skipped:
        logger.warning("Skipped %d labels without readable images in %s", skipped, labels_dir)
    logger.info("Parsed %d YOLO annotations from %s", len(parsed), labels_dir)
    return parsed
