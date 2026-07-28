"""Merging heterogeneous PPE datasets into one unified YOLO-format dataset.

Each source declares a ``class_map`` from its native class names to the
unified CSA-YOLO taxonomy. Objects whose class is not in the map are dropped;
images left with zero mapped objects are dropped (and counted). The merge is
fully deterministic given the config and seed, and emits a per-image manifest
so the exact composition of every split is committed to git.
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from csa_yolo.data.types import AnnotatedObject, ImageAnnotation

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class MergeRecord:
    """One image scheduled for inclusion in the unified dataset."""

    source: str
    split: str
    image_path: Path
    annotation: ImageAnnotation  # classes already mapped to the unified taxonomy


def map_annotation(
    annotation: ImageAnnotation, class_map: dict[str, str], unified_classes: Sequence[str]
) -> ImageAnnotation | None:
    """Remap an annotation's classes to the unified taxonomy.

    Args:
        annotation: Source annotation with native class names.
        class_map: native name → unified name. Native classes absent from the
            map are dropped silently (they are out of scope by design).
        unified_classes: The unified class list; mapped targets must be in it.

    Returns:
        The remapped annotation, or ``None`` if no objects survive mapping.

    Raises:
        ValueError: If the class map targets a class not in ``unified_classes``.
    """
    invalid = set(class_map.values()) - set(unified_classes)
    if invalid:
        raise ValueError(f"class_map targets unknown unified classes: {sorted(invalid)}")
    kept = tuple(
        AnnotatedObject(
            class_name=class_map[obj.class_name], xyxy=obj.xyxy, difficult=obj.difficult
        )
        for obj in annotation.objects
        if obj.class_name in class_map
    )
    if not kept:
        return None
    return ImageAnnotation(
        image_filename=annotation.image_filename,
        image_width=annotation.image_width,
        image_height=annotation.image_height,
        objects=kept,
    )


def to_yolo_lines(annotation: ImageAnnotation, unified_classes: Sequence[str]) -> list[str]:
    """Convert an annotation to YOLO label lines (normalized cx cy w h).

    Boxes are clamped to image bounds; degenerate boxes (zero width/height
    after clamping) are dropped with a warning.
    """
    width, height = annotation.image_width, annotation.image_height
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {annotation.image_filename}")
    lines: list[str] = []
    for obj in annotation.objects:
        x1 = min(max(obj.xyxy[0], 0.0), width)
        y1 = min(max(obj.xyxy[1], 0.0), height)
        x2 = min(max(obj.xyxy[2], 0.0), width)
        y2 = min(max(obj.xyxy[3], 0.0), height)
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            logger.warning("Dropping degenerate box in %s", annotation.image_filename)
            continue
        class_id = list(unified_classes).index(obj.class_name)
        cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def random_split(
    n_items: int, ratios: tuple[float, float, float], seed: int
) -> list[str]:
    """Deterministic split assignment for sources without official splits."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    assignments = []
    n_train = round(n_items * ratios[0])
    n_valid = round(n_items * ratios[1])
    for i in range(n_items):
        if i < n_train:
            assignments.append("train")
        elif i < n_train + n_valid:
            assignments.append("valid")
        else:
            assignments.append("test")
    rng = random.Random(seed)
    rng.shuffle(assignments)
    return assignments


def write_unified_dataset(
    records: Sequence[MergeRecord],
    output_dir: Path,
    unified_classes: Sequence[str],
) -> dict[str, int]:
    """Write images, YOLO labels, data.yaml, and the manifest.

    Returns:
        Per-split image counts.
    """
    output_dir = Path(output_dir)
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = dict.fromkeys(SPLITS, 0)
    manifest_lines = ["source,split,original_image,merged_image"]
    for record in records:
        # Prefix with source to avoid filename collisions across datasets.
        merged_name = f"{record.source}_{record.image_path.name}"
        stem = Path(merged_name).stem
        shutil.copy2(record.image_path, output_dir / "images" / record.split / merged_name)
        label_lines = to_yolo_lines(record.annotation, unified_classes)
        (output_dir / "labels" / record.split / f"{stem}.txt").write_text(
            "\n".join(label_lines) + "\n"
        )
        counts[record.split] += 1
        manifest_lines.append(
            f"{record.source},{record.split},{record.image_path.name},{merged_name}"
        )

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\nval: images/valid\ntest: images/test\n"
        f"nc: {len(unified_classes)}\n"
        f"names: {list(unified_classes)}\n"
    )
    (output_dir / "manifest.csv").write_text("\n".join(manifest_lines) + "\n")
    logger.info("Unified dataset written to %s: %s", output_dir, counts)
    return counts
