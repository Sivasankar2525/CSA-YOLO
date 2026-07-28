"""Quantitative dataset auditing.

Produces the statistics required by the paper's Dataset section and by the
Week 2 selection decision: per-class instance counts, bounding-box size
distribution (COCO small/medium/large convention), per-image object density,
and class co-occurrence — e.g. how many images contain both a person and a
helmet, which directly bounds how much WPAM association data the set can
provide.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

from csa_yolo.data.types import ImageAnnotation

logger = logging.getLogger(__name__)

# COCO area convention (pixels^2) for small/medium/large objects.
_SMALL_AREA = 32.0**2
_MEDIUM_AREA = 96.0**2


@dataclass
class DatasetAudit:
    """Aggregated statistics for one dataset."""

    dataset_name: str
    num_images: int = 0
    num_instances: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)
    bbox_size_bins: dict[str, dict[str, int]] = field(default_factory=dict)
    co_occurrence: dict[str, int] = field(default_factory=dict)
    objects_per_image: dict[str, float] = field(default_factory=dict)
    difficult_instances: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "num_images": self.num_images,
            "num_instances": self.num_instances,
            "class_counts": dict(sorted(self.class_counts.items())),
            "images_per_class": dict(sorted(self.images_per_class.items())),
            "bbox_size_bins": self.bbox_size_bins,
            "co_occurrence": dict(sorted(self.co_occurrence.items())),
            "objects_per_image": self.objects_per_image,
            "difficult_instances": self.difficult_instances,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Audit saved to %s", path)

    def summary_table(self) -> str:
        """Markdown table of class counts for reports."""
        lines = ["| Class | Instances | Images | Small | Medium | Large |",
                 "|---|---|---|---|---|---|"]
        for cls in sorted(self.class_counts):
            bins = self.bbox_size_bins.get(cls, {})
            lines.append(
                f"| {cls} | {self.class_counts[cls]} | {self.images_per_class.get(cls, 0)} "
                f"| {bins.get('small', 0)} | {bins.get('medium', 0)} | {bins.get('large', 0)} |"
            )
        return "\n".join(lines)


def audit_annotations(annotations: Iterable[ImageAnnotation], dataset_name: str) -> DatasetAudit:
    """Compute audit statistics over parsed annotations.

    Args:
        annotations: Parsed per-image annotations.
        dataset_name: Name recorded in the audit output.

    Returns:
        The aggregated :class:`DatasetAudit`.
    """
    audit = DatasetAudit(dataset_name=dataset_name)
    class_counter: Counter[str] = Counter()
    image_class_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()
    size_bins: dict[str, Counter[str]] = {}
    objects_per_image: list[int] = []

    for ann in annotations:
        audit.num_images += 1
        objects_per_image.append(len(ann.objects))
        present_classes = set()
        for obj in ann.objects:
            class_counter[obj.class_name] += 1
            present_classes.add(obj.class_name)
            if obj.difficult:
                audit.difficult_instances += 1
            bin_name = (
                "small" if obj.area < _SMALL_AREA
                else "medium" if obj.area < _MEDIUM_AREA
                else "large"
            )
            size_bins.setdefault(obj.class_name, Counter())[bin_name] += 1
        for cls in present_classes:
            image_class_counter[cls] += 1
        for a, b in combinations(sorted(present_classes), 2):
            pair_counter[f"{a}+{b}"] += 1

    audit.num_instances = sum(class_counter.values())
    audit.class_counts = dict(class_counter)
    audit.images_per_class = dict(image_class_counter)
    audit.bbox_size_bins = {cls: dict(bins) for cls, bins in size_bins.items()}
    audit.co_occurrence = dict(pair_counter)
    if objects_per_image:
        n = len(objects_per_image)
        ordered = sorted(objects_per_image)
        audit.objects_per_image = {
            "mean": round(sum(objects_per_image) / n, 2),
            "median": float(ordered[n // 2]),
            "max": float(ordered[-1]),
        }
    logger.info(
        "Audited %s: %d images, %d instances, %d classes",
        dataset_name, audit.num_images, audit.num_instances, len(audit.class_counts),
    )
    return audit
