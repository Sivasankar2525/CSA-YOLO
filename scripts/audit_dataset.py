"""Audit a registered dataset: statistics + sample visualization grid.

Datasets are declared in ``configs/datasets.yaml`` (format, paths, classes);
this script never hardcodes dataset paths.

Usage:
    python scripts/audit_dataset.py --dataset SHWD [--samples 12]
    python scripts/audit_dataset.py --dataset CHV

Outputs (under results/audits/<name>/):
    audit.json        Full statistics
    audit.md          Markdown class table for reports
    samples.jpg       Grid of annotated sample images for visual spot-check
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from csa_yolo.data import (
    ImageAnnotation,
    audit_annotations,
    parse_voc_directory,
    parse_yolo_directory,
)
from csa_yolo.utils import set_seed, setup_logger

_COLORS = {
    "person": (0, 200, 0),
    "hat": (0, 140, 255),
    "vest": (255, 160, 0),
}
_DEFAULT_COLOR = (200, 0, 200)


def load_registry(registry_path: Path) -> dict:
    with registry_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]


def parse_dataset(name: str, spec: dict) -> list[ImageAnnotation]:
    fmt = spec["format"]
    if fmt == "voc":
        return parse_voc_directory(PROJECT_ROOT / spec["annotations_dir"])
    if fmt == "yolo":
        return parse_yolo_directory(
            PROJECT_ROOT / spec["labels_dir"],
            PROJECT_ROOT / spec["images_dir"],
            spec["class_names"],
        )
    raise ValueError(f"Unknown dataset format '{fmt}' for {name}")


def _color_for(class_name: str) -> tuple[int, int, int]:
    key = class_name.lower()
    for prefix, color in _COLORS.items():
        if key.startswith(prefix):
            return color
    if key.startswith("helmet"):
        return (0, 140, 255)
    return _DEFAULT_COLOR


def draw_annotation(image: np.ndarray, ann: ImageAnnotation) -> np.ndarray:
    for obj in ann.objects:
        x1, y1, x2, y2 = (int(v) for v in obj.xyxy)
        color = _color_for(obj.class_name)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, obj.class_name, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return image


def sample_grid(
    annotations: list[ImageAnnotation], images_dir: Path, n_samples: int, cell: int = 320
) -> np.ndarray:
    """Build a grid image of randomly sampled annotated images."""
    chosen = random.sample(annotations, min(n_samples, len(annotations)))
    cells: list[np.ndarray] = []
    for ann in chosen:
        path = images_dir / ann.image_filename
        image = cv2.imread(str(path))
        if image is None:  # filename field sometimes lacks the real extension
            candidates = list(images_dir.glob(Path(ann.image_filename).stem + ".*"))
            image = cv2.imread(str(candidates[0])) if candidates else None
        if image is None:
            continue
        image = draw_annotation(image, ann)
        cells.append(cv2.resize(image, (cell, cell)))
    if not cells:
        raise RuntimeError(f"No sample images could be read from {images_dir}")
    cols = int(np.ceil(np.sqrt(len(cells))))
    rows = int(np.ceil(len(cells) / cols))
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, img in enumerate(cells):
        r, c = divmod(i, cols)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Name from configs/datasets.yaml")
    parser.add_argument("--registry", type=Path,
                        default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("INFO")
    set_seed(args.seed)

    registry = load_registry(args.registry)
    if args.dataset not in registry:
        logger.error("Dataset '%s' not in registry (%s)", args.dataset, sorted(registry))
        return 1
    spec = registry[args.dataset]

    annotations = parse_dataset(args.dataset, spec)
    audit = audit_annotations(annotations, dataset_name=args.dataset)

    out_dir = PROJECT_ROOT / "results" / "audits" / args.dataset
    audit.save(out_dir / "audit.json")
    (out_dir / "audit.md").write_text(
        f"# {args.dataset} audit\n\nImages: {audit.num_images} — "
        f"Instances: {audit.num_instances}\n\n{audit.summary_table()}\n\n"
        f"Objects/image: {audit.objects_per_image}\n\n"
        f"Co-occurrence (images containing both): {audit.co_occurrence}\n\n"
        f"Semantics: {spec.get('semantics_warning', 'n/a').strip()}\n"
    )
    grid = sample_grid(annotations, PROJECT_ROOT / spec["images_dir"], args.samples)
    cv2.imwrite(str(out_dir / "samples.jpg"), grid)
    logger.info("Audit complete: %s", out_dir)
    print(audit.summary_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
