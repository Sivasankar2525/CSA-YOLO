"""Build the unified CSA-PPE dataset from raw sources per configs/merge.yaml.

NOTE: this produces the ORIGINAL reference copy (datasets/csa_ppe_v1, 1924 images, which
contains exact duplicate images). The leakage-free research dataset used for all training
and evaluation is produced from it by scripts/build_clean_dataset.py -> datasets/csa_ppe_v1_clean.

Fully deterministic: same config + seed produces identical splits. Outputs:
    datasets/csa_ppe_v1/{images,labels}/{train,valid,test}/
    datasets/csa_ppe_v1/data.yaml          (Ultralytics training config)
    datasets/csa_ppe_v1/manifest.csv       (source, split, original → merged name)
    datasets/splits/csa_ppe_v1_manifest.csv (git-committed copy of the manifest)

Usage:
    python training/prepare_dataset.py [--config configs/merge.yaml]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from csa_yolo.data import audit_annotations, parse_voc_directory, parse_yolo_directory
from csa_yolo.data.coco import parse_coco_file
from csa_yolo.data.merge import (
    MergeRecord,
    map_annotation,
    random_split,
    write_unified_dataset,
)
from csa_yolo.utils import set_seed, setup_logger

logger = None  # configured in main()


def _find_image(images_dir: Path, filename: str) -> Path | None:
    path = images_dir / filename
    if path.is_file():
        return path
    matches = list(images_dir.glob(Path(filename).stem + ".*"))
    return matches[0] if matches else None


def collect_chv(name: str, spec: dict, classes: list[str]) -> list[MergeRecord]:
    """CHV: YOLO labels + official author split files."""
    labels_dir = PROJECT_ROOT / spec["labels_dir"]
    images_dir = PROJECT_ROOT / spec["images_dir"]
    splits_dir = PROJECT_ROOT / spec["splits_dir"]
    split_of: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for line in (splits_dir / f"{split}.txt").read_text().splitlines():
            stem = Path(line.strip()).stem
            if stem:
                split_of[stem] = split
    annotations = parse_yolo_directory(labels_dir, images_dir, spec["native_classes"])
    records = []
    unassigned = 0
    for ann in annotations:
        stem = Path(ann.image_filename).stem
        split = split_of.get(stem)
        if split is None:
            unassigned += 1
            continue
        mapped = map_annotation(ann, spec["class_map"], classes)
        if mapped is None:
            continue
        image_path = _find_image(images_dir, ann.image_filename)
        if image_path is None:
            continue
        records.append(MergeRecord(name, split, image_path, mapped))
    if unassigned:
        logger.warning("%s: %d images not in official split files (skipped)", name, unassigned)
    return records


def collect_coco_split_dirs(name: str, spec: dict, classes: list[str]) -> list[MergeRecord]:
    """CSS: Roboflow COCO export with one directory per split."""
    records = []
    for split, rel_dir in spec["split_dirs"].items():
        split_dir = PROJECT_ROOT / rel_dir
        for ann in parse_coco_file(split_dir / "_annotations.coco.json"):
            mapped = map_annotation(ann, spec["class_map"], classes)
            if mapped is None:
                continue
            image_path = _find_image(split_dir, ann.image_filename)
            if image_path is None:
                continue
            records.append(MergeRecord(name, split, image_path, mapped))
    return records


def collect_voc_capped(
    name: str, spec: dict, classes: list[str], seed: int
) -> list[MergeRecord]:
    """SHWD: VOC source, seeded subsample cap + seeded random split."""
    annotations = parse_voc_directory(PROJECT_ROOT / spec["annotations_dir"])
    images_dir = PROJECT_ROOT / spec["images_dir"]
    usable = []
    for ann in annotations:
        mapped = map_annotation(ann, spec["class_map"], classes)
        if mapped is None:
            continue
        image_path = _find_image(images_dir, ann.image_filename)
        if image_path is None:
            continue
        usable.append((image_path, mapped))
    usable.sort(key=lambda pair: pair[0].name)  # deterministic order before sampling
    cap = spec.get("max_images")
    if cap is not None and len(usable) > cap:
        usable = random.Random(seed).sample(usable, cap)
        usable.sort(key=lambda pair: pair[0].name)
    splits = random_split(len(usable), tuple(spec["split_ratios"]), seed)
    return [
        MergeRecord(name, split, image_path, mapped)
        for (image_path, mapped), split in zip(usable, splits)
    ]


_COLLECTORS = {
    "CHV": collect_chv,
    "CSS": collect_coco_split_dirs,
    "SHWD": collect_voc_capped,
}


def main() -> int:
    global logger
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "merge.yaml")
    args = parser.parse_args()

    logger = setup_logger("INFO")
    config = yaml.safe_load(args.config.read_text())
    seed = int(config["seed"])
    set_seed(seed)
    classes = list(config["class_names"])

    all_records: list[MergeRecord] = []
    for name, spec in config["sources"].items():
        collector = _COLLECTORS.get(name)
        if collector is None:
            logger.error("No collector implemented for source '%s'", name)
            return 1
        records = (
            collector(name, spec, classes, seed)
            if collector is collect_voc_capped
            else collector(name, spec, classes)
        )
        by_split = {s: sum(1 for r in records if r.split == s) for s in ("train", "valid", "test")}
        logger.info("%s: %d images (%s)", name, len(records), by_split)
        all_records.extend(records)

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir)  # rebuild from scratch for determinism
    counts = write_unified_dataset(all_records, output_dir, classes)

    # Commit-able copy of the manifest (datasets/ content itself is git-ignored).
    splits_dir = PROJECT_ROOT / "datasets" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "manifest.csv",
                 splits_dir / f"{output_dir.name}_manifest.csv")

    audit = audit_annotations([r.annotation for r in all_records], output_dir.name)
    audit.save(PROJECT_ROOT / "results" / "audits" / output_dir.name / "audit.json")
    print(audit.summary_table())
    print(f"\nSplit counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
