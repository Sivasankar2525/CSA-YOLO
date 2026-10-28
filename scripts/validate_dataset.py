#!/usr/bin/env python3
"""Machine-readable validation of a YOLO-format dataset (read-only).

Checks, per split (train / valid / test):
  - images and label files exist; image/label pairing in both directions
  - malformed label lines (not 5 fields, non-numeric)
  - class ids outside the dataset's class list
  - bounding boxes: coordinates outside [0, 1], non-positive width/height,
    box extending beyond the image by more than a tolerance
  - unreadable images
  - exact-image (SHA-256) overlap across splits and duplicates within a split

It does NOT check near-duplicates (NOT MEASURED) and does NOT change any data.
Counts are compared against the expected values given on the command line but
are never forced to match: a mismatch is recorded as a finding.

Usage:
    python scripts/validate_dataset.py --root datasets/csa_ppe_v1_clean \
        --expect train=1481 valid=210 test=195 --out results/dataset_validation_clean.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import yaml

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BOX_TOLERANCE = 0.01  # fraction of image size a box may overhang before it is flagged


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label_file(path: Path, class_names: list[str], problems: Counter,
                        examples: list, class_counts: Counter) -> int:
    """Check one label file. Returns the number of valid boxes; adds valid boxes to class_counts."""
    valid = 0
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems["malformed_line_field_count"] += 1
            _note(examples, "malformed_line_field_count", path, line_no, line)
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems["malformed_line_non_numeric"] += 1
            _note(examples, "malformed_line_non_numeric", path, line_no, line)
            continue
        if not 0 <= cls < len(class_names):
            problems["class_id_out_of_range"] += 1
            _note(examples, "class_id_out_of_range", path, line_no, line)
            continue
        if w <= 0 or h <= 0:
            problems["box_non_positive_size"] += 1
            _note(examples, "box_non_positive_size", path, line_no, line)
            continue
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            problems["box_coordinate_outside_unit_interval"] += 1
            _note(examples, "box_coordinate_outside_unit_interval", path, line_no, line)
            continue
        if xc - w / 2 < -BOX_TOLERANCE or xc + w / 2 > 1 + BOX_TOLERANCE or \
           yc - h / 2 < -BOX_TOLERANCE or yc + h / 2 > 1 + BOX_TOLERANCE:
            problems["box_overhangs_image"] += 1
            _note(examples, "box_overhangs_image", path, line_no, line)
        class_counts[class_names[cls]] += 1
        valid += 1
    return valid


def _note(examples: list, kind: str, path: Path, line_no: int, line: str) -> None:
    if len(examples) < 50:
        examples.append({"problem": kind, "file": path.name, "line": line_no, "content": line.strip()})


def validate_split(root: Path, split: str, class_names: list[str], examples: list) -> dict:
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    problems: Counter = Counter()
    class_counts: Counter = Counter()
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if img_dir.is_dir() else []
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for image in images:
        lbl = lbl_dir / f"{image.stem}.txt"
        if not lbl.is_file():
            problems["image_without_label"] += 1
            _note(examples, "image_without_label", image, 0, "")
        if cv2.imread(str(image)) is None:
            problems["unreadable_image"] += 1
            _note(examples, "unreadable_image", image, 0, "")
    for lbl in labels:
        if lbl.stem not in image_stems:
            problems["label_without_image"] += 1
            _note(examples, "label_without_image", lbl, 0, "")

    total_boxes = 0
    empty_labels = 0
    for lbl in labels:
        if not any(l.strip() for l in lbl.read_text().splitlines()):
            empty_labels += 1
        total_boxes += validate_label_file(lbl, class_names, problems, examples, class_counts)

    return {
        "images": len(images),
        "label_files": len(labels),
        "valid_boxes": total_boxes,
        "empty_label_files": empty_labels,
        "boxes_per_class": {name: class_counts.get(name, 0) for name in class_names},
        "problems": dict(problems),
    }


def exact_overlap(root: Path) -> dict:
    """SHA-256 of every image; report cross-split and within-split duplicates."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for image in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            by_hash[sha256_file(image)].append(f"{split}/{image.name}")
    cross = {h: f for h, f in by_hash.items() if len({m.split("/")[0] for m in f}) > 1}
    within = {h: f for h, f in by_hash.items() if len(f) > 1 and h not in cross}
    return {
        "exact_image_leakage": len(cross),  # groups whose members span >1 split
        "cross_split_groups": [sorted(f) for f in cross.values()][:50],
        "within_split_duplicate_groups": len(within),
        "within_split_duplicate_files": sum(len(f) - 1 for f in within.values()),
        "near_duplicate_leakage": "NOT MEASURED (exact hashing only)",
    }


def validate_dataset(root: Path, expect: dict[str, int]) -> dict:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found in {root}")
    cfg = yaml.safe_load(data_yaml.read_text())
    class_names = list(cfg["names"])
    examples: list = []

    splits = {s: validate_split(root, s, class_names, examples) for s in SPLITS}
    for s in SPLITS:
        if s in expect:
            splits[s]["expected_images"] = expect[s]
            splits[s]["count_matches_expected"] = splits[s]["images"] == expect[s]
    overlap = exact_overlap(root)

    total_images = sum(v["images"] for v in splits.values())
    problems_total = sum(sum(v["problems"].values()) for v in splits.values())
    return {
        "dataset_root": str(root),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_yaml_names": class_names,
        "splits": splits,
        "total_images": total_images,
        "total_valid_boxes": sum(v["valid_boxes"] for v in splits.values()),
        "expected_total": sum(expect.values()) if expect else None,
        "total_problems": problems_total,
        "overlap": overlap,
        "examples": examples,
        "verdict": "PASS" if (problems_total == 0 and overlap["exact_image_leakage"] == 0) else "FINDINGS PRESENT",
    }


def _parse_expect(items: Optional[list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in SPLITS or not value.isdigit():
            raise argparse.ArgumentTypeError(f"expected SPLIT=COUNT, got {item!r}")
        out[key] = int(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect", nargs="*", default=None, help="e.g. train=1481 valid=210 test=195")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = validate_dataset(args.root, _parse_expect(args.expect))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"total images: {report['total_images']}  boxes: {report['total_valid_boxes']}")
    print(f"problems: {report['total_problems']}  exact leakage: {report['overlap']['exact_image_leakage']}")
    print(f"verdict: {report['verdict']}  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
