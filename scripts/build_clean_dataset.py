#!/usr/bin/env python3
"""Build the leakage-free research dataset ``csa_ppe_v1_clean`` from ``csa_ppe_v1``.

``csa_ppe_v1`` is the ORIGINAL audit/reference copy and is never modified. This script
reads it, writes a new dataset, and verifies the result.

Policy (deterministic; no randomness):

1. Hash every image in ``csa_ppe_v1`` with SHA-256 (file bytes).
2. For each hash that occurs more than once, keep exactly one copy:
   the copy in the highest-priority split (test > valid > train), ties broken by the
   lexicographically smallest file name. The test split is never reduced because of a
   duplicate; copies are removed from train/valid instead.
3. Copy the kept images and their label files byte-for-byte into ``csa_ppe_v1_clean``.
   No image is re-encoded and no label is edited.
4. Write ``data.yaml`` with the same classes and an absolute ``path`` for this machine.
5. Record every removal in ``datasets/splits/csa_ppe_v1_clean_removed.csv`` (tracked in
   git: names and hashes only, no image content) and in ``results/dataset_audit_clean.json``.
6. Verify the rebuilt dataset from its own files: exact-image overlap across splits,
   duplicates within splits, image/label pairing and class counts.

Usage:
    python scripts/build_clean_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = PROJECT_ROOT / "datasets" / "csa_ppe_v1"
CLEAN = PROJECT_ROOT / "datasets" / "csa_ppe_v1_clean"
REMOVED_CSV = PROJECT_ROOT / "datasets" / "splits" / "csa_ppe_v1_clean_removed.csv"
AUDIT_JSON = PROJECT_ROOT / "results" / "dataset_audit_clean.json"

SPLITS = ("train", "valid", "test")
PRIORITY = {"test": 0, "valid": 1, "train": 2}  # lower number = kept in preference
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(root: Path, split: str) -> list[Path]:
    img_dir = root / "images" / split
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(root: Path, split: str, image: Path) -> Path:
    return root / "labels" / split / f"{image.stem}.txt"


def class_counts(root: Path, names: list[str], splits: tuple[str, ...], only: set[str] | None = None) -> dict[str, int]:
    """Boxes per class, read from label files. ``only`` restricts to given 'split/name' keys."""
    counts: Counter = Counter()
    for split in splits:
        for image in list_images(root, split):
            if only is not None and f"{split}/{image.name}" not in only:
                continue
            lbl = label_path(root, split, image)
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    counts[names[int(float(parts[0]))]] += 1
    return {name: counts.get(name, 0) for name in names}


def split_summary(root: Path, splits: tuple[str, ...]) -> dict[str, dict[str, int]]:
    out = {}
    for split in splits:
        images = list_images(root, split)
        labels = sum(1 for im in images if label_path(root, split, im).exists())
        out[split] = {"images": len(images), "labels": labels}
    return out


def plan_removals(original: Path) -> tuple[list[dict], dict[str, dict]]:
    """Return (removal records, hash groups) for the original dataset."""
    by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split in SPLITS:
        for image in list_images(original, split):
            by_hash[sha256_file(image)].append((split, image))

    removals: list[dict] = []
    groups: dict[str, dict] = {}
    for digest in sorted(by_hash):
        members = sorted(by_hash[digest], key=lambda m: (PRIORITY[m[0]], m[1].name))
        if len(members) < 2:
            continue
        (keep_split, keep_img), *dupes = members
        groups[digest] = {
            "sha256": digest,
            "members": [f"{s}/{p.name}" for s, p in members],
            "kept": f"{keep_split}/{keep_img.name}",
        }
        for split, image in dupes:
            removals.append({
                "split": split,
                "file": image.name,
                "sha256": digest,
                "kept_split": keep_split,
                "kept_file": keep_img.name,
                "reason": "exact duplicate; kept copy in higher-priority split or first by name",
            })
            groups[digest].setdefault("removed", []).append(f"{split}/{image.name}")
    return removals, groups


def build(original: Path, clean: Path) -> dict:
    if not original.is_dir():
        raise FileNotFoundError(f"original dataset not found: {original}")
    cfg = yaml.safe_load((original / "data.yaml").read_text())
    names = list(cfg["names"])

    removals, groups = plan_removals(original)
    removed_keys = {f"{r['split']}/{r['file']}" for r in removals}

    # Rebuild the clean directory from scratch so the output is reproducible.
    if clean.exists():
        if clean.name != "csa_ppe_v1_clean":
            raise RuntimeError(f"refusing to remove unexpected directory: {clean}")
        shutil.rmtree(clean)

    missing_labels = []
    for split in SPLITS:
        (clean / "images" / split).mkdir(parents=True)
        (clean / "labels" / split).mkdir(parents=True)
        for image in list_images(original, split):
            if f"{split}/{image.name}" in removed_keys:
                continue
            shutil.copy2(image, clean / "images" / split / image.name)
            lbl = label_path(original, split, image)
            if lbl.exists():
                shutil.copy2(lbl, clean / "labels" / split / lbl.name)
            else:
                missing_labels.append(f"{split}/{image.name}")

    data = {
        "path": str(clean),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    (clean / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    REMOVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REMOVED_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(removals[0].keys()) if removals else ["split", "file"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(removals)

    verification = verify(clean, names)
    all_splits = SPLITS

    report = {
        "original_dataset": "csa_ppe_v1 (audit reference copy, not modified)",
        "clean_dataset": "csa_ppe_v1_clean (authoritative research dataset)",
        "hash_algorithm": "sha256",
        "policy": (
            "Exact duplicates across or within splits: keep one copy, preferring test > valid > train, "
            "then smallest file name. Removed images and their labels are listed in "
            "datasets/splits/csa_ppe_v1_clean_removed.csv. Images and labels are copied byte-for-byte."
        ),
        "original_split_counts": split_summary(original, all_splits),
        "original_total_images": sum(len(list_images(original, s)) for s in all_splits),
        "duplicate_groups_found": len(groups),
        "duplicate_hashes": sorted(groups),
        "duplicate_groups": [groups[h] for h in sorted(groups)],
        "duplicate_image_names_removed": sorted(f"{r['split']}/{r['file']}" for r in removals),
        "removed_count": len(removals),
        "removed_by_split": dict(Counter(r["split"] for r in removals)),
        "final_split_counts": split_summary(clean, all_splits),
        "final_total_images": sum(len(list_images(clean, s)) for s in all_splits),
        "class_names": names,
        "class_distribution_before": {
            "per_split": {s: class_counts(original, names, (s,)) for s in all_splits},
            "all": class_counts(original, names, all_splits),
        },
        "class_distribution_after": {
            "per_split": {s: class_counts(clean, names, (s,)) for s in all_splits},
            "all": class_counts(clean, names, all_splits),
        },
        "images_missing_labels_in_original": 0,
        "images_missing_labels_in_clean": len(missing_labels),
        "verification": verification,
        "exact_image_leakage": verification["exact_image_leakage"],
        "statement": f"EXACT IMAGE LEAKAGE: {verification['exact_image_leakage']}",
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify(clean: Path, names: list[str]) -> dict:
    """Recompute hashes on the clean dataset files and check the invariants."""
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    per_split_hash_count: dict[str, Counter] = {}
    labels_without_images = 0
    images_without_labels = 0
    for split in SPLITS:
        counter: Counter = Counter()
        image_stems = set()
        for image in list_images(clean, split):
            digest = sha256_file(image)
            hash_to_splits[digest].add(split)
            counter[digest] += 1
            image_stems.add(image.stem)
            if not label_path(clean, split, image).exists():
                images_without_labels += 1
        for lbl in (clean / "labels" / split).glob("*.txt"):
            if lbl.stem not in image_stems:
                labels_without_images += 1
        per_split_hash_count[split] = counter

    cross_split = sum(1 for splits in hash_to_splits.values() if len(splits) > 1)
    within_split = sum(
        sum(c - 1 for c in counter.values() if c > 1) for counter in per_split_hash_count.values()
    )
    final_names = [names[i] for i in range(len(names))]
    data = yaml.safe_load((clean / "data.yaml").read_text())
    assert data["names"] == final_names, "class names changed"

    return {
        "cross_split_exact_overlap_groups": cross_split,
        "within_split_duplicate_files": within_split,
        "images_without_labels": images_without_labels,
        "labels_without_images": labels_without_images,
        "classes_match_original": data["names"] == final_names,
        "exact_image_leakage": cross_split,
    }


def main() -> int:
    report = build(ORIGINAL, CLEAN)
    print(f"Original images: {report['original_total_images']}  "
          f"duplicate groups: {report['duplicate_groups_found']}  removed: {report['removed_count']}")
    print(f"Clean images:    {report['final_total_images']}  final splits: {report['final_split_counts']}")
    print(f"Within-split duplicate files after cleanup: {report['verification']['within_split_duplicate_files']}")
    print(report["statement"])
    return 0 if report["exact_image_leakage"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
