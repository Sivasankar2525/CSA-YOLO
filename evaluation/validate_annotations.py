#!/usr/bin/env python3
"""Validate human ground-truth annotation files against the schema in docs/EVALUATION.md.

The validator never creates or edits annotations. It reports every problem it finds, with
the location, so an annotator can fix the file. A file that passes is only *well-formed*;
it is not evidence of anything until the metrics are computed from it.

Usage:
    python evaluation/validate_annotations.py annotations/wpam.json
    python evaluation/validate_annotations.py --module scrm annotations/scrm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_STATUS = {"COMPLIANT", "NON_COMPLIANT"}
PPE_CLASSES = ("helmet", "vest")
REQUIRED_META = ("annotator", "annotation_date")


def _infer_module(path: Path) -> str:
    stem = path.stem.lower()
    for module in ("wpam", "scrm", "tracking", "tsmm"):
        if stem.startswith(module):
            return module
    raise ValueError(f"cannot infer module from file name {path.name!r}; pass --module")


def validate_wpam(data: dict[str, Any]) -> list[str]:
    errors = []
    images = data.get("images")
    if not isinstance(images, dict) or not images:
        return ["'images' must be a non-empty object"]
    for image_id, workers in images.items():
        if not isinstance(workers, dict) or "workers" not in workers:
            errors.append(f"images.{image_id}: must contain 'workers'")
            continue
        for worker_id, ppe in workers["workers"].items():
            if not isinstance(ppe, dict):
                errors.append(f"images.{image_id}.workers.{worker_id}: must be an object")
                continue
            extra = set(ppe) - set(PPE_CLASSES)
            if extra:
                errors.append(f"images.{image_id}.workers.{worker_id}: unknown PPE keys {sorted(extra)}")
            for cls in PPE_CLASSES:
                if cls not in ppe:
                    errors.append(f"images.{image_id}.workers.{worker_id}: missing '{cls}' (use null if none)")
                elif ppe[cls] is not None and not isinstance(ppe[cls], str):
                    errors.append(f"images.{image_id}.workers.{worker_id}.{cls}: must be a string id or null")
    return errors


def validate_scrm(data: dict[str, Any]) -> list[str]:
    errors = []
    images = data.get("images")
    if not isinstance(images, dict) or not images:
        return ["'images' must be a non-empty object"]
    for image_id, workers in images.items():
        if not isinstance(workers, dict):
            errors.append(f"images.{image_id}: must map worker ids to statuses")
            continue
        for worker_id, status in workers.items():
            if status not in ALLOWED_STATUS:
                errors.append(f"images.{image_id}.{worker_id}: status must be one of {sorted(ALLOWED_STATUS)}, "
                              f"got {status!r} (UNKNOWN is a system output, not a human label)")
    return errors


def validate_tracking(data: dict[str, Any]) -> list[str]:
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        return ["'frames' must be a non-empty list in time order"]
    errors = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"frames[{i}]: must be an object mapping gt_worker_id -> track_id")
            continue
        for gt_id, track in frame.items():
            if not isinstance(track, (str, int)):
                errors.append(f"frames[{i}].{gt_id}: track id must be a string or integer")
    return errors


def validate_tsmm(data: dict[str, Any]) -> list[str]:
    clips = data.get("clips")
    if not isinstance(clips, dict) or not clips:
        return ["'clips' must be a non-empty object"]
    errors = []
    for clip_id, clip in clips.items():
        if not isinstance(clip, dict) or not isinstance(clip.get("frame_status"), dict):
            errors.append(f"clips.{clip_id}: must contain 'frame_status' {{worker_id: [status per frame]}}")
            continue
        for worker_id, seq in clip["frame_status"].items():
            if not isinstance(seq, list) or any(s not in ALLOWED_STATUS for s in seq):
                errors.append(f"clips.{clip_id}.frame_status.{worker_id}: every entry must be one of "
                              f"{sorted(ALLOWED_STATUS)}")
        for interval in clip.get("occlusion_intervals", []):
            if not (isinstance(interval, list) and len(interval) == 2 and interval[0] <= interval[1]):
                errors.append(f"clips.{clip_id}: occlusion interval {interval} must be [start, end] with start<=end")
    return errors


VALIDATORS = {"wpam": validate_wpam, "scrm": validate_scrm, "tracking": validate_tracking, "tsmm": validate_tsmm}


def validate_file(path: Path, module: str | None = None) -> list[str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read JSON: {exc}"]
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    module = module or _infer_module(path)
    errors = [f"missing metadata '{key}'" for key in REQUIRED_META if not data.get(key)]
    errors += VALIDATORS[module](data)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--module", choices=sorted(VALIDATORS), default=None)
    args = parser.parse_args()
    exit_code = 0
    for path in args.paths:
        errors = validate_file(path, args.module)
        if errors:
            exit_code = 1
            print(f"{path}: {len(errors)} problem(s)")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"{path}: well-formed (this is not a metric; see docs/EVALUATION.md)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
