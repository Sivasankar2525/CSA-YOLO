#!/usr/bin/env python3
"""Run the full CSA-YOLO pipeline on a video, an image, or a folder of images.

Usage:
    python scripts/run_pipeline.py --source path/to/video.mp4 --detector weights/best.pt \
        --out results/runs/site_a --camera-id site-a-cam1

Writes frames.jsonl, workers.csv, summary.json and an annotated video/images to --out.
Module parameters come from configs/default.yaml (same file used by training and ablation).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csa_yolo.detection import Detector, resolve_device  # noqa: E402
from csa_yolo.pipeline import SafetyPipeline  # noqa: E402
from csa_yolo.runner import open_source, run_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True, help="detector weights (.pt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--camera-id", default="camera-0")
    parser.add_argument("--fps", type=float, default=None, help="override fps (videos only)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-annotated", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.detector.is_file():
        parser.error(f"detector weights not found: {args.detector}")
    raw_settings = yaml.safe_load(args.settings.read_text()) or {}

    try:
        kind, frames, fps = open_source(args.source, fps_override=args.fps)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("cannot read input: %s", exc)
        return 2

    device = resolve_device(args.device)
    detector = Detector(model_path=args.detector, confidence_threshold=args.conf,
                        iou_threshold=args.iou, device=device, image_size=args.imgsz)
    pipeline = SafetyPipeline.from_config(detector, raw_settings)

    summary = run_source(
        pipeline, frames, args.out,
        source_name=str(args.source), camera_id=args.camera_id, fps=fps, kind=kind,
        write_annotated=not args.no_annotated, max_frames=args.max_frames,
    )
    summary["detector_weights"] = str(args.detector)
    summary["device"] = device
    summary["settings_file"] = str(args.settings)
    import json
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("processed %d/%d frames -> %s", summary["frames_processed"], summary["frames_seen"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
