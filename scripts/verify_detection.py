"""Phase 0 sanity check: run the detection backbone end-to-end on one image.

Verifies the full loop the rest of CSA-YOLO depends on: config loading,
seeding, model loading, inference on the configured device, typed parsing,
and structured output (annotated image + per-detection JSON).

Usage:
    python scripts/verify_detection.py --config configs/default.yaml \
        [--image path/to/image.jpg] [--output-name verify_detection]

If ``--image`` is omitted, the Ultralytics sample image (bus.jpg, which
contains multiple persons) is downloaded and used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from csa_yolo.detection import Detector, FrameDetections
from csa_yolo.utils import load_config, set_seed, setup_logger

SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def annotate(image_path: Path, frame: FrameDetections, output_path: Path) -> None:
    """Draw detections on the image and save it."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for det in frame.detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = (0, 200, 0) if det.class_name == "person" else (0, 140, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{det.class_name} {det.confidence:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def fetch_sample_image(destination: Path) -> Path:
    """Download the Ultralytics sample image if no test image was provided."""
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--image", type=Path, default=None, help="Image to run detection on")
    parser.add_argument("--output-name", type=str, default="verify_detection")
    args = parser.parse_args()

    config = load_config(args.config, project_root=PROJECT_ROOT)
    logger = setup_logger(config.logging_level)
    set_seed(config.seed)

    image_path = args.image or fetch_sample_image(config.paths.results_dir / "sample" / "bus.jpg")

    detector = Detector(
        model_path=config.paths.weights_dir / config.detector.model
        if (config.paths.weights_dir / config.detector.model).exists()
        else config.detector.model,
        confidence_threshold=config.detector.confidence_threshold,
        iou_threshold=config.detector.iou_threshold,
        device=config.detector.device,
        image_size=config.detector.image_size,
    )

    start = time.perf_counter()
    frame = detector.detect(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    persons = frame.by_class(["person"])
    logger.info(
        "Detected %d objects (%d persons) in %.1f ms on device=%s",
        len(frame.detections), len(persons), elapsed_ms, detector.device,
    )
    for det in frame.detections:
        logger.info("  %-12s conf=%.3f box=%s", det.class_name, det.confidence,
                    [round(v, 1) for v in det.xyxy])

    out_dir = config.paths.results_dir / args.output_name
    annotate(image_path, frame, out_dir / "annotated.jpg")
    report = {
        "image": str(image_path),
        "device": detector.device,
        "model": config.detector.model,
        "inference_ms": round(elapsed_ms, 1),
        "num_detections": len(frame.detections),
        "num_persons": len(persons),
        **frame.to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detections.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s and %s", out_dir / "annotated.jpg", out_dir / "detections.json")

    if not frame.detections:
        logger.error("Sanity check FAILED: no detections produced.")
        return 1
    logger.info("Sanity check PASSED: detector outputs boxes with class + confidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
