#!/usr/bin/env python3
"""Quick demo of the CSA-YOLO safety pipeline on a single image."""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from csa_yolo.detection import Detector, resolve_device
from csa_yolo.pipeline import SafetyPipeline
from csa_yolo.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_detections_on_frame(
    frame: np.ndarray,
    pipeline_result,
) -> np.ndarray:
    """Draw annotations on frame."""
    annotated = frame.copy()

    # Draw worker and PPE boxes
    for detection in pipeline_result.detections.detections:
        x1, y1, x2, y2 = detection.xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Color by class
        if detection.class_name.lower() in ["person"]:
            color = (0, 255, 0)  # Green
        elif "helmet" in detection.class_name.lower():
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Draw tracked worker IDs and compliance status
    for track_id, compliance in pipeline_result.compliance_results.items():
        if track_id in pipeline_result.tracked_workers:
            detection = pipeline_result.tracked_workers[track_id]
            x1, y1, x2, y2 = detection.xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            status_color = (0, 255, 0) if compliance.status == "COMPLIANT" else (0, 0, 255)
            status_text = f"ID{track_id} {compliance.status}"
            cv2.putText(
                annotated,
                status_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="CSA-YOLO demo")
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output",
        help="Output image path (optional)",
    )
    parser.add_argument(
        "--json-output",
        help="Output JSON results (optional)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Initialize detector
    device = resolve_device(config.detector.device)
    logger.info(f"Using device: {device}")

    detector = Detector(
        model_path=str(config.paths.weights_dir / config.detector.model),
        confidence_threshold=config.detector.confidence_threshold,
        iou_threshold=config.detector.iou_threshold,
        device=device,
        image_size=config.detector.image_size,
    )
    logger.info("Detector initialized")

    # Initialize pipeline
    pipeline = SafetyPipeline(detector)
    logger.info("Pipeline initialized")

    # Load and process image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    logger.info(f"Processing {image_path} ({frame.shape[0]}x{frame.shape[1]})")

    # Process through pipeline
    result = pipeline.process_frame(frame)
    logger.info(f"Detected {len(result.detections.detections)} objects")
    logger.info(f"Tracked {len(result.tracked_workers)} workers")

    # Print results
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    for detection in result.detections.detections:
        print(
            f"  {detection.class_name:12} conf={detection.confidence:.3f} "
            f"box={detection.xyxy}"
        )

    if result.compliance_results:
        print("\n" + "=" * 60)
        print("COMPLIANCE ANALYSIS")
        print("=" * 60)
        for track_id, compliance in result.compliance_results.items():
            print(f"\nWorker #{track_id}:")
            print(f"  Status: {compliance.status}")
            print(f"  Confidence: {compliance.confidence:.3f}")
            if compliance.missing_ppe:
                print(f"  Missing: {', '.join(compliance.missing_ppe)}")
            print(f"  Reason: {compliance.reason}")

    if result.worker_risks:
        print("\n" + "=" * 60)
        print("RISK ASSESSMENT")
        print("=" * 60)
        for track_id, risk in result.worker_risks.items():
            print(f"\nWorker #{track_id}:")
            print(f"  Risk Score: {risk.worker_risk_score:.1f} / 100")
            print(f"  Risk Level: {risk.worker_risk_level}")
            if risk.risk_factors:
                print(f"  Factors: {'; '.join(risk.risk_factors)}")

    # Optionally save annotated image
    if args.output:
        annotated = draw_detections_on_frame(frame, result)
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"Saved annotated image to {output_path}")

    # Optionally save JSON results
    if args.json_output:
        output_dict = result.to_dict()
        output_path = Path(args.json_output)
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        logger.info(f"Saved JSON results to {output_path}")

    print("\n" + "=" * 60)
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
