#!/usr/bin/env python3
"""Ablation study runner — evaluate pipeline with modules enabled/disabled.

Usage:
    python evaluation/run_ablation.py \
      --detector weights/baseline_yolo11n_best.pt \
      --images datasets/csa_ppe_v1/images/test \
      --output results/ablation \
      --experiments detector_only wpam tracking tsmm scrm full_csa_yolo
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from csa_yolo.detection import Detector, resolve_device
from csa_yolo.pipeline import SafetyPipeline
from csa_yolo.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationPipeline:
    """Pipeline wrapper that enables/disables modules for ablation studies."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self.pipeline = SafetyPipeline(detector)

    def process_frame_ablated(
        self,
        frame: np.ndarray,
        enable_wpam: bool = True,
        enable_tracking: bool = True,
        enable_tsmm: bool = True,
        enable_scrm: bool = True,
        enable_cram: bool = True,
    ):
        """Process frame with specified modules enabled/disabled.

        For now, this is a placeholder — the actual ablation requires
        refactoring the pipeline to allow module disabling.

        Current workaround: instantiate a fresh pipeline for each ablation
        and manually skip module outputs.
        """
        result = self.pipeline.process_frame(frame)

        # Simulate module disabling by zeroing outputs
        if not enable_wpam:
            result.associations = []
        if not enable_tracking:
            result.tracked_workers = {}
        if not enable_tsmm:
            result.temporal_states = {}
        if not enable_scrm:
            result.compliance_results = {}
        if not enable_cram:
            result.worker_risks = {}

        return result


def run_ablation_experiment(
    detector_path: Path,
    image_dir: Path,
    output_dir: Path,
    experiment_name: str,
    ablation_config: dict,
    max_images: Optional[int] = None,
) -> dict:
    """Run a single ablation experiment.

    Args:
        detector_path: Path to detector weights.
        image_dir: Directory of test images.
        output_dir: Directory to save results.
        experiment_name: Name of experiment (e.g., "detector_only").
        ablation_config: Dict of {module_name: enabled}.
        max_images: Max images to process (for quick testing).

    Returns:
        Dict of metrics (currently placeholder).
    """
    logger.info(f"Starting ablation experiment: {experiment_name}")
    logger.info(f"  WPAM: {ablation_config['wpam']}")
    logger.info(f"  Tracking: {ablation_config['tracking']}")
    logger.info(f"  TSMM: {ablation_config['tsmm']}")
    logger.info(f"  SCRM: {ablation_config['scrm']}")
    logger.info(f"  CRAM: {ablation_config['cram']}")

    # Initialize detector
    device = resolve_device("auto")
    logger.info(f"Using device: {device}")

    detector = Detector(
        model_path=str(detector_path),
        device=device,
    )

    ablation = AblationPipeline(detector)

    # Process images
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if max_images:
        image_files = image_files[:max_images]

    logger.info(f"Processing {len(image_files)} images")

    results = []
    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        result = ablation.process_frame_ablated(
            frame,
            enable_wpam=ablation_config["wpam"],
            enable_tracking=ablation_config["tracking"],
            enable_tsmm=ablation_config["tsmm"],
            enable_scrm=ablation_config["scrm"],
            enable_cram=ablation_config["cram"],
        )
        results.append(result.to_dict())

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1} / {len(image_files)}")

    # Aggregate metrics (placeholder)
    metrics = {
        "experiment": experiment_name,
        "images_processed": len(results),
        "modules_enabled": ablation_config,
        "placeholder_metrics": {
            "note": "Actual metrics require ground truth annotations",
            "wpam_accuracy": "NOT YET MEASURED",
            "compliance_accuracy": "NOT YET MEASURED",
            "risk_correlation": "NOT YET MEASURED",
        },
    }

    # Save results
    output_dir = Path(output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(ablation_config, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        type=Path,
        required=True,
        help="Path to detector weights",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory of test images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="results/ablation",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to process (for quick testing)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "detector_only",
            "wpam",
            "tracking",
            "tsmm",
            "scrm",
            "full_csa_yolo",
        ],
        help="Ablation experiments to run",
    )
    args = parser.parse_args()

    # Define ablation configurations
    ablation_configs = {
        "detector_only": {
            "wpam": False,
            "tracking": False,
            "tsmm": False,
            "scrm": False,
            "cram": False,
        },
        "wpam": {
            "wpam": True,
            "tracking": False,
            "tsmm": False,
            "scrm": False,
            "cram": False,
        },
        "tracking": {
            "wpam": True,
            "tracking": True,
            "tsmm": False,
            "scrm": False,
            "cram": False,
        },
        "tsmm": {
            "wpam": True,
            "tracking": True,
            "tsmm": True,
            "scrm": False,
            "cram": False,
        },
        "scrm": {
            "wpam": True,
            "tracking": True,
            "tsmm": True,
            "scrm": True,
            "cram": False,
        },
        "full_csa_yolo": {
            "wpam": True,
            "tracking": True,
            "tsmm": True,
            "scrm": True,
            "cram": True,
        },
    }

    # Run requested experiments
    all_metrics = {}
    for experiment_name in args.experiments:
        if experiment_name not in ablation_configs:
            logger.warning(f"Unknown experiment: {experiment_name}")
            continue

        config = ablation_configs[experiment_name]
        metrics = run_ablation_experiment(
            args.detector,
            args.images,
            args.output,
            experiment_name,
            config,
            max_images=args.max_images,
        )
        all_metrics[experiment_name] = metrics

    # Save aggregate metrics
    summary_file = args.output / "ablation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved ablation summary to {summary_file}")


if __name__ == "__main__":
    main()
