#!/usr/bin/env python3
"""Ablation runner: runs the pipeline with cumulative module stacks.

Every configuration in ``configs/ablation.yaml`` is executed for real: the
pipeline is built with ``enabled_modules`` so disabled stages do not run.
The detector is loaded once and shared.

This script records OPERATIONAL outputs only (detection, association, track,
compliance and risk counts). Accuracy-type metrics need human ground truth
and are written as ``NOT YET MEASURED`` until that annotation exists.

Usage:
    python evaluation/run_ablation.py \
      --detector weights/baseline_yolo11n_best.pt \
      --images datasets/csa_ppe_v1_clean/images/test \
      --dataset csa_ppe_v1_clean --split test \
      --output results/ablation
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from csa_yolo import __version__
from csa_yolo.detection import Detector, resolve_device
from csa_yolo.pipeline import ALL_MODULES, SafetyPipeline, settings_from_config
from csa_yolo.utils.reproducibility import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ABLATION_CONFIG = PROJECT_ROOT / "configs" / "ablation.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NOT_MEASURED = "NOT YET MEASURED"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ablation_configs(path: Path) -> dict[str, list[str]]:
    """Return {experiment_name: [enabled module names]} from the YAML ladder."""
    with open(path) as f:
        raw = yaml.safe_load(f)["ablation"]
    configs = {name: list(modules or []) for name, modules in raw.items()}
    for name, modules in configs.items():
        unknown = set(modules) - set(ALL_MODULES)
        if unknown:
            raise ValueError(f"{name}: unknown module(s) {sorted(unknown)}")
    return configs


def run_configuration(
    detector: Detector,
    experiment_name: str,
    enabled_modules: list[str],
    image_paths: list[Path],
    output_dir: Path,
    run_metadata: dict,
    settings_raw: dict,
) -> dict:
    """Run one ablation configuration over ``image_paths`` and save its outputs."""
    logger.info("Starting %s (modules: %s)", experiment_name, enabled_modules or "none")
    pipeline = SafetyPipeline.from_config(detector, settings_raw, enabled_modules=enabled_modules)

    counts = Counter()
    compliance_counts = Counter()
    risk_counts = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.jsonl", "w") as results_file:
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                counts["images_unreadable"] += 1
                continue

            result = pipeline.process_frame(frame)
            counts["images_processed"] += 1
            counts["detections"] += len(result.detections.detections)

            workers = result.detections.by_class(pipeline.worker_class_names)
            counts["workers_detected"] += len(workers)
            if workers:
                counts["images_with_workers"] += 1

            counts["associations"] += len(result.associations)
            for assoc in result.associations:
                counts["helmets_associated"] += len(assoc.associated_ppe.get("helmet", []))
                counts["vests_associated"] += len(assoc.associated_ppe.get("vest", []))

            for compliance in result.compliance_results.values():
                compliance_counts[compliance.status] += 1
            for risk in result.worker_risks.values():
                risk_counts[risk.worker_risk_level] += 1

            record = {"image": path.name, **result.to_dict()}
            results_file.write(json.dumps(record) + "\n")

    distinct_tracks = pipeline.tracker.next_track_id if "tracking" in enabled_modules else None
    metrics = {
        "experiment": experiment_name,
        "enabled_modules": enabled_modules,
        "images_processed": counts["images_processed"],
        "images_unreadable": counts["images_unreadable"],
        "images_with_workers": counts["images_with_workers"],
        "detections_total": counts["detections"],
        "workers_detected_total": counts["workers_detected"],
        "associations_total": counts["associations"],
        "helmets_associated_total": counts["helmets_associated"],
        "vests_associated_total": counts["vests_associated"],
        "distinct_track_ids": distinct_tracks,
        "compliance_status_counts": dict(compliance_counts),
        "risk_level_counts": dict(risk_counts),
        "ground_truth_metrics": {
            "wpam_precision": NOT_MEASURED,
            "wpam_recall": NOT_MEASURED,
            "wpam_f1": NOT_MEASURED,
            "tracking_idf1": NOT_MEASURED,
            "tracking_id_switches": NOT_MEASURED,
            "tsmm_false_alarm_reduction": NOT_MEASURED,
            "scrm_accuracy": NOT_MEASURED,
            "scrm_precision": NOT_MEASURED,
            "scrm_recall": NOT_MEASURED,
            "cram": "qualitative case studies only (rule-based prototype)",
        },
    }

    config = {**run_metadata, "experiment_name": experiment_name, "enabled_modules": enabled_modules}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("Saved %s to %s", experiment_name, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", type=Path, required=True, help="Path to detector weights")
    parser.add_argument("--images", type=Path, required=True, help="Directory of evaluation images")
    parser.add_argument("--dataset", default="csa_ppe_v1_clean", help="Dataset name (recorded in config.json)")
    parser.add_argument("--split", default="test", help="Dataset split name (recorded in config.json)")
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    parser.add_argument("--ablation-config", type=Path, default=DEFAULT_ABLATION_CONFIG)
    parser.add_argument("--experiments", nargs="+", default=None, help="Subset of configuration names")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images (quick checks)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml",
                        help="YAML with module parameters (association, tracking, temporal, compliance)")
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.settings) as f:
        settings_raw = yaml.safe_load(f) or {}
    configs = load_ablation_configs(args.ablation_config)
    if args.experiments:
        unknown = set(args.experiments) - set(configs)
        if unknown:
            parser.error(f"unknown experiment(s): {sorted(unknown)}")
        configs = {name: configs[name] for name in args.experiments}

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        parser.error(f"no images found in {args.images}")

    device = resolve_device(args.device)
    detector = Detector(
        model_path=args.detector,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=device,
        image_size=args.imgsz,
    )

    run_metadata = {
        "detector": str(args.detector),
        "dataset": args.dataset,
        "split": args.split,
        "images_dir": str(args.images),
        "num_images_requested": len(image_paths),
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": device,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_version": __version__,
        "settings_file": str(args.settings),
        "module_settings": settings_from_config(settings_raw),
    }

    summary = {}
    for name, modules in configs.items():
        metrics = run_configuration(
            detector, name, modules, image_paths, args.output / name, run_metadata, settings_raw
        )
        summary[name] = {k: metrics[k] for k in (
            "enabled_modules", "images_processed", "detections_total",
            "associations_total", "distinct_track_ids",
            "compliance_status_counts", "risk_level_counts",
        )}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"run": run_metadata, "configurations": summary}, indent=2) + "\n"
    )
    logger.info("Ablation finished: %d configurations", len(summary))


if __name__ == "__main__":
    main()
