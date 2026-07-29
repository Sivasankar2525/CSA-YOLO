"""Config-driven detector training launcher for CSA-YOLO Phase 2.

Every experiment is a YAML file under ``configs/experiments/`` — no
hyperparameter lives on the command line, so a run is reproducible from its
config plus the pinned environment.

Usage:
    python training/train_detector.py --experiment configs/experiments/baseline_yolo11n.yaml
    python training/train_detector.py --experiment ... --device cpu   # override for smoke tests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# The restored ICSD focus blocks use adaptive pooling with non-divisible
# sizes, unimplemented on MPS (pytorch#96056); fall back to CPU for that op.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csa_yolo.utils.logging_utils import setup_logger  # noqa: E402
from csa_yolo.utils.reproducibility import set_seed  # noqa: E402

logger = logging.getLogger("train_detector")

REQUIRED_SECTIONS = ("name", "seed", "model", "data", "train")


def load_experiment(path: Path) -> dict[str, Any]:
    """Load and validate an experiment config."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise KeyError(f"Missing required section '{section}' in {path}")
    family = cfg["model"].get("family")
    if family not in ("ultralytics", "icsd"):
        raise ValueError(f"model.family must be 'ultralytics' or 'icsd', got {family!r}")
    return cfg


def build_model(model_cfg: dict[str, Any]):
    """Instantiate the detector named by the experiment's ``model`` section."""
    family = model_cfg["family"]
    if family == "ultralytics":
        from ultralytics import YOLO

        weights = model_cfg.get("weights", "yolo11n.pt")
        logger.info("Building ultralytics model from %s", weights)
        return YOLO(weights)
    from training.icsd import load_icsd_yolo

    return load_icsd_yolo(
        scale=model_cfg.get("scale", "n"), arch=model_cfg.get("arch", "x")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default=None, help="override train.device (smoke tests)")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    args = parser.parse_args()

    cfg = load_experiment(args.experiment)
    setup_logger("INFO")
    logging.basicConfig(level=logging.INFO)
    set_seed(int(cfg["seed"]))

    train_cfg = dict(cfg["train"])
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs

    data_yaml = PROJECT_ROOT / cfg["data"]
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    out_project = PROJECT_ROOT / cfg.get("output", {}).get("project", "results/experiments")
    run_name = cfg["name"]

    model = build_model(cfg["model"])

    logger.info("Starting experiment '%s' (%s)", run_name, args.experiment)
    start = time.perf_counter()
    results = model.train(
        data=str(data_yaml),
        project=str(out_project),
        name=run_name,
        exist_ok=True,
        seed=int(cfg["seed"]),
        deterministic=True,
        verbose=True,
        **train_cfg,
    )
    wall = time.perf_counter() - start

    # Persist a machine-readable summary next to the ultralytics run dir.
    summary: dict[str, Any] = {
        "experiment": run_name,
        "config_file": str(args.experiment),
        "config": cfg,
        "overrides": {"device": args.device, "epochs": args.epochs},
        "wall_seconds": round(wall, 1),
    }
    metrics = getattr(results, "results_dict", None)
    if metrics:
        summary["metrics"] = {k: float(v) for k, v in metrics.items()}
    maps = getattr(results, "maps", None)
    names = getattr(results, "names", None)
    if maps is not None and names:
        summary["per_class_map50_95"] = {
            str(names[i]): float(m) for i, m in enumerate(maps)
        }
    out_file = out_project / run_name / "experiment_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Experiment '%s' finished in %.1fs — summary at %s", run_name, wall, out_file)


if __name__ == "__main__":
    main()
