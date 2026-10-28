"""Config-driven detector training and evaluation launcher for CSA-YOLO.

Every experiment is a YAML file under ``configs/experiments/``. Hyperparameters are read
from that file. Command-line options only override the device and epoch count, for
Colab GPU selection and smoke tests.

Outputs, per experiment, under ``results/experiments/<name>/``:
  experiment_summary.json   training summary, validation metrics, effective config
  val_metrics.json          metrics on the validation split with the fixed evaluation settings
  eval_val/, eval_test/     Ultralytics evaluation folders (confusion matrix, per-class plots)
  test_metrics.json         metrics on the test split with the same settings (the reported result)
  manifest.json             git commit, environment, hardware, seed, timings, dataset path
  data_resolved.yaml        dataset YAML with the local dataset folder as its root
  weights/best.pt           best checkpoint (Ultralytics)
  results.csv, *.png        training curves and confusion matrix (Ultralytics)

Usage:
    python training/train_detector.py --experiment configs/experiments/baseline_yolo11n.yaml
    python training/train_detector.py --experiment ... --device cuda --epochs 1      # smoke test
    python training/train_detector.py --experiment ... --eval-only results/experiments/<name>/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

# The restored ICSD focus blocks use adaptive pooling with non-divisible
# sizes, unimplemented on MPS (pytorch#96056); fall back to CPU for that op.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from csa_yolo.utils.logging_utils import setup_logger  # noqa: E402
from csa_yolo.utils.manifest import build_manifest, extract_detector_metrics  # noqa: E402
from csa_yolo.utils.reproducibility import set_seed  # noqa: E402

logger = logging.getLogger("train_detector")

REQUIRED_SECTIONS = ("name", "seed", "model", "data", "train")

# Evaluation settings are identical for every experiment. Do not change them per model.
EVALUATION_SETTINGS = {"conf": 0.001, "iou": 0.7}


class LauncherError(RuntimeError):
    """Raised with an actionable message when a launcher precondition fails."""


def load_experiment(path: Path) -> dict[str, Any]:
    """Load and validate an experiment config."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}. Check the path under configs/experiments/.")
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise KeyError(f"Missing required section '{section}' in {path}")
    family = cfg["model"].get("family")
    if family not in ("ultralytics", "icsd"):
        raise ValueError(f"model.family must be 'ultralytics' or 'icsd', got {family!r}")
    return cfg


def apply_overrides(train_cfg: dict[str, Any], device: Optional[str], epochs: Optional[int]) -> dict[str, Any]:
    """Return a copy of ``train_cfg`` with command-line overrides applied (None = keep YAML)."""
    effective = dict(train_cfg)
    if device is not None:
        effective["device"] = device
    if epochs is not None:
        if epochs < 1:
            raise ValueError(f"--epochs must be >= 1, got {epochs}")
        effective["epochs"] = epochs
    return effective


def resolve_data_yaml(data_yaml: Path, out_dir: Path) -> Path:
    """Write a copy of the dataset YAML whose ``path`` is the folder that holds the data.

    The committed/derived data.yaml may hold an absolute path from another machine.
    The folder containing the YAML is the authoritative root, so this writes
    ``out_dir/data_resolved.yaml`` with ``path`` set to that folder.
    """
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(data_yaml.resolve().parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / "data_resolved.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False))
    return resolved


def check_dataset_folders(resolved_yaml: Path) -> None:
    """Fail early, with a clear message, when the dataset folders are missing."""
    data = yaml.safe_load(resolved_yaml.read_text())
    root = Path(data["path"])
    for key in ("train", "val", "test"):
        folder = root / data[key]
        if not folder.is_dir():
            raise LauncherError(
                f"dataset split folder missing: {folder}. Upload datasets/csa_ppe_v1_clean to the "
                f"Colab clone (see docs/COLAB_TRAINING_NOTEBOOK.md, Cell 4)."
            )


def build_model(model_cfg: dict[str, Any]):
    """Instantiate the detector named by the experiment's ``model`` section."""
    family = model_cfg["family"]
    if family == "ultralytics":
        from ultralytics import YOLO

        weights = model_cfg.get("weights", "yolo11n.pt")
        logger.info("Building ultralytics model from %s", weights)
        return YOLO(weights)
    from training.icsd import load_icsd_yolo

    return load_icsd_yolo(scale=model_cfg.get("scale", "n"), arch=model_cfg.get("arch", "x"))


def evaluate_split(model: Any, data_yaml: Path, split: str, imgsz: int, device: str,
                   run_dir: Path) -> dict[str, Any]:
    """Evaluate ``model`` on one split with the fixed evaluation settings.

    ``split`` is the Ultralytics key ("val" or "test"). Results go to
    ``run_dir/<split>_metrics.json`` and plots to ``run_dir/eval_<split>/``, so the
    two splits never overwrite each other.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir_name = f"eval_{split}"
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, device=device,
        project=str(run_dir), name=plot_dir_name, exist_ok=True, plots=True,
        verbose=False, **EVALUATION_SETTINGS,
    )
    result = extract_detector_metrics(metrics)
    result["split"] = split
    result["evaluation_settings"] = {"imgsz": imgsz, "device": device, **EVALUATION_SETTINGS}
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def optimizer_record(model: Any) -> dict[str, Any]:
    """Read the optimizer actually used by the trainer (None values if unavailable)."""
    trainer = getattr(model, "trainer", None)
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return {"name": None, "lr0": None, "note": "trainer optimizer not available"}
    groups = getattr(optimizer, "param_groups", [])
    return {
        "name": type(optimizer).__name__,
        "lr0": float(groups[0]["lr"]) if groups else None,
        "momentum": float(groups[0].get("momentum", 0.0)) if groups else None,
        "weight_decay": float(groups[0].get("weight_decay", 0.0)) if groups else None,
    }


def run_experiment(args: argparse.Namespace) -> Path:
    cfg = load_experiment(args.experiment)
    setup_logger("INFO")
    logging.basicConfig(level=logging.INFO)
    seed = int(cfg["seed"])
    set_seed(seed)

    train_cfg = apply_overrides(cfg["train"], args.device, args.epochs)
    data_yaml = PROJECT_ROOT / cfg["data"]
    out_project = PROJECT_ROOT / cfg.get("output", {}).get("project", "results/experiments")
    run_name = cfg["name"]
    run_dir = out_project / run_name
    resolved = resolve_data_yaml(data_yaml, run_dir)
    check_dataset_folders(resolved)
    device = train_cfg.get("device", "cpu")
    imgsz = int(train_cfg.get("imgsz", 640))

    timings: dict[str, Any] = {}
    if args.eval_only:
        weights = Path(args.eval_only)
        if not weights.is_file():
            raise FileNotFoundError(f"--eval-only weights not found: {weights}")
        from ultralytics import YOLO

        model = YOLO(str(weights))
        logger.info("Evaluation only: %s", weights)
        optimizer = {"name": None, "lr0": None, "note": "eval-only run; training not performed"}
        train_summary: dict[str, Any] = {"evaluation_only_weights": str(weights)}
    else:
        model = build_model(cfg["model"])
        logger.info("Starting experiment '%s' (%s)", run_name, args.experiment)
        start = time.perf_counter()
        try:
            results = model.train(
                data=str(resolved), project=str(out_project), name=run_name, exist_ok=True,
                seed=seed, deterministic=True, verbose=True, **train_cfg,
            )
        except Exception as exc:
            raise LauncherError(
                f"training failed for '{run_name}' on device '{device}': {type(exc).__name__}: {exc}. "
                f"Check GPU availability (--device) and the dataset folders."
            ) from exc
        timings["train_wall_seconds"] = round(time.perf_counter() - start, 1)
        optimizer = optimizer_record(model)
        train_summary = {"results_dict": {k: float(v) for k, v in (getattr(results, "results_dict", None) or {}).items()}}
        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            model = load_yolo(best)

    eval_start = time.perf_counter()
    validation = evaluate_split(model, resolved, "val", imgsz, device, run_dir)
    test = None
    if not args.no_test_eval:
        test = evaluate_split(model, resolved, "test", imgsz, device, run_dir)
    timings["evaluation_wall_seconds"] = round(time.perf_counter() - eval_start, 1)

    weights_path = run_dir / "weights" / "best.pt"
    manifest = build_manifest(
        experiment=cfg, effective_train=train_cfg, dataset_yaml=data_yaml,
        resolved_dataset_yaml=resolved, output_dir=run_dir, seed=seed, timings=timings,
        optimizer=optimizer,
        evaluation={"settings": {"imgsz": imgsz, "device": device, **EVALUATION_SETTINGS},
                    "split_reported": "test", "weights": str(weights_path) if weights_path.is_file() else None,
                    "eval_only": bool(args.eval_only)},
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary: dict[str, Any] = {
        "experiment": run_name,
        "config_file": str(args.experiment),
        "config": cfg,
        "overrides": {"device": args.device, "epochs": args.epochs, "eval_only": args.eval_only},
        "effective_train": train_cfg,
        "dataset_yaml": str(data_yaml),
        "resolved_dataset_yaml": str(resolved),
        "seed": seed,
        "optimizer": optimizer,
        "timings": timings,
        "training": train_summary,
        "validation": validation,
        "test": test,
        # Kept for backwards compatibility with the metrics builder (training-time validation values).
        "metrics": train_summary.get("results_dict") or None,
        "manifest": str((run_dir / "manifest.json").relative_to(PROJECT_ROOT)),
    }
    out_file = run_dir / "experiment_summary.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Experiment '%s' finished — summary at %s", run_name, out_file)
    return out_file


def load_yolo(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default=None, help="override train.device, e.g. cuda on Colab")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests)")
    parser.add_argument("--eval-only", type=Path, default=None, metavar="WEIGHTS",
                        help="skip training; evaluate these weights on val and test")
    parser.add_argument("--no-test-eval", action="store_true", help="skip the test-split evaluation")
    args = parser.parse_args()
    try:
        run_experiment(args)
    except (FileNotFoundError, KeyError, ValueError, LauncherError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
