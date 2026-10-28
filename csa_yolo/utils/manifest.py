"""Experiment manifest helpers: environment, hardware and git state, and metric extraction.

Everything recorded here is read from the running system or from a result object.
Nothing is estimated. A value that cannot be read is recorded as ``None`` and listed
in ``unavailable`` so the gap is visible.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": None if status is None else bool(status),
        "submodule_icsd_yolo": _run(["git", "submodule", "status", "external/icsd_yolo"]),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "torch_cuda": None,
        "ultralytics": None,
        "numpy": None,
        "opencv": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    for module, key in (("ultralytics", "ultralytics"), ("numpy", "numpy"), ("cv2", "opencv")):
        try:
            mod = __import__(module)
            info[key] = getattr(mod, "__version__", None)
        except ImportError:
            pass
    return info


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False, "gpu_name": None, "gpu_memory_gb": None,
                            "mps_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(cuda_available=True, gpu_name=props.name,
                        gpu_memory_gb=round(props.total_memory / 1024**3, 2))
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        pass
    return info


def extract_detector_metrics(metrics: Any) -> dict[str, Any]:
    """Pull P/R/F1/mAP and per-class values from an Ultralytics validation result.

    Works on the object returned by ``model.val()`` (``metrics.box``). Each field is
    ``None`` when the object does not provide it. Nothing is computed here.
    """
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", None) or {}

    def _f(obj, attr) -> Optional[float]:
        value = getattr(obj, attr, None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _per_class(array_attr: str) -> Optional[dict[str, float]]:
        arr = getattr(box, array_attr, None) if box is not None else None
        classes = getattr(box, "ap_class_index", None)
        if arr is None or classes is None:
            return None
        return {str(names.get(int(c), int(c))): float(v) for c, v in zip(classes, arr)}

    summary = {
        "precision": _f(box, "mp"),
        "recall": _f(box, "mr"),
        "mAP@0.5": _f(box, "map50"),
        "mAP@0.5:0.95": _f(box, "map"),
        # Ultralytics exposes F1 per class only (no overall scalar); the overall value stays None.
        "F1": None,
        "per_class_precision": _per_class("p"),
        "per_class_recall": _per_class("r"),
        "per_class_f1": _per_class("f1"),
        "per_class_AP50": _per_class("ap50"),
        "per_class_AP50-95": _per_class("maps"),
    }
    unavailable = [k for k, v in summary.items() if v is None]
    summary["unavailable"] = unavailable
    return summary


def build_manifest(
    *,
    experiment: dict[str, Any],
    effective_train: dict[str, Any],
    dataset_yaml: Path,
    resolved_dataset_yaml: Path,
    output_dir: Path,
    seed: int,
    timings: dict[str, Any],
    optimizer: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the reproducibility manifest for one run (JSON-serialisable)."""
    return {
        "experiment": experiment.get("name"),
        "git": git_state(),
        "environment": environment_info(),
        "hardware": hardware_info(),
        "dataset": {
            "config_path": str(dataset_yaml),
            "resolved_yaml": str(resolved_dataset_yaml),
            "name": Path(dataset_yaml).parent.name,
        },
        "model": experiment.get("model"),
        "initialization": _initialization(experiment.get("model", {})),
        "seed": seed,
        "effective_train": effective_train,
        "optimizer": optimizer,
        "evaluation": evaluation,
        "timings": timings,
        "output_dir": str(output_dir),
    }


def _initialization(model_cfg: dict[str, Any]) -> str:
    family = model_cfg.get("family")
    weights = str(model_cfg.get("weights", ""))
    if family == "icsd":
        return "random (restored ICSD-YOLO architecture; no released weights)"
    if weights.endswith(".yaml"):
        return "random (architecture YAML only, no pretrained weights)"
    if weights:
        return f"pretrained weights: {weights}"
    return "unspecified"
