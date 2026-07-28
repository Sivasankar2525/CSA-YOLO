"""Typed configuration loading for the CSA-YOLO pipeline.

All scripts receive their parameters from a YAML file (see ``configs/``);
this module parses and validates that file into typed dataclasses so that
misconfiguration fails loudly at startup instead of silently mid-experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, resolved relative to the project root."""

    weights_dir: Path
    results_dir: Path
    datasets_dir: Path


@dataclass(frozen=True)
class DetectorConfig:
    """Detector backbone settings."""

    model: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    image_size: int = 640

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1), got {self.confidence_threshold}"
            )
        if not 0.0 < self.iou_threshold < 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1), got {self.iou_threshold}")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a CSA-YOLO run."""

    seed: int
    paths: PathsConfig
    detector: DetectorConfig
    logging_level: str = "INFO"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(config_path: Path | str, project_root: Path | str | None = None) -> PipelineConfig:
    """Load and validate a pipeline configuration file.

    Args:
        config_path: Path to a YAML config (e.g. ``configs/default.yaml``).
        project_root: Root used to resolve relative paths in the config.
            Defaults to the config file's parent's parent (the repo root).

    Returns:
        A validated :class:`PipelineConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required section is missing.
        ValueError: If a value fails validation.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    root = Path(project_root) if project_root is not None else config_path.resolve().parent.parent

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    for section in ("seed", "paths", "detector"):
        if section not in raw:
            raise KeyError(f"Missing required config section: '{section}' in {config_path}")

    paths = PathsConfig(
        weights_dir=root / raw["paths"]["weights_dir"],
        results_dir=root / raw["paths"]["results_dir"],
        datasets_dir=root / raw["paths"]["datasets_dir"],
    )
    detector = DetectorConfig(**raw["detector"])
    logging_level = str(raw.get("logging", {}).get("level", "INFO"))

    return PipelineConfig(
        seed=int(raw["seed"]),
        paths=paths,
        detector=detector,
        logging_level=logging_level,
        raw=raw,
    )
