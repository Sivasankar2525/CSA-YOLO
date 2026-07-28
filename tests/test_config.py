"""Unit tests for typed configuration loading."""

from pathlib import Path

import pytest

from csa_yolo.utils.config import DetectorConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_default_config_loads() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml", project_root=PROJECT_ROOT)
    assert config.seed == 42
    assert config.detector.model.endswith(".pt")
    assert config.paths.weights_dir == PROJECT_ROOT / "weights"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "configs" / "does_not_exist.yaml")


def test_missing_section_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n")
    with pytest.raises(KeyError):
        load_config(bad, project_root=tmp_path)


@pytest.mark.parametrize("conf", [0.0, 1.0, -0.5, 1.5])
def test_invalid_confidence_rejected(conf: float) -> None:
    with pytest.raises(ValueError):
        DetectorConfig(model="x.pt", confidence_threshold=conf)
