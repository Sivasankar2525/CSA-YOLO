"""Training-curve exporter: reads results.csv, fails clearly when it is absent (synthetic data only)."""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("plot_training", PROJECT_ROOT / "evaluation" / "plot_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pt = _load()
HEADER = "epoch,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\n"


def test_reads_epochs_and_metric_columns(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text(HEADER + "1,0.9,0.1,0.05,0.2,0.3\n2,0.8,0.2,0.1,0.3,0.4\n")
    data = pt.read_results_csv(tmp_path / "results.csv")
    assert data["epoch"] == [1.0, 2.0]
    assert data["metrics/mAP50(B)"] == [0.1, 0.2]


def test_plot_writes_figure_from_csv(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text(HEADER + "1,0.9,0.1,0.05,0.2,0.3\n2,0.8,0.2,0.1,0.3,0.4\n")
    out = pt.plot(tmp_path)
    assert out.is_file() and out.name == "training_curves.png" and out.stat().st_size > 1000


def test_missing_csv_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(pt.PlotError, match="results.csv not found"):
        pt.plot(tmp_path)


def test_empty_csv_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text(HEADER)
    with pytest.raises(pt.PlotError, match="no rows"):
        pt.plot(tmp_path)


def test_missing_optional_columns_do_not_crash(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n2,0.2\n")
    assert pt.plot(tmp_path).is_file()
