"""The generated Colab notebook: valid JSON, parseable code cells, safety gates present."""

import ast
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "CSA_YOLO_Colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    assert NOTEBOOK.is_file(), "run scripts/make_colab_notebook.py"
    return json.loads(NOTEBOOK.read_text())


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat(nb: dict) -> None:
    assert nb["nbformat"] == 4 and nb["cells"]


def test_every_code_cell_is_plain_python(nb: dict) -> None:
    for i, src in enumerate(_code(nb), start=1):
        ast.parse(src, filename=f"cell{i}")  # raises on magics or syntax errors


def test_only_experiment_name_is_chosen_in_cell_one(nb: dict) -> None:
    first = _code(nb)[0]
    assert 'EXPERIMENT = "baseline_yolo11n"' in first
    assert "yolo11n_scratch" in first and "icsd_restored_scratch" in first


def test_no_local_mac_paths(nb: dict) -> None:
    text = NOTEBOOK.read_text()
    assert "/Users/" not in text


def test_leakage_gate_precedes_training(nb: dict) -> None:
    codes = _code(nb)
    leak = next(i for i, c in enumerate(codes) if "EXACT IMAGE LEAKAGE" in c)
    train = next(i for i, c in enumerate(codes) if "train_detector.py" in c and "--epochs" not in c
                 and "--no-test-eval" not in c)
    assert leak < train


def test_training_cell_is_the_only_long_run(nb: dict) -> None:
    codes = _code(nb)
    long_runs = [i for i, c in enumerate(codes) if "train_detector.py" in c and "--epochs" not in c]
    assert len(long_runs) == 1


def test_commit_variable_and_submodule_are_checked(nb: dict) -> None:
    text = "\n".join(_code(nb))
    assert "REPO_COMMIT =" in text  # set to the pushed commit hash before a pinned run
    assert "47018ed" in text
