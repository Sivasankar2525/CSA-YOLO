"""Tests for the ICSD-YOLO restoration adapter (training/icsd.py, ADR 0002).

The heavyweight instantiation test is marked slow; the rest verify the patch
anchors still match the pinned submodule commit so drift fails in CI, not
mid-experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training import icsd  # noqa: E402


def test_compat_dir_layout() -> None:
    assert (icsd.COMPAT_DIR / "einsum.py").is_file()
    assert (icsd.COMPAT_DIR / "Add_modules" / "__init__.py").is_file()
    assert (icsd.COMPAT_DIR / "Add_modules" / "BatchNorm.py").is_file()
    assert (icsd.COMPAT_DIR / "Add_modules" / "Conv.py").is_file()


def test_parse_model_anchors_match_pinned_commit() -> None:
    """Every textual anchor used by _patch_parse_model must exist verbatim."""
    tasks_src = (icsd.ICSD_ROOT / "models" / "nn" / "tasks.py").read_text()
    anchors = [
        "            SparseFocusBlock,\n",
        "            C2PSA,\n        }\n    )\n",
        "            c1, c2 = ch[f], args[1]\n",
        "                args.insert(2, n)  # number of repeats\n                n = 2\n",
        "        n = n_ = max(round(n * depth), 2) if n > 1 else n",
        "                    args[4] = True\n",
        "            args = [c1, c2, *args[1:]]\n",
    ]
    for anchor in anchors:
        assert anchor in tasks_src, f"anchor drifted: {anchor!r}"


def test_get_dataset_anchors_match_pinned_commit() -> None:
    trainer_src = (icsd.ICSD_ROOT / "models" / "engine" / "trainer.py").read_text()
    assert "        args = parse_args()\n" in trainer_src
    assert "data = args.data_path + '/' + args.data" in trainer_src


def test_variant_cfg_rejects_bad_scale() -> None:
    with pytest.raises(ValueError):
        icsd._variant_cfg("x", "q")


def test_variant_cfg_filename_encodes_scale() -> None:
    cfg = icsd._variant_cfg("x", "n")
    assert cfg.name == "icsd-xarch-yolo11n.yaml"
    assert cfg.read_text() == (icsd.ICSD_MODEL_CFG_DIR / "ICSD-YOLOx.yaml").read_text()


@pytest.mark.slow
def test_restored_icsd_instantiates_and_forwards() -> None:
    torch = pytest.importorskip("torch")
    model = icsd.load_icsd_yolo(scale="n", arch="x")
    n_params = sum(p.numel() for p in model.model.parameters())
    # Restored x-arch @ n-scale measured at 5.67M params (ADR 0002); a wide
    # band still catches scale collapse (2.6M) or mlp_ratio explosion (>20M).
    assert 4e6 < n_params < 8e6
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))
    feats = out[0] if isinstance(out, tuple) else out
    assert [tuple(t.shape) for t in feats] == [
        (1, 144, 80, 80),
        (1, 144, 40, 40),
        (1, 144, 20, 20),
    ]
