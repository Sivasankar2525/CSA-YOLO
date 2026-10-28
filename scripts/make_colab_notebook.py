#!/usr/bin/env python3
"""Generate notebooks/CSA_YOLO_Colab.ipynb, the single reproducible Colab workflow.

The notebook is generated from this file so the workflow is versioned and reviewable.
Every code cell is plain Python that calls subprocess (no notebook magics), so each cell
can be checked with ``ast.parse`` and run without hidden state.

Usage:
    python scripts/make_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "notebooks" / "CSA_YOLO_Colab.ipynb"

MARKDOWN_INTRO = """# CSA-YOLO — Colab training and evaluation

Run the cells **top to bottom**, one experiment at a time.

1. Set `EXPERIMENT` in Cell 1 (one of the three listed there). Nothing else needs editing.
2. Cells 2–8 check the environment and the data. They do not train.
3. Cell 9 trains and evaluates the chosen experiment (2–4 h on a T4). Run it only after Cell 8 passes.
4. Cells 10–12 export curves, metrics and artifacts to Google Drive.

The clean dataset (`csa_ppe_v1_clean`) must already be on Google Drive. It is never committed to GitHub.
Every result is written by the scripts. Nothing in this notebook writes a number by hand."""

CELLS: list[tuple[str, str]] = [
    ("markdown", MARKDOWN_INTRO),
    ("code", '''# Cell 1 — choose ONE experiment and the run options
EXPERIMENT = "baseline_yolo11n"       # baseline_yolo11n | yolo11n_scratch | icsd_restored_scratch
DEVICE = "cuda"                        # Colab GPU
SMOKE_TEST = True                      # one epoch first; strongly recommended

REPO_URL = "https://github.com/Sivasankar2525/CSA-YOLO.git"
# Use "main" until the workflow is pushed; then replace with the pushed commit hash so the run is pinned.
REPO_COMMIT = "main"
REPO = "/content/csa-yolo"
DRIVE_DATASET = "/content/drive/MyDrive/CSA-YOLO/datasets/csa_ppe_v1_clean"
DRIVE_RESULTS = "/content/drive/MyDrive/CSA-YOLO/results"
EXPECTED_COUNTS = {"train": 1481, "valid": 210, "test": 195}

EXPERIMENTS = {"baseline_yolo11n", "yolo11n_scratch", "icsd_restored_scratch"}
assert EXPERIMENT in EXPERIMENTS, f"EXPERIMENT must be one of {sorted(EXPERIMENTS)}"
print("experiment:", EXPERIMENT, "| device:", DEVICE, "| smoke test first:", SMOKE_TEST)'''),
    ("code", '''# Cell 2 — GPU verification (must show a T4, V100 or A100)
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True)
print(result.stdout or result.stderr)
if result.returncode != 0:
    raise SystemExit("No GPU. Runtime > Change runtime type > T4 GPU, then restart and run again.")'''),
    ("code", '''# Cell 3 — clone the repository at the pinned commit, with the ICSD-YOLO submodule
import os, subprocess
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--recursive", REPO_URL, REPO], check=True)
subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin"], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "--quiet", REPO_COMMIT], check=True)
subprocess.run(["git", "-C", REPO, "submodule", "update", "--init", "--recursive"], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
icsd = subprocess.run(["git", "-C", f"{REPO}/external/icsd_yolo", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
print("ICSD-YOLO submodule:", icsd)
assert icsd == "47018ed", "submodule is not at the pinned commit"'''),
    ("code", '''# Cell 4 — install the pinned dependencies (pip install of torch can take several minutes)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO}/requirements.txt"], check=True)
print("dependencies installed")'''),
    ("code", '''# Cell 5 — verify the runtime: versions and CUDA
import sys, torch, ultralytics
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| CUDA available:", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
assert torch.cuda.is_available(), "CUDA not available. Restart the runtime after Cell 4 and check the GPU."
print("GPU:", torch.cuda.get_device_name(0))'''),
    ("code", '''# Cell 6 — mount Google Drive and copy the clean dataset into the clone (only if not already there)
from google.colab import drive
import os, shutil
drive.mount("/content/drive")
dst = f"{REPO}/datasets/csa_ppe_v1_clean"
assert os.path.isdir(DRIVE_DATASET), f"Upload csa_ppe_v1_clean to {DRIVE_DATASET} first (never to GitHub)."
if not os.path.isdir(dst):
    shutil.copytree(DRIVE_DATASET, dst)
print("dataset at", dst)'''),
    ("code", '''# Cell 7 — dataset verification and exact-image leakage check (read-only)
import json, subprocess, sys
subprocess.run([sys.executable, f"{REPO}/scripts/validate_dataset.py",
                "--root", f"{REPO}/datasets/csa_ppe_v1_clean",
                "--expect", "train=1481", "valid=210", "test=195",
                "--out", f"{REPO}/results/dataset_validation_colab.json"], check=True)
report = json.load(open(f"{REPO}/results/dataset_validation_colab.json"))
for split, expected in EXPECTED_COUNTS.items():
    got = report["splits"][split]["images"]
    assert got == expected, f"{split}: expected {expected} images, found {got}"
assert report["total_problems"] == 0, f"label problems found: {report['total_problems']}"
leak = report["overlap"]["exact_image_leakage"]
assert leak == 0, f"EXACT IMAGE LEAKAGE: {leak}. Do not train; re-upload the clean dataset."
print(f"EXACT IMAGE LEAKAGE: {leak}")
print("Near-duplicate leakage: NOT MEASURED")'''),
    ("code", '''# Cell 8 — smoke test: one epoch, the pipeline runs end to end (skipped if SMOKE_TEST is False)
import subprocess, sys, os
if SMOKE_TEST:
    subprocess.run([sys.executable, "training/train_detector.py",
                    "--experiment", f"configs/experiments/{EXPERIMENT}.yaml",
                    "--device", DEVICE, "--epochs", "1", "--no-test-eval"], cwd=REPO, check=True)
    # keep the smoke run out of the real results folder
    smoke_dir = f"{REPO}/results/experiments/{EXPERIMENT}"
    if os.path.isdir(smoke_dir):
        subprocess.run(["mv", smoke_dir, f"/content/smoke_{EXPERIMENT}"], check=True)
    print("smoke test passed; smoke output moved to /content/smoke_" + EXPERIMENT)
else:
    print("smoke test skipped")'''),
    ("code", '''# Cell 9 — TRAIN and evaluate (validation and test, same settings). Long run: 2-4 h on a T4.
import subprocess, sys
subprocess.run([sys.executable, "training/train_detector.py",
                "--experiment", f"configs/experiments/{EXPERIMENT}.yaml",
                "--device", DEVICE], cwd=REPO, check=True)
print("training and evaluation finished for", EXPERIMENT)'''),
    ("code", '''# Cell 10 — show the reported (test) and validation metrics exactly as written by the scripts
import json
run = f"{REPO}/results/experiments/{EXPERIMENT}"
for name in ("test_metrics.json", "val_metrics.json"):
    data = json.load(open(f"{run}/{name}"))
    print(name, {k: data[k] for k in ("precision", "recall", "mAP@0.5", "mAP@0.5:0.95")})
    print("  unavailable fields:", data.get("unavailable"))'''),
    ("code", '''# Cell 11 — training curves and the metrics summary (both read the files written in Cell 9)
import subprocess, sys
subprocess.run([sys.executable, "evaluation/plot_training.py", "--run", f"results/experiments/{EXPERIMENT}"],
               cwd=REPO, check=True)
subprocess.run([sys.executable, "evaluation/build_metrics_summary.py"], cwd=REPO, check=True)
print("figures and results/metrics_summary.json updated")'''),
    ("code", '''# Cell 12 — copy every artifact of this run to Drive (weights, manifest, summaries, figures)
import datetime, shutil
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
dst = f"{DRIVE_RESULTS}/{EXPERIMENT}_{stamp}"
shutil.copytree(f"{REPO}/results/experiments/{EXPERIMENT}", dst)
shutil.copy(f"{REPO}/results/metrics_summary.json", f"{DRIVE_RESULTS}/metrics_summary_{stamp}.json")
print("saved to", dst)'''),
]


def build_notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell: dict = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
