# CSA-YOLO

**Construction Safety Intelligence Framework with Worker–PPE Association, Temporal Safety
Memory, Compliance Reasoning, and Explainable Risk Assessment.**

CSA-YOLO extends the ICSD-YOLO detector ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425046093),
[official code](https://github.com/PrintSC/code)) from *object-level detection* ("is there a
helmet?") to *worker-level safety intelligence* ("is Worker #17 compliant, for how long has the
violation existed, and why is their risk HIGH?").

## Pipeline

```
Video → Frame Extraction → ICSD-YOLO (worker + PPE detection)
      → WPAM (worker–PPE association)
      → TSMM (temporal safety memory)
      → SCRM (compliance reasoning)
      → CRAM (explainable risk assessment)
      → Supervisor Dashboard
```

## Research contributions

| Module | Purpose | Key claim |
|---|---|---|
| **WPAM** | Assign every PPE detection to exactly one worker | Association accuracy vs. greedy/IoU baselines |
| **TSMM** | Per-worker temporal memory over tracked identities | False-alarm reduction under occlusion |
| **SCRM** | Zone-aware, configurable worker-level compliance reasoning | Compliance decision accuracy |
| **CRAM** | Explainable risk scoring grounded in OSHA severity guidance | Risk-ranking correctness + human-readable rationale |

## Repository layout

```
csa_yolo/            Python package (importable framework code)
  detection/         Typed wrapper around the YOLO/ICSD-YOLO detector
  wpam/  tsmm/  scrm/  cram/   The four research modules (built in this order)
  tracking/          Worker identity tracking (ByteTrack/IoU)
  utils/             Reproducibility, logging, configuration
configs/             All experiment/pipeline configuration (YAML) — no hardcoded paths
datasets/            Datasets (git-ignored; see docs/dataset_plan.md)
weights/             Model checkpoints (git-ignored)
external/            Pinned third-party code (official ICSD-YOLO, as submodule)
scripts/             Entry-point scripts (verification, demos)
training/            Detector fine-tuning configs and launchers
evaluation/          Metrics, ablation harness
dashboard/           Supervisor dashboard app
experiments/         Versioned experiment outputs (config + seed + results)
docs/                ADRs, related work, dataset plan, weekly reports
tests/               Unit tests (pytest)
paper/  ppt/         Manuscript and presentation material
```

## Quickstart

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/verify_detection.py --config configs/default.yaml
pytest tests/ -q
```

## Reproducibility

- Global seed fixed via `csa_yolo.utils.reproducibility.set_seed` (default `42`, set in config).
- Exact dependency pins in `requirements.txt` / `environment.yml`.
- ICSD-YOLO pinned at commit `47018ed` (Ultralytics 8.3.74 fork).
- Every experiment writes its resolved config + seed next to its results under `experiments/`.
