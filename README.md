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

| Module | Purpose | Hypothesis to test (NOT YET MEASURED) |
|---|---|---|
| **WPAM** | Spatially gated, one-to-one assignment of PPE items to workers | Association precision/recall vs. a greedy or IoU baseline |
| **TSMM** | Per-worker temporal evidence window over tracked identities | Fewer false violations than per-frame decisions |
| **SCRM** | Evidence-based `COMPLIANT` / `NON_COMPLIANT` / `UNKNOWN` decisions | Compliance accuracy against human labels |
| **CRAM** | Rule-based risk prototype with transparent factors | Qualitative case studies only; **not a validated risk model** |

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

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run Tests

```bash
pytest tests/ -q
```

### Demo: Single Image

```bash
python scripts/demo.py <image_path> [--output <output_path>] [--json-output <json_path>]
```

Example:
```bash
python scripts/demo.py datasets/csa_ppe_v1_clean/images/test/CHV_ppe_0003.jpg --output output.jpg
```

### Dashboard: Interactive Web UI

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` — upload images/videos for analysis.

### Detector Verification

```bash
python scripts/verify_detection.py --config configs/default.yaml
```

## Module Documentation

Full specifications, decision rules and parameters: **[docs/MODULES.md](docs/MODULES.md)**.

All four research modules are **implemented and unit tested. None is experimentally evaluated yet.**

- **WPAM** — spatial gate around the head and torso regions, then one-to-one Hungarian assignment. `csa_yolo/wpam/associate.py`
- **Tracking** — IoU matching with removal of tracks unmatched for `max_age` frames. `csa_yolo/tracking/tracker.py`
- **TSMM** — per-worker sliding window of PPE observations, with stale-state cleanup. `csa_yolo/tsmm/memory.py`
- **SCRM** — `COMPLIANT` / `NON_COMPLIANT` / `UNKNOWN` from evidence counts and thresholds. `csa_yolo/scrm/compliance.py`
- **CRAM** — a rule-based **prototype** risk layer. It is **not a validated predictive model** and is not OSHA-certified. `csa_yolo/cram/risk_assessment.py`

All parameters are in `configs/default.yaml`. The WPAM gate margin is marked PROVISIONAL and has not been tuned.

## Datasets, Training and Evaluation

- **Research dataset:** `datasets/csa_ppe_v1_clean/` (1481 / 210 / 195 images; 0 exact-image leakage).
  The original `csa_ppe_v1` is kept as a reference copy and must not be used for evaluation.
  Details: [docs/DATASETS.md](docs/DATASETS.md). The archive `csa_ppe_v1.zip` is never committed.
- **Training** (on Colab GPU; see [docs/COLAB_TRAINING_NOTEBOOK.md](docs/COLAB_TRAINING_NOTEBOOK.md)):
  ```bash
  python training/train_detector.py --experiment configs/experiments/baseline_yolo11n.yaml
  python training/train_detector.py --experiment configs/experiments/baseline_yolo11n.yaml --device cuda --epochs 1  # smoke test
  ```
  The launcher resolves the dataset path itself and records its effective settings in
  `experiment_summary.json`.
- **Ablation:** `python evaluation/run_ablation.py --detector <weights> --images datasets/csa_ppe_v1_clean/images/test`
  (six cumulative configurations in `configs/ablation.yaml`).
- **Metrics:** `python evaluation/build_metrics_summary.py` writes `results/metrics_summary.json` from
  artifacts on disk. A missing artifact is reported as `NOT MEASURED`. See [docs/EVALUATION.md](docs/EVALUATION.md).

**No experimental results are reported yet.** No detector has been trained and no ground truth exists.

## Pipeline Orchestration

### Using the SafetyPipeline class

```python
from csa_yolo.detection import Detector
from csa_yolo.pipeline import SafetyPipeline
import cv2

detector = Detector(model_path="weights/yolo11n.pt", device="mps")
pipeline = SafetyPipeline(detector)

frame = cv2.imread("image.jpg")
result = pipeline.process_frame(frame)

# Access results:
print(result.detections)           # Raw detections
print(result.associations)          # Worker-PPE associations
print(result.tracked_workers)       # {track_id: Detection}
print(result.temporal_states)       # {track_id: WorkerSafetyState}
print(result.compliance_results)    # {track_id: ComplianceResult}
print(result.worker_risks)          # {track_id: RiskResult}
```

### JSON Output

All results are JSON-serializable:
```python
result_dict = result.to_dict()
import json
print(json.dumps(result_dict, indent=2))
```

## System Requirements

- **Python:** 3.10+
- **PyTorch:** 2.13.0+ (with CPU, CUDA, or Apple MPS support)
- **Memory:** 8 GB minimum (for MacBook Air M2; larger GPUs: 16 GB+)
- **GPU:** Optional (NVIDIA CUDA, Apple MPS, or CPU inference supported)

### Installation on macOS (Apple Silicon)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Apple MPS is auto-detected and enabled if available.

## Known Limitations

1. **No experimental evaluation yet.** No detector has been trained and no accuracy, mAP, FPS, tracking or compliance metric has been measured. See `docs/PROJECT_STATUS.md`.
2. **Detector:** the demo uses the COCO-pretrained YOLO11n. The fine-tuned detector is not trained yet.
3. **Single image demo:** video processing requires a frame loop; no real-time video path is measured.
4. **Risk model is a prototype:** rule-based weights were chosen by hand. It is not validated.
5. **Occlusion:** a PPE item hidden by occlusion looks like a missing one. See `docs/MODULES.md`.
6. **No zone awareness:** SCRM applies global rules. Zone-specific rules are future work.
7. **Limited PPE classes:** helmet and vest only.
8. **Leakage:** use `datasets/csa_ppe_v1_clean` for evaluation. The original `csa_ppe_v1` contains duplicate images. See `docs/DATASETS.md`.

## Reproducibility

- Global seed fixed via `csa_yolo.utils.reproducibility.set_seed` (default `42`, set in config).
- Exact dependency pins in `requirements.txt` / `environment.yml`.
- ICSD-YOLO pinned at commit `47018ed` (Ultralytics 8.3.74 fork).
- Every experiment writes its resolved config + seed next to its results under `experiments/`.

## Citation

If you use this framework in research, please cite:

```bibtex
@misc{csa-yolo-2026,
  title={CSA-YOLO: Construction Safety-Aware Detector with Worker-PPE Association,
         Temporal Memory, and Risk Assessment},
  author={Siva Sankar B},
  year={2026},
  howpublished={\url{https://github.com/Sivasankar2525/CSA-YOLO}}
}
```

## License and Third-Party Components

**CSA-YOLO license: NOT YET DECIDED.** The repository does not yet include a LICENSE file.
Until one is added, no reuse rights are granted beyond viewing the code.

Third-party components and their terms:

- **ICSD-YOLO** (`external/icsd_yolo`, PrintSC/code, pinned at `47018ed`): the submodule
  contains **no LICENSE file**. Its model configuration header reads
  `Ultralytics AGPL-3.0 License`, which indicates that it derives from Ultralytics code.
  Its licence status is **unresolved**.
- **Ultralytics** (runtime dependency, see `requirements.txt`): AGPL-3.0.
- **Pretrained weights** `weights/yolo11n.pt`: Ultralytics COCO-pretrained weights, under the Ultralytics terms.
- **Datasets** (CHV, SHWD): see `datasets/LICENSES.md`. Images are not redistributed.

Because the dependencies are AGPL-3.0, the licence choice for CSA-YOLO must be made with
these constraints in mind. This is a decision for the author.
