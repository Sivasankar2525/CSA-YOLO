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
python scripts/demo.py datasets/raw/CSS/valid/example.jpg --output output.jpg
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

### WPAM — Worker–PPE Association Module

Associates detected PPE items with individual workers using geometric proximity.

**Algorithm:**
- Defines worker head region (top 20% of bounding box)
- Defines worker torso region (middle 60% of box)
- Matches helmets to head region and vests to torso using IoU + center distance
- Produces: `{worker_id: {helmet: [detections], vest: [detections]}}`

**Code:** `csa_yolo/wpam/associate.py`

### Tracking — IoU-Based Worker Tracking

Maintains persistent worker IDs across frames.

**Algorithm:**
- Matches workers frame-to-frame by maximum IoU overlap
- Assigns new ID when match confidence drops below threshold (0.3)
- Tracks age to remove inactive workers after 30 frames

**Code:** `csa_yolo/tracking/tracker.py`

### TSMM — Temporal Safety Memory Module

Reduces false alarms by smoothing PPE status over time.

**Algorithm:**
- Maintains sliding window (default: 5 frames) of PPE presence per worker
- Uses majority voting to determine confirmed PPE status
- Tracks violation duration (frames without required PPE)
- Confirmed status computed as: `present_frames / window_size >= threshold (0.7)`

**Code:** `csa_yolo/tsmm/memory.py`

### SCRM — Safety Compliance Reasoning Module

Determines worker compliance based on confirmed PPE status.

**Algorithm:**
- Reads required PPE list from config (default: `[helmet, vest]`)
- Compares confirmed PPE status against requirements
- Produces: `COMPLIANT | NON_COMPLIANT | UNKNOWN` with confidence score

**Code:** `csa_yolo/scrm/compliance.py`

### CRAM — Construction Risk Assessment Module

**⚠️ DISCLAIMER:** This is a PROTOTYPE RULE-BASED MODEL for demonstration only.
It is NOT OSHA-certified and should NOT be used for real safety decisions.

**Risk Scoring Algorithm:**
```
risk_score = (helmet_missing * 25.0)
           + (vest_missing * 15.0)
           + (violation_duration / 10 * 10.0)
           + (low_confidence_penalty * 5.0)

risk_level = LOW (< 25) | MEDIUM (25-60) | HIGH (> 60)
```

Risk factors are configurable in code. See `csa_yolo/cram/risk_assessment.py`.

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

1. **No real-time video:** Demo is single-image. Video processing requires frame extraction.
2. **YOLO11n baseline:** Fine-tuned ICSD-YOLO weights not yet available; currently using COCO-pretrained YOLO11n.
3. **Risk model is prototype:** Rule-based, not ML-trained. Weights are not optimized.
4. **No zone awareness:** SCRM currently applies global compliance rules; zone-specific rules are future work.
5. **Limited PPE classes:** Currently supports helmet and vest; other PPE requires dataset labeling.

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
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/csa-yolo}}
}
```

## License

See `LICENSE` in the repository root. Dataset licenses documented in `datasets/LICENSES.md`.
