# CSA-YOLO Training on Google Colab

This document provides copy-paste Colab cells for training the detector.

## Prerequisites

- Google account (free Colab access)
- Colab notebook open (colab.research.google.com)
- Runtime → Change runtime type → **T4 GPU** (or V100/A100 if available)
- `csa_ppe_v1_clean/` uploaded to Google Drive (from your Mac; not on GitHub). See Cell 4.
- Nothing is trained automatically: each training cell below is run by you.

## Setup Cells

### Cell 1: Clone Repository

```python
!git clone --recursive https://github.com/Sivasankar2525/CSA-YOLO.git /content/csa-yolo
%cd /content/csa-yolo
# Make sure the ICSD-YOLO submodule is present (pinned at 47018ed)
!git submodule update --init --recursive
!git -C external/icsd_yolo rev-parse --short HEAD
!pwd
```

**Expected output:**
```
/content/csa-yolo
```

### Cell 2: Install Dependencies

```python
!pip install -r requirements.txt -q
!pip install -e . -q
```

**Time:** ~2 minutes

### Cell 3: Verify GPU

```python
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

**Expected output (V100):**
```
Tesla V100-SXM2-16GB, 16GB, ~16GB
```

**If T4:**
```
Tesla T4, 16GB, ~16GB
```

### Cell 4: Put the dataset on Colab (upload, then verify)

`datasets/` is git-ignored, so a fresh clone has **no data**. Upload the dataset to Google
Drive once (never to GitHub), then copy it into the clone:

- Local file to upload: `csa_ppe_v1.zip` (original, 517 MiB, kept locally, never committed), **or**
  the folder `datasets/csa_ppe_v1_clean/` (leakage-free, the research dataset).
- Use `csa_ppe_v1_clean/` for every experiment. Do not train on the original folder.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
src = '/content/drive/MyDrive/CSA-YOLO/datasets/csa_ppe_v1_clean'   # adjust to your Drive folder
dst = '/content/csa-yolo/datasets/csa_ppe_v1_clean'
assert os.path.isdir(src), f"Upload csa_ppe_v1_clean to Drive first: {src}"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.isdir(dst):
    shutil.copytree(src, dst)

for split in ['train', 'valid', 'test']:
    n = len(os.listdir(f'{dst}/images/{split}'))
    print(f"{split:8}: {n:5} images")
```

**Expected output (clean dataset):**
```
train   :  1481 images
valid   :   210 images
test    :   195 images
```

### Cell 5: Verify no exact-image leakage on the uploaded copy

```python
import importlib.util, json
spec = importlib.util.spec_from_file_location('bcd', '/content/csa-yolo/scripts/build_clean_dataset.py')
bcd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bcd)
from pathlib import Path
names = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest']
report = bcd.verify(Path('/content/csa-yolo/datasets/csa_ppe_v1_clean'), names)
print(json.dumps(report, indent=2))
assert report['exact_image_leakage'] == 0, 'leakage detected: do not train'
print(f"EXACT IMAGE LEAKAGE: {report['exact_image_leakage']}")
```

**Expected:** `EXACT IMAGE LEAKAGE: 0`. If not, stop and re-upload.

**No path edits are needed.** The training launcher writes a resolved copy of the dataset YAML
(`data_resolved.yaml`) whose `path` is the folder that actually holds the data, so the Mac path in
the committed YAML is ignored on Colab.

---

## Training Cells

### Cell 6A: Train Baseline (YOLO11n Pretrained)

**Estimated time:** 2–4 hours (V100)

```python
!python training/train_detector.py \
  --experiment configs/experiments/baseline_yolo11n.yaml \
  --device cuda
```

**Expected output (final lines):**
```
Epoch 100/100: loss=X.XX, val/loss=X.XX, ...
Experiment 'baseline_yolo11n' finished in XXX.Xs — summary at results/experiments/baseline_yolo11n/experiment_summary.json
```

**Metrics saved to:** `results/experiments/baseline_yolo11n/experiment_summary.json`

### Cell 6B: Train YOLO11n from Scratch

```python
!python training/train_detector.py \
  --experiment configs/experiments/yolo11n_scratch.yaml \
  --device cuda
```

**Time:** 2–4 hours

### Cell 6C: Train ICSD-YOLO (Optional, Slower)

```python
!python training/train_detector.py \
  --experiment configs/experiments/icsd_restored_scratch.yaml \
  --device cuda
```

**Time:** 3–5 hours (ICSD blocks are heavier)

---

## Validation Cells

### Cell 7: Validate Baseline

```python
from ultralytics import YOLO
import json

# Load best model
model = YOLO('/content/csa-yolo/results/experiments/baseline_yolo11n/weights/best.pt')

# Validate on test set
results = model.val(
    data='/content/csa-yolo/datasets/csa_ppe_v1_clean/data.yaml',
    device='cuda',
    verbose=True
)

# Print key metrics
print(f"\nmAP@0.5: {results.box.map50:.3f}")
print(f"mAP@0.5:0.95: {results.box.map:.3f}")
print(f"Precision: {results.box.p:.3f}")
print(f"Recall: {results.box.r:.3f}")
```

---

## Download & Backup Cells

### Cell 8: Save Results Locally

```python
from google.colab import files

# Download best weights
print("Downloading weights...")
files.download('/content/csa-yolo/results/experiments/baseline_yolo11n/weights/best.pt')

# Download metrics
print("Downloading metrics...")
files.download('/content/csa-yolo/results/experiments/baseline_yolo11n/experiment_summary.json')

print("\nFiles ready for download in Colab sidebar →")
```

```python
# Record checksums for the downloaded weights (store them in docs/PROJECT_STATUS.md)
!sha256sum /content/csa-yolo/results/experiments/baseline_yolo11n/weights/best.pt
```

### Cell 9: Versioning the weights (Optional)

`weights/` and `results/` are git-ignored (see `.gitignore`), so `git add` does not
commit trained weights. Keep weights outside git: download them (Cell 8), store them
with their checksum, and record the location in `docs/PROJECT_STATUS.md`.

To commit only the small artifacts (summaries, configs, logs), copy them out of the
ignored folders first:

```python
!git config --global user.name "Siva Sankar B"
!git config --global user.email "YOUR_EMAIL@example.com"
!mkdir -p /content/csa-yolo/docs/run_summaries/baseline_yolo11n
!cp /content/csa-yolo/results/experiments/baseline_yolo11n/experiment_summary.json /content/csa-yolo/docs/run_summaries/baseline_yolo11n/
!git add -f docs/run_summaries/baseline_yolo11n/experiment_summary.json
!git commit -m "docs: Add baseline_yolo11n training summary (Colab GPU)"
!git push
```

**Note:** Requires GitHub credentials. Use a personal access token if requested.

---

## Quick Test Cell (For Smoke Tests)

### Smoke Test: 1 Epoch

```python
# Quick test with --epochs 1
!python training/train_detector.py \
  --experiment configs/experiments/baseline_yolo11n.yaml \
  --device cuda \
  --epochs 1
```

**Time:** ~30 seconds  
**Purpose:** Verify setup works before committing to full training

---

## Troubleshooting

### Issue: OOM (Out of Memory)

```python
# Reduce batch size (in Colab cell, NOT in yaml)
!python training/train_detector.py \
  --experiment configs/experiments/baseline_yolo11n.yaml \
  --device cuda \
  --epochs 100 \
  --batch-size 8   # <-- Add this
```

### Issue: CUDA Out of Memory

```
# Free GPU memory
import torch
torch.cuda.empty_cache()
```

### Issue: Dataset not found

```python
!ls -la /content/csa-yolo/datasets/csa_ppe_v1_clean/images/train | head -5
```

Should show ~1500 JPG files.

---

## Recording Metrics

After training completes, extract metrics:

```python
import json

summary_path = '/content/csa-yolo/results/experiments/baseline_yolo11n/experiment_summary.json'
with open(summary_path) as f:
    summary = json.load(f)

print("Training Summary:")
print(json.dumps(summary, indent=2))

# Copy paste into evaluation/metrics_template.json
```

---

## Expected Timeline

| Task | Time |
|------|------|
| Setup + install | 5 min |
| Smoke test (1 epoch) | 30 sec |
| Full training (100 epochs) | 2–4 hrs |
| Validation | 5 min |
| Download | 1 min |
| **Total** | **~3–5 hrs** |

---

## After Training

1. Download weights from Colab
2. Copy to local: `weights/baseline_yolo11n_best.pt`
3. Run local ablation: `python evaluation/run_ablation.py ...`
4. Fill in `evaluation/metrics_template.json`

---

**Ready to start?** Copy-paste Cell 1 into a new Colab notebook and begin!
