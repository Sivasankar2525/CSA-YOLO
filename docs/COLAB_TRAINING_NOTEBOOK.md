# CSA-YOLO Training on Google Colab

This document provides copy-paste Colab cells for training the detector.

## Prerequisites

- Google account (free Colab access)
- Colab notebook open (colab.research.google.com)

## Setup Cells

### Cell 1: Clone Repository

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/csa-yolo.git /content/csa-yolo
%cd /content/csa-yolo
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

### Cell 4: Verify Dataset

```python
import os
import yaml

# Check dataset exists
dataset_path = '/content/csa-yolo/datasets/csa_ppe_v1'
assert os.path.isdir(dataset_path), f"Dataset not found at {dataset_path}"

# Count images
for split in ['train', 'valid', 'test']:
    count = len(os.listdir(f'{dataset_path}/images/{split}'))
    print(f"{split:8} : {count:5} images")

# Read data.yaml
with open(f'{dataset_path}/data.yaml') as f:
    cfg = yaml.safe_load(f)
    print(f"\nClasses: {cfg['names']}")
```

**Expected output:**
```
train    :  1516 images
valid    :   213 images
test     :   195 images

Classes: ['person', 'helmet', 'no_helmet', 'vest', 'no_vest']
```

### Cell 5: Fix Dataset Paths (If Needed)

```python
# If dataset.yaml has absolute Mac paths, fix them
import yaml

data_yaml = '/content/csa-yolo/datasets/csa_ppe_v1/data.yaml'
with open(data_yaml) as f:
    cfg = yaml.safe_load(f)

# Update paths
cfg['path'] = '/content/csa-yolo/datasets/csa_ppe_v1'

with open(data_yaml, 'w') as f:
    yaml.dump(cfg, f)

print("Updated data.yaml:")
print(yaml.dump(cfg))
```

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
    data='/content/csa-yolo/datasets/csa_ppe_v1/data.yaml',
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

### Cell 9: Push to GitHub (Optional)

```python
!git config --global user.email "YOUR_EMAIL@example.com"
!git config --global user.name "Your Name"

# Add trained weights
!git add weights/ results/experiments/
!git commit -m "feat: Add trained detector weights (baseline_yolo11n on Colab V100)"
!git push
```

**Note:** Requires GitHub credentials. Use personal access token if requested.

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
!ls -la /content/csa-yolo/datasets/csa_ppe_v1/images/train | head -5
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
