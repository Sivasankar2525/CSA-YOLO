# Evaluation infrastructure

Status legend used throughout: **IMPLEMENTED** (code exists), **TESTED** (automated tests pass),
**NOT MEASURED** (no executed experiment), **EXPERIMENT REQUIRED** (needs training, annotation or
video ground truth). Nothing here is EXPERIMENTALLY VALIDATED yet.

## Pipeline commands

| Step | Command | Status |
|---|---|---|
| Train + validate + test (one experiment) | `python training/train_detector.py --experiment configs/experiments/<name>.yaml --device cuda` | IMPLEMENTED, TESTED (no training run yet) — **EXPERIMENT REQUIRED** |
| Smoke test (one epoch) | add `--epochs 1 --no-test-eval` | IMPLEMENTED — EXPERIMENT REQUIRED |
| Re-evaluate saved weights | `... --eval-only results/experiments/<name>/weights/best.pt` | IMPLEMENTED — EXPERIMENT REQUIRED |
| Dataset validation (machine-readable) | `python scripts/validate_dataset.py --root datasets/csa_ppe_v1_clean --expect train=1481 valid=210 test=195 --out results/dataset_validation_clean.json` | IMPLEMENTED, TESTED; run on both datasets (`results/dataset_validation_*.json`) |
| Full video/image pipeline | `python scripts/run_pipeline.py --source <file-or-folder> --detector <weights> --out <dir>` | IMPLEMENTED, TESTED (scripted detector); smoke-tested on real images with COCO weights |
| Ablation (six configurations) | `python evaluation/run_ablation.py --detector <weights> --images datasets/csa_ppe_v1_clean/images/test` | IMPLEMENTED, TESTED — EXPERIMENT REQUIRED |
| Training curves | `python evaluation/plot_training.py --run results/experiments/<name>` | IMPLEMENTED, TESTED |
| Metrics summary (single source of truth) | `python evaluation/build_metrics_summary.py` | IMPLEMENTED, TESTED |
| Results tables (CSV) | `python evaluation/collect_results.py` | IMPLEMENTED, TESTED |
| Annotation well-formedness | `python evaluation/validate_annotations.py annotations/<module>.json` | IMPLEMENTED, TESTED (format only — not evidence) |

## Where every number comes from

- `results/experiments/<name>/test_metrics.json` — detector, **test split**, fixed settings
  (`conf=0.001`, `iou=0.7`, same `imgsz` as training). This is the reported detector result.
- `results/experiments/<name>/val_metrics.json` — detector, validation split, same settings.
- `results/experiments/<name>/manifest.json` — git commit, Python/PyTorch/CUDA/Ultralytics
  versions, GPU name and memory, seed, optimizer and learning rate actually used, initialization,
  wall-clock timings, and output paths.
- `results/experiments/<name>/experiment_summary.json` — training summary and effective config.
- `results/experiments/<name>/eval_test/`, `eval_val/` — Ultralytics confusion matrix and per-class plots.
- `results/eval/{wpam,tracking,tsmm,scrm}.json` — module metrics, written by the module evaluations
  once annotations exist. **Not yet produced.**
- `results/ablation/summary.json` — operational counts only.

Every table cell is either a value from one of these files or `NOT MEASURED`.

## Definitions (arithmetic is tested on synthetic inputs only)

- **Detector (Ultralytics).** Precision, recall, mAP@0.5, mAP@0.5:0.95 are overall values from
  `model.val`. Ultralytics does not provide an overall F1 scalar, so `F1` is recorded per class
  only; the overall F1 field is `null` and listed under `unavailable`.
- **Association (WPAM).** A link is `(image, worker, ppe_class, ppe_id)`. Precision = predicted links
  in truth / predicted links. Recall = truth links predicted / truth links. Undefined when the
  denominator is zero (reported as NOT MEASURED).
- **Compliance (SCRM).** Positive class NON_COMPLIANT. Accuracy, precision, recall, F1 use only
  workers the system decided (COMPLIANT or NON_COMPLIANT). UNKNOWN is **coverage**, not an error.
- **Tracking.** ID switch = change of predicted track for the same ground-truth worker between two
  consecutive frames in which the worker is visible.
- **Temporal (TSMM).** Status transitions per sequence. Reduction = (raw − smoothed) / raw on the
  same sequence. Undefined when raw has no transitions.

## Annotation schemas (human ground truth; nothing generates it)

`annotations/wpam.json`
```json
{"annotator": "ROLE", "annotation_date": "YYYY-MM-DD",
 "images": {"<image_id>": {"workers": {"<worker_id>": {"helmet": "<ppe_id or null>", "vest": "<ppe_id or null>"}}}}}
```

`annotations/scrm.json` — labels are human decisions only: `COMPLIANT` or `NON_COMPLIANT`. `UNKNOWN`
is a system output and is rejected in annotation files.
```json
{"annotator": "ROLE", "annotation_date": "YYYY-MM-DD",
 "images": {"<image_id>": {"<worker_id>": "COMPLIANT"}}}
```

`annotations/tracking.json` — one object per frame, in time order, mapping ground-truth worker id
to the track id a tracker should report.
```json
{"annotator": "ROLE", "annotation_date": "YYYY-MM-DD",
 "frames": [{"<gt_worker_id>": "<track_id>"}]}
```

`annotations/tsmm.json` — per clip, the per-frame human status of each worker, and the frame
intervals during which a worker is occluded.
```json
{"annotator": "ROLE", "annotation_date": "YYYY-MM-DD",
 "clips": {"<clip_id>": {"frame_status": {"<worker_id>": ["COMPLIANT", "NON_COMPLIANT"]},
                         "occlusion_intervals": [[start_frame, end_frame]]}}}
```

Validate any file with `python evaluation/validate_annotations.py <file>`. A passing file is only
well-formed. It is not a metric.

## Current status of each module's evaluation

| Module | Metric code | Ground truth | Result |
|---|---|---|---|
| Detector | Ultralytics `model.val` | dataset labels (exist) | **EXPERIMENT REQUIRED** (no trained detector) |
| WPAM | `association_prf` | annotations/wpam.json — **not created** | NOT MEASURED |
| Tracking | `id_switches` | annotations/tracking.json — **not created** | NOT MEASURED |
| TSMM | `flicker_reduction` | annotations/tsmm.json — **not created** | NOT MEASURED |
| SCRM | `compliance_metrics` | annotations/scrm.json — **not created** | NOT MEASURED |
| CRAM | qualitative scenarios | none | RULE-BASED PROTOTYPE; case study not run |
| Ablation | runner, counts | none needed for counts | counts only; **EXPERIMENT REQUIRED** |

## What still has to be built or done (in order)

1. Train the three detectors on Colab (notebook `notebooks/CSA_YOLO_Colab.ipynb`). **EXPERIMENT REQUIRED.**
2. Human annotation: WPAM links, SCRM labels, tracking identities, TSMM sequences and occlusions.
   Schemas above. Target counts are not fixed (the plan gives two different SCRM ranges).
3. The module evaluation script that joins annotations with pipeline output on the same images or
   frames. Not written yet; it must not be run until step 2 exists.
4. Error analysis and paper figures from the artifacts above.

## Rules for reporting

1. A number is reported only when the file that produced it exists under `results/`.
2. A metric whose ground truth is missing is NOT MEASURED. It is never 0 and never an estimate.
3. Ablation counts are operational, not accuracy.
4. CRAM is reported qualitatively, as a rule-based prototype.
5. Each detector result states its split (test), its settings and its manifest.
