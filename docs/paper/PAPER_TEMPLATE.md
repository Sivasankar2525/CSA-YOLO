# [Title — to be finalised after results exist]

> **Status: TEMPLATE. No experimental result is reported here.**
> Every `[MEASURED RESULT REQUIRED]` must be replaced with a value from `results/` (see
> `docs/EVALUATION.md`). Every claim must cite either a measured result in this repository or a
> published source. Do not add statistics, comparisons or significance tests that were not computed.

## Abstract
[MEASURED RESULT REQUIRED: summary written last, from the measured tables only]

## 1. Introduction
- Problem: construction-site PPE compliance is a worker-level question, not only an object-level one.
- Gap: detection alone does not say which PPE belongs to which worker, or whether a violation persists.
- Contribution list: the modules as implemented (WPAM, tracking, TSMM, SCRM, CRAM), each labelled
  with its status (implemented / tested / experimentally validated).

## 2. Related Work
- Object detection for PPE (YOLO family; ICSD-YOLO) — citations to be added from `docs/related_work.md`.
- Worker-level association and tracking — to be written from the literature survey; no claim of novelty
  beyond what the survey supports.
- Rule-based risk assessment in industrial safety — the CRAM prototype is not validated against OSHA data.

## 3. Problem Definition
- Inputs (frames), outputs (per-worker PPE status, compliance state, risk level).
- Definitions of PRESENT, ABSENT and UNKNOWN exactly as in `docs/MODULES.md`.

## 4. Proposed Method
- Detector → WPAM (spatial gate + one-to-one Hungarian assignment) → tracking → TSMM → SCRM → CRAM.
- Parameters and their provenance (configs/default.yaml). The WPAM gate margin is PROVISIONAL.
- Learned components: the detector only. All other modules are rule-based.

## 5. Dataset
- Source datasets and licences (`datasets/LICENSES.md`).
- Label merge and class mapping (`configs/merge.yaml`).
- Original and clean counts from `results/dataset_validation_*.json`:
  [MEASURED RESULT REQUIRED: per-split images, boxes, classes]
- Exact-image leakage: 0 after cleanup (SHA-256). Near-duplicates: [MEASURED RESULT REQUIRED / NOT MEASURED].

## 6. Experimental Setup
- Hardware: Google Colab T4 (from `manifest.json`): [MEASURED RESULT REQUIRED]
- Software versions: from `manifest.json` (PyTorch, Ultralytics, Python).
- Training: seed, epochs, batch, image size, optimizer and learning rate from `manifest.json`.
- Evaluation settings: identical for all detectors (`conf=0.001`, `iou=0.7`, test split).

## 7. Results
### 7.1 Detector comparison (test split)
[MEASURED RESULT REQUIRED: from results/detector_comparison.csv]
### 7.2 Per-class AP
[MEASURED RESULT REQUIRED: from results/per_class_metrics.csv]
### 7.3 Module evaluation (WPAM, tracking, TSMM, SCRM)
[EXPERIMENT REQUIRED: annotations not yet created — see docs/EVALUATION.md]
### 7.4 CRAM case studies (qualitative)
[EXPERIMENT REQUIRED: case studies not yet run]

## 8. Ablation Study
[EXPERIMENT REQUIRED: six configurations on a trained detector; results/ablation_metrics.csv]

## 9. Error Analysis
[EXPERIMENT REQUIRED: categories defined in docs/ERROR_ANALYSIS.md; counts from measured runs only]

## 10. Discussion
[MEASURED RESULT REQUIRED: interpret only measured results]

## 11. Limitations
- Detector results depend on the Colab training budget.
- The ICSD-YOLO release is restored, not reproduced (ADR 0002).
- Occlusion is indistinguishable from absence in the current PPE evidence model.
- The tracker's identity assumption (overlap between consecutive frames) fails under fast motion.
- CRAM is a rule-based prototype and is not a validated risk model.
- Near-duplicate leakage is NOT MEASURED.

## 12. Conclusion
[MEASURED RESULT REQUIRED]

## 13. References
[To be completed from verified sources only. No reference is added without a checked citation.]
