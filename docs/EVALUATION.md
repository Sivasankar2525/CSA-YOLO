# Evaluation infrastructure

Status: **infrastructure implemented and unit tested. No metric has been measured.**

## Single source of truth

`results/metrics_summary.json` is written only by `evaluation/build_metrics_summary.py`. Every number
in it is copied from an artifact on disk. A missing artifact gives `NOT MEASURED` and names the path
that was searched. Nothing is typed in by hand or copied from documentation.

```bash
python evaluation/build_metrics_summary.py
```

The current file reports every detector run and every module as NOT MEASURED, because no training
run, ablation run or annotation exists yet.

## What exists, by area

| Area | Implemented | Source of numbers | Status |
|---|---|---|---|
| Detector metrics (P, R, mAP50, mAP50-95, per-class) | summary read by builder | `results/experiments/<run>/experiment_summary.json` (written by `training/train_detector.py`) | NOT MEASURED |
| Training curves | path recorded | `results/experiments/<run>/results.csv` (Ultralytics) | NOT MEASURED |
| Confusion matrix | path recorded | `results/experiments/<run>/confusion_matrix.png` (Ultralytics) | NOT MEASURED |
| Model comparison (baseline vs. scratch vs. ICSD) | builder lists all three runs | the three summaries above | NOT MEASURED |
| Ablation (6 cumulative configurations) | runner writes counts | `results/ablation/summary.json` | counts only; no ground-truth metric |
| WPAM association precision / recall / F1 | `evaluation/metrics.py::association_prf` | human annotation, `annotations/wpam.json` | NOT MEASURED (no annotation) |
| Compliance accuracy / precision / recall / F1 / coverage | `compliance_metrics` | human annotation, `annotations/scrm.json` | NOT MEASURED (no annotation) |
| Tracking ID switches | `id_switches` | human annotation, `annotations/tracking.json` | NOT MEASURED (no annotation) |
| TSMM flicker reduction | `flicker_reduction` | human annotation, `annotations/tsmm.json` | NOT MEASURED (no annotation) |
| CRAM | qualitative case studies only | none | not a metric; prototype |

The metric functions are arithmetic definitions. Their tests use synthetic inputs with hand-computed
answers (`tests/test_eval_metrics.py`). Those tests verify the arithmetic. They are not experimental
results.

## Definitions

- **Association (WPAM).** A link is `(image, worker, ppe_class, ppe_id)`. Precision is the share of
  predicted links that are in the truth. Recall is the share of truth links that were predicted.
  Undefined values (zero denominator) are `None`, reported as NOT MEASURED.
- **Compliance (SCRM).** Positive class is NON_COMPLIANT. Accuracy, precision, recall and F1 use only
  workers the system decided (COMPLIANT or NON_COMPLIANT). UNKNOWN is reported as **coverage**, not
  as an error. Coverage = decided / evaluated.
- **ID switches (tracking).** A switch is a change in the predicted track ID for the same ground-truth
  worker between two consecutive frames in which that worker is visible. Gaps do not reset the comparison.
- **Flicker (TSMM).** Number of status changes in a sequence. Reduction = (raw − smoothed) / raw, on the
  same sequence. It is undefined if the raw sequence has no changes. A value is evidence for that
  sequence only, not a claim about the population.

## Ground-truth schema (required before any module metric)

Ground truth must be produced by a human annotator. Nothing in this repository generates it.

`annotations/wpam.json`
```json
{
  "annotator": "NAME_OR_ROLE",
  "annotation_date": "YYYY-MM-DD",
  "images": {
    "<image_id>": {
      "workers": {
        "<worker_id>": {"helmet": "<ppe_id or null>", "vest": "<ppe_id or null>"}
      }
    }
  }
}
```

`annotations/scrm.json`: `{"images": {"<image_id>": {"<worker_id>": "COMPLIANT | NON_COMPLIANT"}}}`

`annotations/tracking.json`: `{"frames": [{"<gt_worker_id>": "<track_id>"}, ...]}` in time order per clip.

`annotations/tsmm.json`: per-clip sequences of per-frame worker status, plus occlusion intervals.

Targets and their status are recorded in `docs/PROJECT_STATUS.md`. The SCRM target count is not
fixed: the plan gives 50–100 in one section and 100–200 in another. A human decision is needed.

## Rules for reporting

1. A number is reported only if the artifact that produced it exists in `results/`.
2. A metric whose ground truth is missing is NOT MEASURED. It is never 0 and never an estimate.
3. Ablation counts are operational. They are not accuracy.
4. CRAM is reported qualitatively, as a rule-based prototype.
5. Training runs must record their effective settings (`effective_train` in the summary). Overrides
   such as `--device cuda` or `--epochs 1` are visible there.
