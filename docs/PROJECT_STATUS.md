# CSA-YOLO — Project Status

**Last updated:** 2026-10-28
**Baseline commit before this milestone:** `f2c2a94` (pushed to `origin/main`)

> **CSA-YOLO is NOT experimentally validated.** The software is implemented and tested.
> No detector has been trained. No human annotation exists. No research metric has been measured.
> Anything not produced by an executed experiment is `NOT MEASURED`, `EXPERIMENT REQUIRED` or `TBD`.

Status vocabulary: **IMPLEMENTED** (code runs) · **TESTED** (automated tests pass) ·
**NOT MEASURED** · **EXPERIMENT REQUIRED** · **IMPLEMENTED — VALIDATION REQUIRED** ·
**RULE-BASED PROTOTYPE**.

---

## Phase status

| Phase | Area | Status | Evidence / what is missing |
|---|---|---|---|
| 0 | Baseline protection | DONE | clean tree at `f2c2a94`; submodule clean at `47018ed`; 243 tests before this milestone |
| 1 | Colab / training readiness | IMPLEMENTED, TESTED; EXPERIMENT REQUIRED | launcher writes manifest, val and test metrics, optimizer, seed, errors; `notebooks/CSA_YOLO_Colab.ipynb` (13 cells). Not run on Colab. |
| 2 | Dataset validation | IMPLEMENTED, TESTED, RUN | `scripts/validate_dataset.py`; clean: PASS (0 problems, 0 exact leakage); original: 12 cross-split groups recorded. Near-duplicates NOT MEASURED. |
| 3 | Detector training (3 experiments) | EXPERIMENT REQUIRED | configs, launcher and notebook ready. No run has happened. |
| 4 | Detector comparison | PARTIAL | `collect_results.py` writes `detector_comparison.csv` and `per_class_metrics.csv` (NOT MEASURED until runs exist). Markdown/HTML report and comparison plots: not yet written. Inference latency benchmark: not defined yet. |
| 5 | WPAM evaluation | PARTIAL | metric code and `wpam_metrics.csv` (NOT MEASURED); annotation schema and validator done; no annotations; the join between annotations and pipeline output is not written. |
| 6 | Tracking | PARTIAL | tracker audited and tested; `tracking_metrics.csv` (NOT MEASURED); no annotated video. |
| 7 | TSMM | PARTIAL | evidence window and cleanup tested; `flicker_reduction` metric; no annotated video, so the with/without-TSMM experiment is EXPERIMENT REQUIRED. |
| 8 | SCRM | PARTIAL | three-state semantics tested; `compliance_metrics` (accuracy, P, R, F1, confusion, coverage); no labels, so NOT MEASURED. |
| 9 | CRAM | RULE-BASED PROTOTYPE | scenario tests exist; `cram_results.csv` placeholder; case studies not run. Not validated. |
| 10 | Full pipeline | IMPLEMENTED, TESTED | `csa_yolo/runner.py` + `scripts/run_pipeline.py`; 18 runner tests (edge cases); smoke-tested on real images with COCO weights. |
| 11 | Dashboard | NOT UPDATED | existing Streamlit dashboard unchanged in this milestone. Worker history, alerts and timestamps are not yet in the dashboard. |
| 12 | Ablation | IMPLEMENTED, TESTED; EXPERIMENT REQUIRED | runner with real toggles; `ablation_metrics.csv` (counts only). |
| 13 | Error analysis | TAXONOMY ONLY | `docs/ERROR_ANALYSIS.md` defines categories and rules. The generator and `error_analysis.csv` are not written. |
| 14 | Reproducibility | IMPLEMENTED, TESTED | `manifest.json` per run (git commit, versions, GPU, seed, optimizer, timings, paths). |
| 15 | Results automation | IMPLEMENTED, TESTED | `build_metrics_summary.py`, `collect_results.py`; NOT MEASURED where artifacts are absent. |
| 16 | Research figures | PARTIAL | `dataset_distribution.png` generated from the validation artifact (real). Training curves via `plot_training.py`. System architecture diagram, WPAM examples, tracking examples, TSMM behaviour, SCRM matrix, ablation plot, pipeline examples: NOT WRITTEN. |
| 17 | Documentation | PARTIAL | README, EVALUATION, DATASETS, MODULES, ERROR_ANALYSIS, COLAB, PROJECT_STATUS updated. A separate methodology document: not written. |
| 18 | Paper preparation | TEMPLATE ONLY | `docs/paper/PAPER_TEMPLATE.md` with `[MEASURED RESULT REQUIRED]` placeholders. No results. |
| 19 | Final deliverables | NOT STARTED | presentation, demo checklist, final tables and figures depend on Phases 3–8. |
| 20 | Git discipline | DONE for this milestone | see the commit report |

---

## Dataset (from `results/dataset_validation_clean.json`)

| Split | Images | Labelled boxes |
|---|---|---|
| train | 1481 | 8733 |
| valid | 210 | 1214 |
| test | 195 | 1371 |
| total | 1886 | 11318 |

- Problems found: 0 (malformed lines, bad class ids, bad boxes, missing or orphan labels, unreadable images).
- Exact-image leakage across splits: **0**. Within-split duplicates: 0.
- Near-duplicate leakage: **NOT MEASURED**.
- Original `csa_ppe_v1` (reference, not used): 1924 images, 11432 labelled boxes, 12 cross-split duplicate groups.
- **Open discrepancy:** the stored audit and an older note say 11,433 instances and 5,269 helmets. Both the
  current files and the project parser give 11,432 and 5,268. The cause is not yet explained. See `docs/DATASETS.md`.

---

## Detector experiments

| Experiment | Config | Status |
|---|---|---|
| A. YOLO11n pretrained | `configs/experiments/baseline_yolo11n.yaml` | EXPERIMENT REQUIRED (Colab T4) |
| B. YOLO11n from scratch | `configs/experiments/yolo11n_scratch.yaml` | EXPERIMENT REQUIRED (Colab T4) |
| C. Restored ICSD-YOLO from scratch | `configs/experiments/icsd_restored_scratch.yaml` | EXPERIMENT REQUIRED (Colab T4); restored approximation, not a reproduction (ADR 0002) |
| Optional: SHWD ablation dataset | — | not planned for this milestone |

Metrics to be reported: precision, recall, F1 (per class; overall F1 is not provided by Ultralytics), mAP@0.5,
mAP@0.5:0.95, per-class AP, confusion matrix, training curves, model size, training time. Inference latency is
NOT MEASURED: the benchmark setup is not defined yet.

---

## Modules

| Module | Type | Implemented | Tested | Validated |
|---|---|---|---|---|
| Detector | learned (Ultralytics) | yes | smoke only | **no** |
| WPAM | rule-based (geometric gate + Hungarian) | yes | yes (63 + runner) | **no** — annotations missing |
| Tracking | rule-based (IoU) | yes | yes | **no** — no annotated video |
| TSMM | rule-based (evidence window) | yes | yes | **no** — no annotated video |
| SCRM | rule-based (three-state rules) | yes | yes | **no** — no labels |
| CRAM | RULE-BASED PROTOTYPE | yes | yes | **no** — not a validated risk model |

---

## Human input required (cannot be produced by code)

1. **Annotation** (see `docs/EVALUATION.md`):
   - WPAM worker→PPE links, on the test images.
   - SCRM compliance labels (COMPLIANT / NON_COMPLIANT). The target count is not fixed: the plan gives two ranges.
   - Tracking identities on annotated video clips.
   - TSMM sequences and occlusion intervals. Only one CSA-Vid clip is accepted so far (`haringey_16.webm`); the target is 10–20.
2. **Licence decision.** No LICENSE file. ICSD-YOLO has no license file; its header says `Ultralytics AGPL-3.0`.
   Ultralytics is AGPL-3.0. **License decision required: Yes.**
3. **Conference or journal submission.** Not decided. No acceptance exists.
4. **Running the Colab notebook.** Set `REPO_COMMIT` in the notebook to the hash of the pushed commit, then run cells in order.

---

## Remaining work (in priority order)

1. Run the three detector experiments on Colab (human action), then `collect_results.py` and `make_figures.py`.
2. Human annotation, then the module evaluation join script (not yet written).
3. Error-analysis generator (taxonomy is ready).
4. Dashboard: worker history, alerts, timestamps and camera fields from `runner.py` output.
5. Ablation on the trained detector, then the ablation figure.
6. Comparison report (Markdown/HTML) and the remaining figures (architecture, WPAM and tracking examples, TSMM behaviour).
7. Paper draft from the template, using measured results only.

---

## Remaining research risks

- The detector may not beat the baseline, and the paper's claims depend on measured results.
- The WPAM gate margin (0.10), the SCRM thresholds (0.7 / 0.3) and the minimum evidence (5) are provisional.
- PPE occlusion looks like absence, so an occluded worker can be reported NON_COMPLIANT.
- The IoU tracker's identity breaks under fast motion or occlusion longer than `max_age`.
- The clean dataset removes exact duplicates only; near-duplicates may still leak.
- The ICSD-YOLO restoration is an approximation (ADR 0002).
- CRAM weights are design choices and are not validated.
