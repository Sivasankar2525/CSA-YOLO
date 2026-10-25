# CSA-YOLO — Project Status

**Last updated:** 2026-10-26 (repository finalization before Colab training)
**Machine-readable records:** `results/PRE_TRAINING_AUDIT.json`, `results/metrics_summary.json`

> **CSA-YOLO is NOT experimentally validated.** The software is implemented and unit tested.
> No detector has been trained, and no research metric (detection, association, tracking,
> compliance, ablation) has been measured. Anything not produced by an executed experiment is
> `NOT MEASURED`, `TBD` or `EXPERIMENT REQUIRED`.

Vocabulary: **IMPLEMENTED** (code runs) · **TESTED** (automated tests pass) ·
**EXPERIMENTALLY VALIDATED** (measured on labelled data by a reproducible run).
Today, every module is IMPLEMENTED and TESTED. None is EXPERIMENTALLY VALIDATED.

Learned vs. rule-based vs. prototype components are listed in `docs/MODULES.md`.
Only the detector is learned. WPAM, tracking, TSMM and SCRM are rule-based. CRAM is a rule-based prototype.

---

## COMPLETED (IMPLEMENTED + TESTED)

| Area | Status | Evidence |
|---|---|---|
| Detection contract | IMPLEMENTED, TESTED | `tests/test_detection_contract.py` |
| WPAM: spatial gate, one-to-one Hungarian assignment | IMPLEMENTED, TESTED | `tests/test_wpam.py` (63 tests, brute-force optimality checks) |
| Tracking: IoU matching, pruning on every frame | IMPLEMENTED, TESTED | `tests/test_tracking.py` (14) |
| TSMM: evidence window, cleanup on every frame | IMPLEMENTED, TESTED | `tests/test_tsmm.py` (19) |
| SCRM: PRESENT / ABSENT / UNKNOWN; min. 5 observations | IMPLEMENTED, TESTED | `tests/test_scrm.py` (19) |
| CRAM: rule-based prototype, scores scored workers only | IMPLEMENTED, TESTED | `tests/test_cram.py` (20) |
| Pipeline and module toggles | IMPLEMENTED, TESTED | `tests/test_pipeline.py` (17), `tests/test_pipeline_modules.py` (11) |
| Training launcher: `--experiment`, `--device`, `--epochs`, dataset path resolution | IMPLEMENTED, TESTED (no training run) | `tests/test_train_launcher.py` (14) |
| Evaluation metric definitions | IMPLEMENTED, TESTED (synthetic arithmetic only) | `tests/test_eval_metrics.py` (17) |
| Metrics summary builder (artifact-only) | IMPLEMENTED, TESTED | `tests/test_metrics_summary.py` (7) |
| ICSD-YOLO adapter (restored `Add_modules`, patches) | IMPLEMENTED, TESTED | `tests/test_icsd_adapter.py` (6) |
| Dataset leakage cleanup (`csa_ppe_v1_clean`) | IMPLEMENTED, TESTED on real data | `tests/test_clean_dataset.py` (8) |
| Repository hygiene (ignores, submodule clean) | DONE | see Git status section |

**Test status:** `pytest -q` → **243 passed, 0 failed, 0 skipped, 0 xfailed.**
(One harmless `timm` FutureWarning from the ICSD import.)

**Smoke tests run on the final code (not research results):**
- `scripts/demo.py` on `CHV_ppe_0003.jpg` (clean test split): exit 0.
- `evaluation/run_ablation.py` on 5 clean test images with the COCO-pretrained YOLO11n: all six
  configurations ran, and the summary records the module settings used.
- No training run was started on the Mac.

---

## DEFECTS

- All six audit defects were fixed earlier, with regression tests. No known open defect remains.
- Discrepancy under investigation: the stored audit of the original dataset reports 11,433 instances and
  5,269 helmets. The current files give 11,432 and 5,268 (both by direct count and by the project parser).
  See `docs/DATASETS.md`. No conclusion depends on it.

---

## DATASET

| | Original `csa_ppe_v1` (reference) | Research `csa_ppe_v1_clean` |
|---|---|---|
| train | 1516 | 1481 |
| valid | 213 | 210 |
| test | 195 | 195 |
| total | 1924 | 1886 |
| exact-image leakage (SHA-256) | duplicates present | **0** |

- 38 exact duplicates removed: 35 from train, 3 from valid, 0 from test.
- Removal record: `datasets/splits/csa_ppe_v1_clean_removed.csv`.
- Clean dataset box count (from labels): 11,318 (original 11,432; difference equals the removed boxes).
- Near-duplicates (re-encoded or cropped copies): **NOT MEASURED.** Exact hashing cannot find them.
- The original zip `csa_ppe_v1.zip` (517 MiB) is kept locally for upload and is never committed.

---

## ICSD-YOLO integration

- Submodule `external/icsd_yolo` is pinned at `47018ed` and is **clean** (no changes).
- The adapter (`training/icsd.py`) copies the restored `Add_modules` package into the submodule tree
  at runtime. This copy is excluded from git **locally** via the submodule's
  `.git/modules/.../info/exclude`. Nothing is added to the submodule's tracked files.
- Upstream tracks some `cpython-39` bytecode files of its own. They are left untouched.
- Restored experiment (`icsd_restored_scratch`) is a restored approximation, not a faithful reproduction
  (ADR 0002). Its complexity does not match the paper's reported figures.

---

## NOT YET DONE (EXPERIMENT REQUIRED / TBD)

- **GPU detector training:** `baseline_yolo11n`, `yolo11n_scratch`, `icsd_restored_scratch`
  (optional `csa_ppe_v1_noshwd`). Colab T4. Metrics: **NOT MEASURED**.
- **Ablation on a trained detector:** **EXPERIMENT REQUIRED.**
- **Ground truth** for WPAM, SCRM, tracking, TSMM: **NOT YET COMPLETED** (human annotation required).
  Schema: `docs/EVALUATION.md`.
- **CSA-Vid:** 1 of 10–20 clips accepted (`haringey_16.webm`). Annotation not started.
- **CRAM validation:** not possible without risk ground truth. Qualitative case studies only.
- **Paper and presentation:** `paper/` and `ppt/` are empty. Results: **TBD**.

---

## REQUIRES HUMAN INPUT

1. **License.** `REQUIRES_USER_DECISION`. No LICENSE file has been added.
   - ICSD-YOLO has no license file; its config header says `Ultralytics AGPL-3.0`.
   - Ultralytics is an AGPL-3.0 dependency.
   - The README's third-party section records these facts.
   **License decision required: Yes.**
2. **Annotation:** who annotates, and the SCRM target count (plan says both 50–100 and 100–200).
3. **Conference submission:** none made; no acceptance exists.

---

## Git status (see the finalization report for the exact list)

- Nothing committed in this phase. Commit only after the user approves the list.
- Intentionally ignored: `csa_ppe_v1.zip`, `datasets/*` (except `splits/` and `LICENSES.md`), `weights/*`,
  `results/*` (except the four small audit JSON files), `runs/`, `.venv/`, `__pycache__/`, `*.pyc`,
  `*.zip`, `*.pt`, `*.log`.
- Versioned audit records: `results/metrics_summary.json`, `results/PRE_TRAINING_AUDIT.json`,
  `results/dataset_audit.json`, `results/dataset_audit_clean.json`.

---

## Remaining research risks

- Detector results may not beat the baseline; the paper's claims depend on measured results.
- The WPAM gate margin (0.10), SCRM thresholds (0.7 / 0.3) and minimum evidence (5) are provisional.
- Occlusion of PPE looks like absence, so an occluded worker can be NON_COMPLIANT.
- The tracker's identity breaks under fast motion or occlusion longer than `max_age`.
- The clean dataset removes exact duplicates only. Near-duplicates may still leak.
- The ICSD-YOLO restoration is an approximation (ADR 0002).
- CRAM weights are design choices and are not validated.
