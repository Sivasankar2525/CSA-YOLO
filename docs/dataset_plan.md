# Dataset plan & risk assessment

Status: Week 1 assessment. Acquisition and verification happen in Weeks 2–3.

## Requirements derived from the experiments we must run

| Experiment | Data needed |
|---|---|
| Detector fine-tuning (Phase 2) | Images: `person` + PPE classes, YOLO format, ≥5k instances/class preferred |
| WPAM association accuracy (Phase 3) | 100–200 frames with **ground-truth person↔PPE ownership** (we annotate this ourselves — no public set provides it) |
| TSMM false-alarm reduction (Phase 4) | **Video** with tracked workers, occlusion events, per-frame PPE ground truth |
| SCRM compliance accuracy (Phase 5) | Frames labeled compliant/non-compliant per worker per zone |
| CRAM case studies (Phase 6) | Scenario clips (reused from TSMM video set) |

## Candidate public image datasets (evaluate in Week 2)

| Dataset | Classes (approx.) | Notes |
|---|---|---|
| **SHWD** (Safety Helmet Wearing) | helmet, head (~7.5k images) | Helmet-only; no vest |
| **GDUT-HWD** | helmet variants | Helmet-only |
| **CHV** (Wang et al. 2021) | person, vest, 4 helmet colours (1.3k images) | Small but multi-PPE |
| **SODA** | person, helmet, vest, machinery | Broader site objects |
| **Roboflow "Construction Site Safety"** family | person, helmet/no-helmet, vest/no-vest, boots in some variants | YOLO-format export; license must be checked per dataset |

Selection criteria: multi-PPE coverage (helmet+vest minimum, boots desirable), license
permitting academic redistribution of *splits* (we publish split files, not images),
annotation quality on visual spot-check. Final choice may merge 2 sources with class
remapping — exact recipe will be scripted (`training/prepare_dataset.py`) and versioned.

## THE critical gap: video for TSMM (project risk #1)

All candidate public datasets are **still images**. TSMM's headline claim (false-alarm
reduction over time under occlusion) cannot be evaluated on stills.

**Mitigation plan (decided Week 1):**
1. Collect 10–20 short construction-site video clips (30–120 s): own recordings where
   feasible + permissively-licensed online footage (license logged per clip).
2. Annotate a subset per-frame (worker boxes + IDs + PPE state) with CVAT
   (interpolation-assisted — feasible effort: ~2–4 h per annotated minute).
3. Package as a small benchmark ("CSA-Vid") with a documented annotation protocol —
   this becomes a *secondary contribution* of the paper (temporal PPE compliance
   benchmark), which strengthens rather than pads the submission.
4. Fallback if collection stalls: synthetic occlusion injection on image sequences
   (dropout of detections) — weaker, explicitly labeled as simulation in the paper,
   used only in addition to (never instead of) real video.

## Reproducibility rules

- Exact train/val/test splits committed as text files under `datasets/splits/`.
- Every preprocessing/remapping step scripted — no manual file edits.
- Per-source licenses recorded in `datasets/LICENSES.md` before first training run.
