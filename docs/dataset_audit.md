# Dataset audit findings (Week 2)

Full statistics: `results/audits/<name>/audit.json`; visual spot-checks: `samples.jpg`.
Tooling: `scripts/audit_dataset.py` over the registry `configs/datasets.yaml`.

## CHV — primary multi-PPE training candidate ✅

- 1,330 images, 9,209 instances, 6 classes; official 1064/133/133 train/valid/test
  splits ship with the dataset (we reuse them verbatim for comparability).
- Class mapping **verified** against the authors' published distribution figure
  (rank-order match): `0=person, 1=vest, 2=helmet_blue, 3=helmet_red,
  4=helmet_white, 5=helmet_yellow`. For CSA-YOLO, helmet colors merge → `helmet`
  (merged: 3,538 helmet instances).
- Person boxes are **full-body and predominantly large** (3,023/3,887 large) — exactly
  what WPAM association needs; 696 images contain vests, 1,323 contain persons.
- Visual spot-check: annotations align well; multi-worker scenes common (good for
  association ground truth). Caveats: ~small fraction of studio/stock imagery
  (including non-site scenes); flagged for possible filtering in the merge recipe.

## SHWD — auxiliary helmet data only ⚠️

- 7,581 images, 120,561 instances: `hat` 9,044 (helmeted heads), `person` 111,514
  (**bare heads, NOT full bodies** — sourced partly from SCUT-HEAD classrooms),
  plus **3 mislabeled `dog` instances** (annotation noise found by the audit).
- 81,402 of the `person` boxes are small (<32² px) — crowded-scene head boxes.
- 2 corrupt/unreadable image files detected.
- **Decision:** SHWD must never contribute a full-body `person` class. Usable as
  auxiliary head/helmet data (e.g., a helmet-vs-bare-head auxiliary training source)
  and for helmet-scale robustness. Many scenes are non-construction; if used, filter.

## CSS (Roboflow Construction Site Safety v1, via HF mirror) — Week 3 ✅

- 398 images (307/57/34 official splits), COCO format, 17 classes. Public download,
  no credentials (keremberke/construction-safety-object-detection).
- Contributes the negative classes: `no-hardhat` (→ `no_helmet`), `no-safety vest`
  (→ `no_vest`), plus person/hardhat/safety-vest.

## Unified dataset: csa_ppe_v1 (Week 3)

Recipe: `configs/merge.yaml` → `training/prepare_dataset.py` (deterministic, seed 42;
manifest committed at `datasets/splits/csa_ppe_v1_manifest.csv`).
Composition: CHV (official splits) + CSS (official splits) + SHWD capped at 400 images
(head boxes: hat→helmet, bare-head→no_helmet; seeded 80/10/10).

- 1,924 images — 1,516 train / 213 valid / 195 test; 11,433 instances as stored at the time of this note.
  **Recount on 2026-10-26: 11,432 instances (helmet 5,268, not 5,269).** See `docs/DATASETS.md` (known discrepancy).
- person 3,956 · helmet 5,269 · vest 1,837 · no_helmet 333 · **no_vest 38** ⚠️

### Critical finding: negative-class imbalance

`no_vest` (38 instances) is untrainable as a detection class; `no_helmet` (333) is
marginal. Design consequences (recorded now, before Phase 2):
1. SCRM's primary evidence for "missing vest" must be **reasoned absence** (worker has
   no associated vest detection over TSMM-confirmed frames), with `no_vest` detections
   as corroboration only, never sole evidence.
2. Detector training (Week 4) reports per-class results; `no_vest` is expected to be
   weak and is disclosed as such — or dropped from v1 training if mAP is degenerate.
3. Acquisition of the larger css-data (2.6k imgs, Kaggle credentials) remains open as
   the fix for v2.

## Remaining gaps

1. **CSA-Vid annotation** — seed clips acquired (see `docs/csa_vid_protocol.md`);
   annotation effort is the long pole.
2. Boots/gloves classes deferred to v2 (data too thin in construction domain).
