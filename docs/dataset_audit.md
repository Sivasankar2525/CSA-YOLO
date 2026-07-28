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

## Gaps after Week 2 (drive Week 3)

1. **No `no_vest` / `boots` classes anywhere yet.** Candidates: Roboflow
   "Construction Site Safety" (10 classes incl. NO-Hardhat/NO-Safety-Vest) — requires
   a free Roboflow API key (user action), or SH17 (17 classes, manufacturing domain,
   Kaggle — requires Kaggle credentials).
2. **Video for TSMM still outstanding** (project risk #1) — collection must start.
3. Merge recipe + unified class taxonomy to be scripted (`training/prepare_dataset.py`)
   once the negative-class source is settled.
