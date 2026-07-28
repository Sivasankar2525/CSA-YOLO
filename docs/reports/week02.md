# Week 2 report — Phase 1: Dataset acquisition & audit

**Research question:** do public multi-PPE datasets meet our class/quality/license
requirements, and where exactly must our own collection fill gaps?

## Completed

- Downloaded + extracted **CHV** (419 MB, 1,330 imgs) and **SHWD** (1.0 GB, 7,581 imgs)
  from public mirrors; licenses recorded in `datasets/LICENSES.md`.
- Built format-agnostic audit tooling: `csa_yolo/data/` (shared `ImageAnnotation` types,
  VOC + YOLO parsers, statistics engine), registry-driven `scripts/audit_dataset.py`;
  21 unit tests passing.
- Ran both audits (stats + visual grids). Key findings in `docs/dataset_audit.md`:
  CHV = primary multi-PPE set (mapping verified against authors' figure, official splits
  reused); SHWD = head-boxes only, auxiliary use; audit caught 3 mislabeled instances +
  2 corrupt files.
- Deep-pass details on closest competitors added to `docs/related_work.md`:
  AutoCon 2025 (YoloPose head region + StrongSORT + preceding-frame state, helmet-only,
  self-built data) and Sensors 2025 (fuzzy state tracking) → our baselines/comparators.

## Blockers needing user action

- **Roboflow API key** (free account) to download "Construction Site Safety" with
  NO-Hardhat/NO-Safety-Vest classes — or Kaggle credentials for SH17.

## Next (Week 3)

- Negative-class dataset acquisition; unified taxonomy + scripted merge recipe with
  committed split files; begin CSA-Vid video collection.
