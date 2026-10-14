# Week 4 report — Phase 2 open: ICSD-YOLO restoration + training infrastructure

**Research question:** can the published ICSD-YOLO release actually serve as our base
detector — and if not, what is the honest, reproducible alternative?

## Completed

- **ICSD-YOLO release forensics + restoration (ADR 0002)**: the authors' code
  (PrintSC/code @ 47018ed) cannot instantiate any of its own models as shipped.
  Eleven distinct defects catalogued and repaired via `training/icsd.py` +
  `training/icsd_compat/` — missing packages, a typo'd import, five argument-handling
  bugs in `parse_model`, a filename-regex scale collapse, a hijacked dataset resolver,
  and import-time argv parsing. **No tracked submodule file is modified**: fixes are
  untracked package copies plus in-memory source patches whose anchors fail loudly
  (and are unit-tested) against commit drift.
- **Key finding — the release cannot reproduce the paper**: the five variant YAMLs
  are mutually inconsistent architectures, not one architecture at five scales; the
  scale-collapse bug means the authors' own runs silently used n-scale; and the
  reconstructed X variant measures 486 GFLOPs vs the paper's reported 116.8. No
  reconstruction we tested closes the gap (the focus block's lighter internal
  configuration crashes in its own forward). We therefore frame our detector as a
  *restored* ICSD-YOLO, not a reproduction — the restoration itself is documented
  reproducibility work.
- **Base detector decision**: restored x-arch (the only YAML matching the paper's
  described ConvStemBlock + FocusBlock design) at n-scale — 5.67M params /
  17.7 GFLOPs, the same size class as the YOLO11n baseline.
- **MPS enablement**: the focus blocks' PSP-style pooling is unimplemented for
  non-divisible sizes on Apple MPS; adapter swaps in an `MPSSafeAdaptiveAvgPool2d`
  (CPU round-trip only for the offending op), re-applied via trainer callback after
  ultralytics rebuilds the model. Verified: forward pass on MPS.
- **Training infrastructure**: `training/train_detector.py` — config-driven launcher
  (experiment = YAML under `configs/experiments/`, machine-readable summary JSON per
  run); experiments defined for YOLO11n-pretrained, YOLO11n-scratch, ICSD-restored-
  scratch.
- **SHWD ablation dataset built**: `configs/merge_noshwd.yaml` →
  `csa_ppe_v1_noshwd` (1,196/173/155 images), manifest committed. Same seed and
  taxonomy as v1 for direct comparability.
- **Throughput probe** (`scripts/probe_throughput.py`, 2-epoch runs on M-series MPS,
  batch 16 @ 640): YOLO11n ≈ TBD s/epoch; restored ICSD ≈ TBD s/epoch.
- **Tests**: 34 passing (6 new adapter tests incl. patch-anchor drift guards).

## Remaining

- Full training runs (100 epochs): baseline_yolo11n, yolo11n_scratch,
  icsd_restored_scratch, baseline on csa_ppe_v1_noshwd — launch per the probe's
  local-vs-cloud verdict.
- Per-class P/R/mAP50/mAP50-95 comparison table from those runs.
- CSA-Vid: annotate haringey_16 in CVAT; harvest more Haringey-series clips.

## Reviewer comments (hostile ESWA reviewer)

1. *"Your 'base detector' is not the published ICSD-YOLO — you modified their code
   until it ran and picked one of five inconsistent configs. Why should I believe
   comparisons involving it?"* — Fair. Mitigation: every change is catalogued,
   anchored, and tested; the paper's own numbers are unreproducible from the release
   (no weights, non-running code, 4× complexity mismatch), which we state rather than
   hide. Our contributions are measured against *our* controlled baselines, not
   against the paper's table.
2. *"The interesting comparison is against a well-maintained detector, not a broken
   release."* — Agreed: YOLO11n (pretrained and scratch) is the primary baseline;
   restored ICSD-YOLO is kept because the framework claims backbone-agnosticism and
   the restoration documents the state of the field's reproducibility.
3. *"mlp_ratio/ctx_dim choices in gap 9 are your interpretation of author intent."*
   — Yes, and flagged as such; the alternative configurations either crash or explode
   parameters, and the choice is ablatable.
4. *"Still no trained detector at Week 4's close."* — Infrastructure-first was
   deliberate (a silently broken base detector would poison every downstream week);
   runs are configured and launch immediately after the probe.

## Improvement plan

- Launch the four training runs; if MPS per-epoch time makes 100 epochs infeasible
  locally, move to Colab/Kaggle CUDA with the same configs (launcher is
  device-agnostic) and record environment in the run summaries.
- Add the per-class metrics table + SHWD ablation verdict to this report when runs
  complete.
- Week 5 (WPAM) does not depend on final detector weights — the detection contract
  isolates it — so it starts on schedule regardless.
