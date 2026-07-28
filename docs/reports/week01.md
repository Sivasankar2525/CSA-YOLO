# Week 1 report — Phase 0: Environment, base paper, repo foundation

**Research question this week answers:** Is the base paper's detector actually available and
reproducible, and can we build a verified, seeded, end-to-end detection foundation for the
four research modules to stand on?

## Completed

- **Backbone verified**: official ICSD-YOLO code is public (`PrintSC/code`), contains
  ConvStemBlock/FocusBlock/SparseSelfAttenFocusBlock + ICSD-YOLO n/s/m/l/x YAMLs on an
  Ultralytics 8.3.74 fork with ByteTrack/BoT-SORT. Pinned at `47018ed` as a git submodule.
  No pretrained weights released → we fine-tune in Phase 2. Full ADR: `docs/adr/0001`.
- **Environment**: Python 3.11.13 venv; torch 2.13.0 (Apple MPS verified), ultralytics
  8.4.108, pinned `requirements.txt` + `environment.yml`.
- **Repo scaffold**: `csa_yolo` installable package; typed detection contract
  (`Detection`/`FrameDetections`) that isolates all research modules from the backbone;
  utils for seeding, logging, typed YAML config; no hardcoded paths.
- **End-to-end verification**: `scripts/verify_detection.py` — config → seed → model →
  MPS inference → typed parsing → annotated image + JSON (`results/verify_detection/`).
  4 persons detected on sample image. PASSED.
- **Tests**: 14 unit tests passing (config validation, detection contract parsing,
  geometry, seeding repeatability).
- **Research docs**: related-work map with calibrated novelty claims
  (`docs/related_work.md`) — including two adjacent 2025 *Automation in Construction*
  papers (helmet–head matching + state tracking; pose-based compliance) that WPAM/TSMM
  must explicitly compare against — and dataset plan with the video-data risk and
  CSA-Vid mitigation (`docs/dataset_plan.md`).

## Key decisions

1. Real ICSD-YOLO code as submodule (no stand-in backbone) — ADR 0001.
2. Detector-agnostic module contract — enables the ablation ladder and backbone swap.
3. CSA-Vid micro-benchmark planned to close the video-evaluation gap for TSMM.

## Next (Week 2 — Phase 1 start)

- Download + audit candidate datasets (classes, license, quality); choose merge recipe.
- Begin deep literature pass (read the two 2025 AutoCon papers in full).
- Start video clip collection for CSA-Vid.
