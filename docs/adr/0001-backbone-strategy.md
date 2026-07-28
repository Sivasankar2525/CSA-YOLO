# ADR 0001 — Detection backbone strategy

**Date:** 2026-07-28 · **Status:** Accepted

## Context

CSA-YOLO's contributions (WPAM/TSMM/SCRM/CRAM) sit *after* detection. The base paper,
ICSD-YOLO (Shi et al., *Expert Systems with Applications*, 2025), provides the detector. The
execution plan allowed a fallback to stock YOLOv8/v11 if the authors' code was unavailable.

## Findings (verified 2026-07-28)

- The official code **is public**: <https://github.com/PrintSC/code>, inspected at commit
  `47018ed6028e29567bf990053b3833a578d5e492`.
- It is a **vendored Ultralytics 8.3.74 fork** (package renamed `models`, imported as
  `from models import YOLO`) with the paper's custom modules present and registered:
  `ConvStemBlock.py` (134 LOC), `FocusBlock.py` (95 LOC), `SparseSelfAttenFocusBlock.py`
  (115 LOC), plus `ICSD-YOLO{n,s,m,l,x}.yaml` model definitions.
- ByteTrack and BoT-SORT trackers are included in the fork — usable for worker tracking (TSMM).
- **No pretrained ICSD weights are released** (README: "will gradually be opened up") and the
  authors' `requirements.txt` is a corrupted UTF-16 Windows conda dump — unusable.

## Decision

1. **Use the authors' real ICSD-YOLO implementation** — no stand-in backbone. Vendored as a
   git submodule at `external/icsd_yolo`, pinned to commit `47018ed`.
2. Because no pretrained weights exist, Phase 2 (Weeks 4–5) **fine-tunes ICSD-YOLO ourselves**
   on the PPE dataset, initialized from COCO-pretrained YOLO weights where layer shapes allow.
3. All downstream modules consume the typed contract in `csa_yolo/detection/detector.py`
   (`Detection` / `FrameDetections`), never raw Ultralytics objects — so the backbone can be
   swapped (stock YOLO11 for early development, ICSD-YOLO for reported results) without
   touching WPAM/TSMM/SCRM/CRAM. Week 1 verification ran on stock `yolo11n.pt`.
4. Our own environment (`requirements.txt`: Python 3.11, torch 2.13.0, ultralytics 8.4.108)
   is the single source of dependency truth; the fork's internal 8.3.74 package is used only
   for ICSD training/inference in Phase 2, isolated behind the detection contract.

## Consequences

- (+) Faithful to the base paper; reviewer cannot object to a substituted backbone.
- (+) Pinned commit gives exact reproducibility of the base detector.
- (−) We bear the training cost for detector weights (planned in Weeks 4–5; Colab/Kaggle GPU
  fallback if MPS training is too slow — decision point recorded for Week 4).
- (−) The fork's `models` package name can shadow imports; it is only ever imported inside
  the Phase 2 training adapter, never in `csa_yolo` core.
