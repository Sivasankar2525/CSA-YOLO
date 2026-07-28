# Related work map & novelty calibration

Purpose: state precisely what already exists so every CSA-YOLO claim is calibrated, not
aspirational. Reviewers reject overclaimed novelty faster than modest novelty.
Status: Week 1 survey from targeted searches; to be deepened into the full literature
review during Weeks 2–3 (each entry below gets read in full, not just abstract-level).

## 1. Base detector

- **ICSD-YOLO** (Shi, Chen, Zhang, Chang, Chen, Wang — *ESWA* 2025,
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425046093),
  [code](https://github.com/PrintSC/code)). YOLOv12-family detector with ConvStemBlock +
  FocusBlock; +1.2% mAP50:95 and −41.3% FLOPs vs YOLOv12-X. **Our backbone — we do not
  claim any detection-architecture novelty.**

## 2. PPE detection (object-level) — crowded field, not our contribution

- Fang et al., *Automation in Construction* 2018 — non-hardhat-use detection (Faster R-CNN).
- Nath, Behzadan, Paal, *Automation in Construction* 2020 — Pictor-v3, YOLO-based PPE
  compliance "machine" classification (worker/hat/vest compound classes).
- Wang et al. 2021 — **CHV dataset** (1,330 images; person, vest, 4 helmet colours);
  YOLOv5x 86.55% mAP ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8156681/)).
- Sec-YOLO, GCEA-YOLO, OAM-YOLO (2024–25) — continued detector-centric improvements.

**Position:** CSA-YOLO differs by operating *after* any such detector: detector-agnostic
worker-level intelligence. This framing must be explicit in the introduction.

## 3. Worker–PPE association (WPAM's neighborhood) — prior art EXISTS

- **Helmet–head region matching + state tracking**, *Automation in Construction* 2025
  ([S0926580525000275](https://www.sciencedirect.com/science/article/abs/pii/S0926580525000275),
  [HKUST record](https://researchportal.hkust.edu.hk/en/publications/detection-of-helmet-use-among-construction-workers-via-helmet-hea/))
  — the closest known prior work. **Method (Week 2 pass):** YOLOv9 detections +
  YoloPose keypoints to extract the head region, helmet box matched to head region,
  StrongSORT worker tracking, and preceding-frame matching integration to update
  helmet-wearing state. Reports 98.89% accuracy on a **self-built (non-public) dataset**.
  **Limits we exploit:** single PPE type (helmet only), pairwise matching (no joint
  assignment), non-public evaluation data. → WPAM baseline: reimplement their
  head-region matching rule; TSMM comparison: their preceding-frame integration.
- **Fuzzy-logic-based helmet state tracking**, *Sensors* 2025
  ([DOI](https://doi.org/10.3390/s25206487)) — temporal helmet-state smoothing via
  fuzzy logic; a second temporal competitor for TSMM's related-work table. Helmet-only.
- Pose-estimation-based PPE compliance, *Automation in Construction* 2025
  ([S0926580525002717](https://www.sciencedirect.com/science/article/pii/S0926580525002717))
  — cascaded detection + pose keypoints; classifies worker state into 4 categories.
- Nath et al. 2020 (above) sidesteps association entirely via compound classes
  (e.g. "worker-with-hat") — known to scale poorly with PPE-type count (class explosion:
  2^k compound classes for k PPE types).

**Calibrated WPAM claim (draft):** not "first to associate PPE with workers", but:
*a unified multi-PPE association formulated as body-region-constrained bipartite
assignment (jointly over helmet, vest, boots), evaluated against greedy-distance and
plain-IoU baselines with a dedicated association-accuracy protocol.* The joint
formulation + evaluation protocol is the contribution; must beat the pairwise heuristics.

## 4. Temporal smoothing of compliance state (TSMM's neighborhood)

- The helmet-state-tracking paper above includes state tracking for helmets — TSMM must
  differ: **per-worker multi-PPE temporal memory with an explicit false-alarm-reduction
  evaluation protocol** (flicker rate, occlusion-robustness curves vs. window size N).
- MOT foundations: ByteTrack (Zhang et al., ECCV 2022), BoT-SORT — used, not claimed.
- Gap (to confirm in the Week 2–3 deep pass): no work found yet that *quantifies*
  false-alarm reduction for PPE compliance under occlusion as a primary metric. If this
  survives the deep pass, it anchors TSMM as the strongest contribution.

## 5. Compliance reasoning & risk (SCRM/CRAM's neighborhood)

- Existing systems hardcode rules (helmet+vest) at detection level; zone-conditional
  requirements (e.g. harness only in height-work zones) are rarely modeled. SCRM claim:
  worker-level, zone-aware compliance reasoning decoupled from the detector.
- Construction risk scoring exists in safety-management literature (risk matrices,
  OSHA severity classifications) but is rarely coupled to *live vision output per worker
  with human-readable rationale*. CRAM claim: explainable risk assessment grounding
  severity weights in OSHA guidance (e.g., 29 CFR 1926 Subpart E; exact clauses to be
  cited when CRAM is designed in Week 10).

## 6. Datasets

See `docs/dataset_plan.md` — SHWD, GDUT-HWD, CHV, SODA, Roboflow construction-safety sets;
and the video-data gap that TSMM evaluation must solve.

## Standing risks for the paper

1. The two 2025 *Automation in Construction* papers are the works a reviewer will cite
   against us — the related-work section must compare against them explicitly, and WPAM/TSMM
   experiments should include a reimplementation of their matching heuristic as a baseline.
2. If association via pose keypoints is strictly better than box-region constraints, WPAM
   should consider a pose-assisted variant (decision point at Week 6, before implementation).
