# CSA-YOLO module reference

This document describes what the implemented code does. It makes no performance claims.
Every number below is a configured parameter or a design choice. None is a measured result.
Parameters live in `configs/default.yaml`; defaults in code are in `csa_yolo/pipeline.py`
(`DEFAULT_SETTINGS`), and a test keeps the two equal.

Status of every module: **implemented and unit tested; not experimentally evaluated.**

Pipeline order: Detector → WPAM → Tracking → TSMM → SCRM → CRAM.

---

## Detector (input contract)

- **Input:** BGR frame (numpy array).
- **Output:** `FrameDetections` of `Detection(xyxy, confidence, class_id, class_name)`.
- The worker class is `person`. PPE classes are matched by name, case-insensitively:
  helmet classes `helmet, hat, hardhat`; vest classes `vest, safety_vest`.
- Other modules never see raw Ultralytics objects.

---

## WPAM: worker–PPE association

**Input.** Worker detections and PPE detections for one frame.

**Spatial regions.** Each worker box `(x1, y1, x2, y2)` of height `h` defines:
- Head region: from `y1 − 0.15h` to `y1 + 0.25h`, expanded horizontally by `0.15h`. Used for helmets.
- Torso region: from `y1 + 0.20h` to `y1 + 0.75h`, full worker width. Used for vests.

**Spatial gate (`association.gate_margin_ratio`).** A PPE item is a *candidate* for a worker
only if its **centre** lies inside that region expanded by `gate_margin_ratio × h`.
Candidates outside the gate are never assigned. The default `0.10` is **PROVISIONAL**: it is
an engineering choice, not tuned or validated. Set it on a validation split before reporting
any association result.

**One-to-one assignment.** Helmets and vests are assigned separately. For each class group:
1. Build a worker × PPE cost matrix. Cost for a gated pair is the PPE centre's distance to the
   region centre, normalised by region width and height. Ungated pairs get a sentinel cost.
2. Solve with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. Keep only pairs whose cost is not the sentinel.

This gives a global assignment: each worker receives at most one helmet and at most one vest,
and each PPE item goes to at most one worker. Cardinality is maximised first, then total cost
is minimised. Tests check this against exhaustive search on random cost matrices.

**Output.** One `WorkerPPEAssociation` per worker, in input order, with `associated_ppe`
holding only matched items. Unmatched classes are absent from the dict.

**Unknown handling.** WPAM reports matches only and makes no compliance decision. An
unmatched class in one frame is recorded as "absent" by TSMM for that frame (see TSMM).
WPAM cannot distinguish "no PPE" from "PPE hidden by occlusion". Only the temporal window
and SCRM's evidence rules limit the effect of that ambiguity.

**Known limitations.**
- The cost uses centre distance only; IoU is not used.
- Association is per frame. It does not use motion or identity.
- A worker's head region can overlap a neighbour's; the gate plus assignment resolves ties
  deterministically, not necessarily correctly.

---

## Tracking: IoU worker tracker

**Lifecycle.**
- **Match.** Each live track is matched to at most one detection: the unused detection with
  the highest IoU against the track's last box, if IoU ≥ `tracking.iou_threshold` (default 0.3).
  Tracks are visited in ID order (greedy matching).
- **Create.** Each unmatched detection starts a new track with the next integer ID. IDs are
  never reused within a tracker instance.
- **Prune.** A track unmatched for more than `tracking.max_age` consecutive frames (default 30)
  is removed. Pruning runs on every frame, including frames with no workers.

**Identity assumption.** A worker's box overlaps its box in the previous frame. Fast motion,
an occlusion longer than `max_age` frames, or two workers swapping positions will break
identity. An ID identifies one continuous track, not a physical person across long gaps.

**Not measured.** Identity switches, MOTA, IDF1 and tracking accuracy require annotated video
(see `docs/PROJECT_STATUS.md`). They are NOT YET MEASURED.

---

## TSMM: temporal safety memory

**Per-frame observation.** For each tracked worker, each required PPE class gets one
observation per frame:
- **Present** if the matched PPE confidence ≥ `temporal.presence_confidence` (default 0.5).
- **Absent** otherwise. This includes "no matched PPE this frame", recorded with confidence 0.

Recording absence for every required class is what lets a helmet that is never seen become
evidence. Without it, the class would never enter the history.

**Window.** The most recent `temporal.window_size` observations (default 5) are kept per
worker and class. Older observations drop out.

**Violation duration.** Consecutive absent frames are counted per class and reset when the
class is present. CRAM uses this count.

**Cleanup.** Each state records `last_seen_frame`. `remove_old_tracks` runs on every frame with
`max_age = tracking.max_age`, so memory is bounded by workers seen in the last `max_age` frames.
Regression tests cover a 1000-frame run and a 2000-frame churn scenario.

**Uncertainty.** TSMM does not claim a confidence in the observations. It reports the present
ratio and observation count. SCRM decides how much evidence is enough.

**Not measured.** The effect of TSMM on false alarms or flicker needs annotated violation
intervals (NOT YET MEASURED).

---

## SCRM: compliance reasoning

**Per-class decision.** From a class's window of `n` observations with present ratio `r`:

| Condition | Class state |
|---|---|
| `n < compliance.min_evidence_frames` (default 5) | UNKNOWN |
| `r ≥ compliance.present_threshold` (default 0.7) | PRESENT |
| `r ≤ compliance.absent_threshold` (default 0.3) | ABSENT |
| otherwise | UNKNOWN (conflicting evidence) |

**Worker status.**

| Condition | Status |
|---|---|
| any required class ABSENT | **NON_COMPLIANT** (violation evidenced) |
| every required class PRESENT | **COMPLIANT** (positive evidence for all) |
| otherwise | **UNKNOWN** |

A helmet that is not detected is **not** automatically NON_COMPLIANT. It must be absent in at
least 70% of at least 5 observations. A single frame, or a class never observed, produces UNKNOWN.

**Confidence.** The weakest required class's evidence strength, `max(r, 1 − r)`. UNKNOWN
classes contribute 0.

**Limitation.** Occlusion of PPE looks like absence. An occluded worker can therefore be
reported NON_COMPLIANT. The current architecture has no occlusion model. The rate of this error
is NOT YET MEASURED.

---

## CRAM: rule-based risk prototype

**This is a rule-based prototype risk-assessment layer. It is not a validated predictive
risk model.** There are no ground-truth risk labels, and it is not OSHA-certified. Its output
must not be used for real safety decisions.

**Scope.** Scored workers only: COMPLIANT and NON_COMPLIANT. An UNKNOWN worker has no
evidenced violation and is not scored.

**Worker score (0–100, clamped).**
- +25 for missing helmet (`helmet_weight`), +15 for missing vest (`vest_weight`).
- For a missing class with a violation longer than 10 frames: `+ (duration / 10) × 10`
  points, where `duration` is the number of consecutive violating frames.
- +5 for each present required class whose mean detection confidence is below 0.5.
  Missing classes are excluded from this penalty, so a violation is not counted twice.

**Levels.** LOW below 25, MEDIUM from 25 to below 60, HIGH from 60.

**Site summary.** Compliance counts come from the compliance results, one entry per observed
worker, so `compliant + non_compliant + unknown = total_workers`. The site score is the mean
over scored workers only. Site level is LOW below 20, MEDIUM below 50, HIGH otherwise. With no
scored workers the level is UNKNOWN.

**Validation.** The weights are design choices, not fitted values. Validation is qualitative
case study only (see `docs/PROJECT_STATUS.md`). No accuracy or calibration claim is made.
