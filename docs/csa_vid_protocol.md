# CSA-Vid: temporal PPE compliance benchmark — collection & annotation protocol

Purpose: the evaluation substrate for TSMM's headline claim (false-alarm reduction
under occlusion) and for CRAM case studies. No public video benchmark for per-worker
temporal PPE compliance exists (Week 1–2 survey) — CSA-Vid is therefore a secondary
contribution of the paper.

## Collection rules

1. Sources: (a) permissively licensed public footage (CC BY / CC BY-SA / CC0 —
   license + attribution logged per clip in `datasets/csa_vid/SOURCES.md`);
   (b) own recordings (with site permission; faces are incidental — clips are used
   for research evaluation only).
2. Clip length 30–180 s, ≥720p, ≥10 fps effective.
3. Selection criteria: ≥1 visible worker at annotatable scale; prefer clips with
   natural occlusion events (machinery, structures, worker crossings) — these carry
   TSMM's evaluation signal.
4. Target: 10–20 clips, ≥5 with occlusion events, covering helmet+vest visibility.

## Annotation protocol (CVAT, interpolation-assisted)

- Per-frame (annotate every Nth frame, interpolate, then correct): worker bounding
  box + persistent `worker_id`; per-worker attributes `helmet ∈ {on, off, unclear}`,
  `vest ∈ {on, off, unclear}`; `occluded ∈ {none, partial, full}`.
- `unclear` is a first-class label — frames where a human cannot judge PPE status
  are excluded from accuracy denominators (and reported).
- Event log per clip: ground-truth violation intervals (start/end frame, worker_id,
  missing item) — this is what false-alarm metrics compare against.
- Every clip annotated by one annotator, spot-checked by a second on ≥10% of frames;
  disagreement rate reported (inter-annotator agreement).

## Metrics enabled

- Violation-flicker rate (state changes/minute) with vs. without TSMM.
- False alarms per minute vs. ground-truth violation intervals.
- Detection→confirmation latency (frames from true violation start to confirmed flag).

## Status log

- 2026-07-28: first two seed clips downloaded from Wikimedia Commons (CC BY-SA 4.0,
  1080p, ~6 min each; Haringey building-site series, 2025). Detector spot-check of
  worker visibility recorded in SOURCES.md.
