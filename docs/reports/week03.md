# Week 3 report — Phase 1 close: unified dataset + CSA-Vid seed

**Research question:** can a unified multi-PPE taxonomy be assembled reproducibly from
public sources, and can CSA-Vid begin with properly licensed footage?

## Completed

- **CSS acquired without credentials**: found the Roboflow Construction Site Safety
  set mirrored on Hugging Face (398 imgs, COCO, 17 classes incl. no-hardhat /
  no-safety-vest); public direct download.
- **Unified dataset csa_ppe_v1 built**: `configs/merge.yaml` +
  `training/prepare_dataset.py` (COCO parser, class-mapping merge engine, deterministic
  splits, committed manifest). 1,924 images (1,516/213/195), 11,433 instances, 5 classes.
  CHV + CSS keep official splits; SHWD capped at 400 as auxiliary head-box data.
- **Critical audit finding**: `no_vest` = 38 instances (untrainable), `no_helmet` = 333
  (marginal). Design consequence recorded: SCRM's missing-vest evidence = reasoned
  absence over TSMM-confirmed frames; negative-class detections are corroboration only.
- **CSA-Vid seeded**: Wikimedia Commons search → 2 CC BY-SA 4.0 1080p clips downloaded;
  detector spot-check accepted clip 16 (2–4 close-range workers, occlusions, an
  apparently helmetless worker — ideal TSMM material), rejected clip 13 (no workers).
  Protocol doc (`docs/csa_vid_protocol.md`) + source log with licenses.
- 28 unit tests passing (merge mapping, YOLO writing, split determinism, COCO parsing).

## Next (Week 4 — Phase 2)

- Fine-tune ICSD-YOLO on csa_ppe_v1 (MPS throughput test → Colab/Kaggle fallback
  decision); baseline YOLO11n comparison; per-class metrics table.
