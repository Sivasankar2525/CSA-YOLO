# CSA-Vid source log

Every clip: origin, license, attribution, acceptance decision per
`docs/csa_vid_protocol.md` (criteria: ≥1 annotatable worker, occlusion events valued).

| Clip | Source | License | Duration/specs | Decision | Notes |
|---|---|---|---|---|---|
| `haringey_16.webm` | [Wikimedia Commons: Building construction Moira Close Broadwater Farm Haringey 2025 16](https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm) | CC BY-SA 4.0 | 362 s, 1080p30 | **ACCEPT** | Close-range concrete pour; 2–4 workers; vests + helmets visible incl. an apparently helmetless worker; natural occlusions (pole, worker crossings). Detector spot-check: 2–3 persons in sampled frames (`results/csa_vid_check/`) |
| `haringey_13.webm` (removed) | same series, file 13 | CC BY-SA 4.0 | 393 s, 1080p30 | REJECT | No detectable workers in 6 sampled frames — fails criterion 3 |

Attribution (CC BY-SA 4.0): video by the respective Wikimedia Commons uploader
(see file page history); derivative annotation data will be shared under a
compatible license.

Target remains 10–20 clips (protocol doc); collection continues alongside Phase 2.
