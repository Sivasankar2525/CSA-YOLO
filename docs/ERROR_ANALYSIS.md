# Error analysis framework

Status: **taxonomy and reporting rules defined. No error has been counted.**
EXPERIMENT REQUIRED: counts come only from measured runs and human-verified cases.

## Rules
1. An error is recorded only when a human has checked it against the image or frame. An automatic
   disagreement with the detector is a *candidate* until verified.
2. Every recorded error carries: run name, source image or frame, object or worker identifier,
   the category below, and a short note.
3. Counts are reported per category with the total number of cases examined. A rate is reported
   only with its denominator.
4. Categories are not mutually exclusive by default. A case may carry several categories, and the
   primary category is listed first.

## Categories

### Detection
| Category | Definition |
|---|---|
| false positive | a box with no matching object in the ground truth |
| false negative | a labelled object with no detection above the confidence threshold |
| small object | GT box shorter or narrower than [threshold to be fixed before measuring] pixels |
| occlusion | object partly hidden by another object or by the frame edge |
| lighting | visibility degraded by very dark or saturated frames |
| crowded scene | several workers whose boxes overlap substantially |

### WPAM
| Category | Definition |
|---|---|
| incorrect association | PPE linked to the wrong worker |
| overlapping workers | PPE lies in the gate of two workers; assignment chose one |
| ambiguous PPE | PPE box lies inside the gate but its owner is not visually clear |

### Tracking
| Category | Definition |
|---|---|
| ID switch | the same worker reported under a different track id |
| lost track | a visible worker has no track id for a frame range |
| re-identification issue | a worker re-appears after a gap and receives a new id |

### TSMM
| Category | Definition |
|---|---|
| delayed transition | the reported state changes several frames after the true change |
| unstable state | the reported state alternates while the true state is constant |
| persistent wrong state | the reported state stays wrong for longer than the window |

### SCRM
| Category | Definition |
|---|---|
| incorrect compliance state | a decided (COMPLIANT / NON_COMPLIANT) state disagrees with the human label |
| insufficient evidence | the state is UNKNOWN where a human label exists (coverage loss, not an error in the decision) |
| ambiguous evidence | the human label itself is uncertain; the case is excluded from accuracy and listed separately |

### CRAM (rule-based prototype)
| Category | Definition |
|---|---|
| incorrect rule outcome | the rule output does not match the qualitative case expectation |
| ambiguous risk | the case is not clearly low, medium or high even for a human reviewer |

## Output (to be generated from measured artifacts)
`results/error_analysis.csv` with one row per verified case: run, source, object id, primary
category, all categories, note. The generator is **not yet written**. Until then this file does not exist.
