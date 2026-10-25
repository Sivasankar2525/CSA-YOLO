"""Evaluation metric functions for CSA-YOLO.

These are *definitions*, tested on synthetic inputs so the arithmetic is verified. They do
not contain any measured result. A metric is ``None`` (reported as NOT MEASURED) whenever
its denominator is zero or the ground truth it needs is absent.

Ground-truth formats are documented in ``docs/EVALUATION.md``. Every function here takes
plain Python structures so it can be used without the detector.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Mapping, Optional, Sequence

NON_COMPLIANT = "NON_COMPLIANT"
COMPLIANT = "COMPLIANT"
UNKNOWN = "UNKNOWN"


def _prf(tp: int, fp: int, fn: int) -> dict[str, Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def association_prf(
    predicted: Mapping[Hashable, Mapping[str, Optional[Hashable]]],
    truth: Mapping[Hashable, Mapping[str, Optional[Hashable]]],
) -> dict[str, object]:
    """Worker→PPE association precision / recall / F1 at the link level.

    Args:
        predicted: ``{image_id: {worker_id: {"helmet": ppe_id or None, "vest": ppe_id or None}}}``
        truth: same structure, from human annotation. ``None`` means "no PPE of that class".

    A link is a (image, worker, ppe_class, ppe_id) tuple. A predicted link is a true positive
    if it appears in the truth. A truth link not predicted is a false negative. Workers
    annotated with no PPE contribute no links, so they do not inflate precision.
    """
    def links(structure) -> set[tuple]:
        out: set[tuple] = set()
        for image_id, workers in structure.items():
            for worker_id, ppe in workers.items():
                for ppe_class, ppe_id in ppe.items():
                    if ppe_id is not None:
                        out.add((image_id, worker_id, ppe_class, ppe_id))
        return out

    pred_links, true_links = links(predicted), links(truth)
    tp = len(pred_links & true_links)
    result = _prf(tp, len(pred_links - true_links), len(true_links - pred_links))
    result["truth_links"] = len(true_links)
    result["predicted_links"] = len(pred_links)
    return result


def compliance_metrics(
    predicted: Mapping[Hashable, str],
    truth: Mapping[Hashable, str],
) -> dict[str, object]:
    """SCRM compliance metrics against human labels, with abstention reported separately.

    Args:
        predicted: ``{worker_key: COMPLIANT | NON_COMPLIANT | UNKNOWN}``
        truth: ``{worker_key: COMPLIANT | NON_COMPLIANT}`` (human labels)

    Accuracy, precision, recall and F1 use only workers the system decided
    (COMPLIANT or NON_COMPLIANT). UNKNOWN is counted in ``coverage``, not as an error.
    The positive class is NON_COMPLIANT.
    """
    keys = [k for k in truth if k in predicted]
    decided = [k for k in keys if predicted[k] in (COMPLIANT, NON_COMPLIANT)]
    tp = sum(1 for k in decided if predicted[k] == NON_COMPLIANT and truth[k] == NON_COMPLIANT)
    fp = sum(1 for k in decided if predicted[k] == NON_COMPLIANT and truth[k] == COMPLIANT)
    fn = sum(1 for k in decided if predicted[k] == COMPLIANT and truth[k] == NON_COMPLIANT)
    tn = sum(1 for k in decided if predicted[k] == COMPLIANT and truth[k] == COMPLIANT)
    prf = _prf(tp, fp, fn)
    n_decided = len(decided)
    accuracy = (tp + tn) / n_decided if n_decided else None
    return {
        "accuracy": accuracy,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "coverage": n_decided / len(keys) if keys else None,
        "unknown_count": sum(1 for k in keys if predicted[k] == UNKNOWN),
        "evaluated_workers": len(keys),
    }


def id_switches(frames: Sequence[Mapping[Hashable, Hashable]]) -> dict[str, object]:
    """Count identity switches for ground-truth workers.

    Args:
        frames: one mapping per frame, ``{gt_worker_id: predicted_track_id}``, in time order.
            A ground-truth worker absent from a frame is simply not in its mapping.

    A switch is counted when a worker's predicted track changes between two consecutive
    frames in which the worker is present. Gaps do not reset the comparison.
    """
    last: dict[Hashable, Hashable] = {}
    switches = Counter()
    observed = Counter()
    for frame in frames:
        for gt_id, track in frame.items():
            observed[gt_id] += 1
            if gt_id in last and last[gt_id] != track:
                switches[gt_id] += 1
            last[gt_id] = track
    total_observations = sum(observed.values())
    return {
        "id_switches_total": sum(switches.values()),
        "id_switches_per_worker": {str(k): v for k, v in sorted(switches.items(), key=str)},
        "gt_worker_observations": total_observations,
        "gt_workers": len(observed),
    }


def state_transitions(statuses: Sequence[str]) -> int:
    """Number of changes between consecutive statuses in a sequence."""
    return sum(1 for a, b in zip(statuses, statuses[1:]) if a != b)


def flicker_reduction(raw: Sequence[str], smoothed: Sequence[str]) -> dict[str, Optional[float]]:
    """Compare status transitions before and after temporal smoothing on the same sequence.

    ``reduction`` is ``(raw - smoothed) / raw``. It is None when the raw sequence has no
    transitions, because the ratio is undefined. A positive value means fewer transitions
    after smoothing on this sequence; it is evidence only for the sequence it was computed on.
    """
    if len(raw) != len(smoothed):
        raise ValueError("raw and smoothed sequences must be aligned")
    r, s = state_transitions(raw), state_transitions(smoothed)
    return {
        "raw_transitions": r,
        "smoothed_transitions": s,
        "reduction": (r - s) / r if r else None,
    }
