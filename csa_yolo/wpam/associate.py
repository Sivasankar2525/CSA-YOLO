"""Worker-PPE association (WPAM).

Assignment strategy (per PPE class group, helmet and vest independently):

1. **Spatial gate.** A PPE item is a *candidate* for a worker only if its centre lies
   inside that worker's body region (head region for helmets, torso region for vests),
   expanded by ``WPAMConfig.gate_margin_ratio`` times the worker's box height.
   Candidates outside the gate are never assigned, however well the rest of the
   scene fits.
2. **One-to-one matching.** Candidate pairs are solved as a global assignment with
   the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) over the
   worker x PPE cost matrix. Each worker gets at most one helmet and at most one vest;
   each PPE item goes to at most one worker. Infeasible (ungated) pairs carry a
   sentinel cost and are discarded after solving.
3. **Cost.** For a candidate pair, the cost is the PPE centre's distance to the body
   region centre, normalised by the region size (scale-invariant). Lower is better.

Unknown handling: WPAM only reports matched PPE. A worker with no matched item of a
class gets no entry for that class in this frame. WPAM does not decide compliance;
frame-level absence is turned into evidence by TSMM, and UNKNOWN is decided by SCRM.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from csa_yolo.detection import Detection

# Solver sentinel for ungated (infeasible) worker/PPE pairs. Not a research parameter:
# it only needs to exceed every feasible normalised distance.
_INFEASIBLE_COST = 1e6


@dataclass(frozen=True)
class WPAMConfig:
    """Parameters of the worker-PPE association.

    Attributes:
        gate_margin_ratio: Margin added around a body region before a PPE centre is
            accepted, as a fraction of the worker box height. **PROVISIONAL**: this
            is an engineering default, not a tuned or validated value. It must be
            set on a validation split before any claim about association accuracy.
    """

    gate_margin_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.gate_margin_ratio < 0.0:
            raise ValueError(f"gate_margin_ratio must be >= 0, got {self.gate_margin_ratio}")


@dataclass
class WorkerPPEAssociation:
    """Association result for one worker."""
    worker_id: int
    worker_box: tuple[float, float, float, float]  # xyxy
    worker_confidence: float
    associated_ppe: dict[str, list[Detection]] = field(default_factory=dict)  # {"helmet": [...], "vest": [...]}

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "worker_id": self.worker_id,
            "worker_box": [round(v, 2) for v in self.worker_box],
            "worker_confidence": round(self.worker_confidence, 4),
            "associated_ppe": {
                class_name: [d.to_dict() for d in detections]
                for class_name, detections in self.associated_ppe.items()
            },
        }


def iou(box1: tuple, box2: tuple) -> float:
    """Calculate IoU between two boxes (xyxy format)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
        return 0.0

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def get_head_region(worker_box: tuple) -> tuple:
    """Head region: top 25% of the worker box, expanded 15% of its height on each side."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    head_bottom = y1 + height * 0.25
    expansion = height * 0.15
    return (x1 - expansion, max(0, y1 - expansion), x2 + expansion, head_bottom)


def get_torso_region(worker_box: tuple) -> tuple:
    """Torso region: from 20% to 75% of the worker box height."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    torso_top = y1 + height * 0.20
    torso_bottom = y1 + height * 0.75
    return (x1, torso_top, x2, torso_bottom)


def _expand(box: tuple, margin: float) -> tuple:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _contains(box: tuple, point: tuple) -> bool:
    x1, y1, x2, y2 = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def _normalized_distance(point: tuple, region: tuple) -> float:
    """Distance from a point to a region centre, normalised by region size."""
    x1, y1, x2, y2 = region
    rw, rh = x2 - x1, y2 - y1
    dx = abs(point[0] - (x1 + x2) / 2)
    dy = abs(point[1] - (y1 + y2) / 2)
    return dx / max(rw, 1.0) + dy / max(rh, 1.0)


def candidate_costs(
    workers: Sequence[Detection],
    ppe_items: Sequence[Detection],
    region_fn: Callable[[tuple], tuple],
    config: WPAMConfig,
) -> np.ndarray:
    """Worker x PPE cost matrix: normalised distance for gated pairs, sentinel otherwise."""
    costs = np.full((len(workers), len(ppe_items)), _INFEASIBLE_COST)
    for i, worker in enumerate(workers):
        region = region_fn(worker.xyxy)
        worker_height = worker.xyxy[3] - worker.xyxy[1]
        gate = _expand(region, config.gate_margin_ratio * worker_height)
        for j, ppe in enumerate(ppe_items):
            if _contains(gate, ppe.center):
                costs[i, j] = _normalized_distance(ppe.center, region)
    return costs


def match_one_to_one(costs: np.ndarray) -> list[tuple[int, int]]:
    """Solve the feasible one-to-one assignment with the Hungarian algorithm.

    Returns (worker_index, ppe_index) pairs. Pairs at the sentinel cost are dropped.
    """
    if costs.size == 0:
        return []
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if costs[r, c] < _INFEASIBLE_COST]


def associate_ppe_to_workers(
    workers: Sequence[Detection],
    ppe_detections: Sequence[Detection],
    helmet_classes: Optional[list[str]] = None,
    vest_classes: Optional[list[str]] = None,
    config: Optional[WPAMConfig] = None,
) -> list[WorkerPPEAssociation]:
    """Associate PPE items to workers (spatial gate + one-to-one Hungarian matching).

    Args:
        workers: Person/worker detections. Output order follows this sequence.
        ppe_detections: All PPE detections (helmet, vest, etc).
        helmet_classes: Class names for helmets (default: ["helmet", "hat", "hardhat"]).
        vest_classes: Class names for vests (default: ["vest", "safety_vest"]).
        config: Association parameters (default: ``WPAMConfig()``).

    Returns:
        One WorkerPPEAssociation per worker. A PPE item appears under at most one worker.
    """
    if helmet_classes is None:
        helmet_classes = ["helmet", "hat", "hardhat"]
    if vest_classes is None:
        vest_classes = ["vest", "safety_vest"]
    config = config or WPAMConfig()

    associations = [
        WorkerPPEAssociation(worker_id=i, worker_box=w.xyxy, worker_confidence=w.confidence)
        for i, w in enumerate(workers)
    ]
    if not workers:
        return associations

    groups = (
        ("helmet", helmet_classes, get_head_region),
        ("vest", vest_classes, get_torso_region),
    )
    lowered = {"helmet": {c.lower() for c in helmet_classes}, "vest": {c.lower() for c in vest_classes}}
    for ppe_name, _, region_fn in groups:
        candidates = [d for d in ppe_detections if d.class_name.lower() in lowered[ppe_name]]
        if not candidates:
            continue
        costs = candidate_costs(workers, candidates, region_fn, config)
        for worker_idx, ppe_idx in match_one_to_one(costs):
            associations[worker_idx].associated_ppe[ppe_name] = [candidates[ppe_idx]]

    return associations
