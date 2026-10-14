"""Worker-PPE geometric association using region matching."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from csa_yolo.detection import Detection


@dataclass
class WorkerPPEAssociation:
    """Association result for one worker."""
    worker_id: int
    worker_box: tuple[float, float, float, float]  # xyxy
    worker_confidence: float
    associated_ppe: dict[str, list[Detection]]  # {"helmet": [...], "vest": [...]}

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
    """Extract head region (top 20% of worker box, expanded upward)."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    head_bottom = y1 + height * 0.25  # top 20-25%
    expansion = height * 0.15
    return (x1 - expansion, max(0, y1 - expansion), x2 + expansion, head_bottom)


def get_torso_region(worker_box: tuple) -> tuple:
    """Extract torso region (middle 60% of worker box)."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    torso_top = y1 + height * 0.20
    torso_bottom = y1 + height * 0.75
    return (x1, torso_top, x2, torso_bottom)


def associate_ppe_to_workers(
    workers: Sequence[Detection],
    ppe_detections: Sequence[Detection],
    helmet_classes: list[str] = None,
    vest_classes: list[str] = None,
) -> list[WorkerPPEAssociation]:
    """Associate PPE items to workers using region-based matching.

    Args:
        workers: Person/worker detections.
        ppe_detections: All PPE detections (helmet, vest, etc).
        helmet_classes: Class names for helmets (default: ["helmet", "hat", "hardhat"])
        vest_classes: Class names for vests (default: ["vest", "safety_vest"])

    Returns:
        List of WorkerPPEAssociation, one per worker.
    """
    if helmet_classes is None:
        helmet_classes = ["helmet", "hat", "hardhat"]
    if vest_classes is None:
        vest_classes = ["vest", "safety_vest"]

    associations = []

    for worker_id, worker in enumerate(workers):
        assoc = WorkerPPEAssociation(
            worker_id=worker_id,
            worker_box=worker.xyxy,
            worker_confidence=worker.confidence,
            associated_ppe={},
        )

        head_region = get_head_region(worker.xyxy)
        torso_region = get_torso_region(worker.xyxy)

        # Match helmets to head region
        helmet_detections = [
            d for d in ppe_detections
            if d.class_name.lower() in [c.lower() for c in helmet_classes]
        ]
        if helmet_detections:
            matched_helmets = _match_ppe_to_region(
                helmet_detections, head_region
            )
            assoc.associated_ppe["helmet"] = matched_helmets

        # Match vests to torso region
        vest_detections = [
            d for d in ppe_detections
            if d.class_name.lower() in [c.lower() for c in vest_classes]
        ]
        if vest_detections:
            matched_vests = _match_ppe_to_region(
                vest_detections, torso_region
            )
            assoc.associated_ppe["vest"] = matched_vests

        associations.append(assoc)

    return associations


def _match_ppe_to_region(ppe_items: list[Detection], region: tuple) -> list[Detection]:
    """Match PPE items to a specific body region using Hungarian algorithm.

    Args:
        ppe_items: Detections to match.
        region: Body region box (xyxy).

    Returns:
        PPE items whose center is in or closest to the region.
    """
    if not ppe_items:
        return []

    # Cost matrix: negative IoU + distance penalty
    costs = []
    for ppe in ppe_items:
        overlap_iou = iou(ppe.xyxy, region)
        ppe_cx, ppe_cy = ppe.center
        reg_x1, reg_y1, reg_x2, reg_y2 = region
        reg_cx = (reg_x1 + reg_x2) / 2
        reg_cy = (reg_y1 + reg_y2) / 2

        # Distance from PPE center to region center (normalized by region size)
        dx = abs(ppe_cx - reg_cx)
        dy = abs(ppe_cy - reg_cy)
        region_w = reg_x2 - reg_x1
        region_h = reg_y2 - reg_y1
        norm_dist = (dx / max(region_w, 1)) + (dy / max(region_h, 1))

        # Cost: prefer high IoU and low distance
        cost = -overlap_iou + 0.1 * norm_dist
        costs.append(cost)

    # For now, select PPE with lowest cost (highest IoU + closest distance)
    # In future, could add multi-assignment handling
    best_idx = np.argmin(costs)
    return [ppe_items[best_idx]] if costs else []
