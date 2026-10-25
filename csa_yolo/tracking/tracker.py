"""Simple IoU-based worker tracker for maintaining consistent IDs across frames.

Track lifecycle (see docs/MODULES.md, "Tracking"):

* **Match.** Each live track is matched to at most one detection in the current frame:
  the unused detection with the highest IoU against the track's last box, provided that
  IoU is at least ``iou_threshold``. Tracks are visited in ID order (greedy matching).
* **Create.** Each unmatched detection starts a new track with the next integer ID.
  IDs are never reused within a tracker instance, so an ID identifies one continuous
  track, not a physical person across long gaps.
* **Prune.** A track that has gone unmatched for more than ``max_age`` consecutive frames
  is removed. Pruning runs on every ``update`` call, including frames with no detections,
  so memory is bounded by the number of workers seen in the last ``max_age`` frames.

Identity assumption: a worker's box overlaps its box in the previous frame. Fast motion,
occlusion longer than ``max_age`` frames, or two workers swapping positions will break
identity. These failure modes are not yet measured (see docs/PROJECT_STATUS.md).
"""

from dataclasses import dataclass, field
from typing import Sequence

from csa_yolo.detection import Detection
from csa_yolo.wpam.associate import iou


@dataclass
class TrackedWorker:
    """A worker with persistent identity across frames."""
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    last_box: tuple[float, float, float, float] = None
    frame_count: int = 0  # frame index at which this track was last matched

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Update with a new detection in current frame."""
        self.detections.append(detection)
        self.last_box = detection.xyxy
        self.frame_count = frame_idx


class SimpleIOUTracker:
    """Lightweight worker tracking using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU to match a detection to existing track.
            max_age: A track is removed after more than ``max_age`` consecutive
                frames without a match.
        """
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {max_age}")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, TrackedWorker] = {}
        self.next_track_id = 0
        self.frame_count = 0

    def update(self, detections: Sequence[Detection]) -> dict[int, Detection]:
        """Update tracker with new worker detections.

        Args:
            detections: Worker detections in current frame.

        Returns:
            Mapping {track_id: Detection} for all tracks matched or created in this frame.
        """
        self.frame_count += 1
        matched_tracks = {}
        used_detections = set()

        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            for det_idx, detection in enumerate(detections):
                if det_idx in used_detections:
                    continue
                overlap = iou(track.last_box, detection.xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                detection = detections[best_det_idx]
                track.update(detection, self.frame_count)
                matched_tracks[track_id] = detection
                used_detections.add(best_det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = TrackedWorker(track_id=track_id)
                track.update(detection, self.frame_count)
                self.tracks[track_id] = track
                matched_tracks[track_id] = detection

        self.prune_stale_tracks()
        return matched_tracks

    def prune_stale_tracks(self) -> None:
        """Remove tracks unmatched for more than ``max_age`` frames."""
        stale = [
            track_id for track_id, track in self.tracks.items()
            if self.frame_count - track.frame_count > self.max_age
        ]
        for track_id in stale:
            del self.tracks[track_id]

    def reset(self) -> None:
        """Reset tracker state (for new video)."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
