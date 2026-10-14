"""Simple IoU-based worker tracker for maintaining consistent IDs across frames."""

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
    frame_count: int = 0

    def update(self, detection: Detection, frame_idx: int) -> None:
        """Update with a new detection in current frame."""
        self.detections.append(detection)
        self.last_box = detection.xyxy
        self.frame_count = frame_idx


class SimpleIOUTracker:
    """Lightweight worker tracking using IoU matching.

    Matches workers across frames by finding nearest bounding box (highest IoU).
    Assigns new ID when no match found above threshold.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU to match a detection to existing track.
            max_age: Max frames a track can exist without a detection before deletion.
        """
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
            Mapping {track_id: Detection} for all active tracks in this frame.
        """
        self.frame_count += 1
        matched_tracks = {}
        used_detections = set()

        # Try to match existing tracks to new detections
        for track_id, track in list(self.tracks.items()):
            if track.last_box is None:
                continue

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
                # Match found
                detection = detections[best_det_idx]
                track.update(detection, self.frame_count)
                matched_tracks[track_id] = detection
                used_detections.add(best_det_idx)
            else:
                # No match; track ages out after max_age frames
                if self.frame_count - track.frame_count > self.max_age:
                    del self.tracks[track_id]

        # Assign new track IDs to unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = TrackedWorker(track_id=track_id)
                track.update(detection, self.frame_count)
                self.tracks[track_id] = track
                matched_tracks[track_id] = detection

        return matched_tracks

    def reset(self) -> None:
        """Reset tracker state (for new video)."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
