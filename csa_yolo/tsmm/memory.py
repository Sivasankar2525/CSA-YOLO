"""Temporal safety memory per worker."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkerSafetyState:
    """Current temporal safety state for one worker."""
    track_id: int
    frame_count: int  # frames in which this worker was active
    ppe_history: dict[str, deque] = field(default_factory=dict)  # {"helmet": [True, False, True, ...]}
    confidence_history: dict[str, deque] = field(default_factory=dict)  # {"helmet": [0.95, 0.87, ...]}
    violation_duration: dict[str, int] = field(default_factory=dict)  # {"helmet": 5} = 5 frames missing

    def get_confirmed_status(self, ppe_class: str, threshold: float = 0.7) -> Optional[bool]:
        """Get confirmed PPE status via majority voting.

        Args:
            ppe_class: PPE class name (e.g., "helmet", "vest").
            threshold: Ratio of frames where PPE must be present to consider "confirmed present".

        Returns:
            True if confirmed present, False if confirmed absent, None if unknown.
        """
        if ppe_class not in self.ppe_history or not self.ppe_history[ppe_class]:
            return None

        history = list(self.ppe_history[ppe_class])
        if not history:
            return None

        present_ratio = sum(history) / len(history)
        if present_ratio >= threshold:
            return True
        else:
            return False

    def get_avg_confidence(self, ppe_class: str) -> Optional[float]:
        """Get average detection confidence for PPE class."""
        if ppe_class not in self.confidence_history or not self.confidence_history[ppe_class]:
            return None
        history = list(self.confidence_history[ppe_class])
        return sum(history) / len(history) if history else None

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "track_id": self.track_id,
            "frame_count": self.frame_count,
            "ppe_confirmed": {
                ppe_class: self.get_confirmed_status(ppe_class)
                for ppe_class in self.ppe_history.keys()
            },
            "ppe_avg_confidence": {
                ppe_class: round(conf, 4) if (conf := self.get_avg_confidence(ppe_class)) else None
                for ppe_class in self.confidence_history.keys()
            },
            "violation_duration": {
                k: v for k, v in self.violation_duration.items()
            },
        }


class TemporalSafetyMemory:
    """Maintains temporal PPE history per tracked worker.

    Reduces false alarms by smoothing per-frame detections over a sliding window.
    """

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.5):
        """Initialize memory.

        Args:
            window_size: Number of past frames to remember per worker.
            confidence_threshold: Min confidence to consider detection as "present".
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.states: dict[int, WorkerSafetyState] = {}

    def update(
        self,
        track_id: int,
        ppe_detections: dict[str, float],  # {"helmet": 0.92, "vest": 0.87}
    ) -> WorkerSafetyState:
        """Update memory for a tracked worker.

        Args:
            track_id: Persistent worker ID.
            ppe_detections: Dict of {"ppe_class": confidence} for this frame.

        Returns:
            Updated WorkerSafetyState.
        """
        if track_id not in self.states:
            self.states[track_id] = WorkerSafetyState(track_id=track_id, frame_count=0)

        state = self.states[track_id]
        state.frame_count += 1

        # Update PPE history
        for ppe_class in set(ppe_detections.keys()) | set(state.ppe_history.keys()):
            if ppe_class not in state.ppe_history:
                state.ppe_history[ppe_class] = deque(maxlen=self.window_size)
                state.confidence_history[ppe_class] = deque(maxlen=self.window_size)

            confidence = ppe_detections.get(ppe_class, 0.0)
            is_present = confidence >= self.confidence_threshold

            state.ppe_history[ppe_class].append(is_present)
            state.confidence_history[ppe_class].append(confidence)

            # Track violation duration (consecutive frames missing PPE)
            if not is_present:
                state.violation_duration[ppe_class] = state.violation_duration.get(ppe_class, 0) + 1
            else:
                state.violation_duration[ppe_class] = 0

        return state

    def get_state(self, track_id: int) -> Optional[WorkerSafetyState]:
        """Retrieve current state for a worker."""
        return self.states.get(track_id)

    def remove_old_tracks(self, current_frame: int, max_age: int = 50) -> None:
        """Remove workers not seen recently."""
        to_delete = [
            track_id for track_id, state in self.states.items()
            if current_frame - state.frame_count > max_age
        ]
        for track_id in to_delete:
            del self.states[track_id]

    def reset(self) -> None:
        """Clear all memory (new video)."""
        self.states.clear()
