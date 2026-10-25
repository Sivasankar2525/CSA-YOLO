"""Temporal safety memory per worker (TSMM).

For each tracked worker and each PPE class, TSMM keeps a bounded window of recent
per-frame observations. Each observation is "present" if the associated PPE confidence
reaches ``confidence_threshold``, otherwise "absent". A worker with no associated PPE in
a frame contributes an "absent" observation for every class being tracked.

Cleanup: each state records ``last_seen_frame``. ``remove_old_tracks`` drops states whose
worker has not been updated for more than ``max_age`` frames. It is called by the
pipeline on every frame.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkerSafetyState:
    """Current temporal safety state for one worker."""
    track_id: int
    frame_count: int  # number of frames in which this worker was updated
    last_seen_frame: int = -1  # pipeline frame index of the most recent update
    ppe_history: dict[str, deque] = field(default_factory=dict)  # {"helmet": [True, False, True, ...]}
    confidence_history: dict[str, deque] = field(default_factory=dict)  # {"helmet": [0.95, 0.87, ...]}
    violation_duration: dict[str, int] = field(default_factory=dict)  # {"helmet": 5} = 5 consecutive absent frames

    def get_confirmed_status(self, ppe_class: str, threshold: float = 0.7) -> Optional[bool]:
        """Majority-vote view of the window: True if present ratio >= threshold, else False.

        Returns None only when no observation has been recorded for ``ppe_class``.
        Compliance decisions use the stricter three-way rule in ``scrm.compliance``.
        """
        if ppe_class not in self.ppe_history or not self.ppe_history[ppe_class]:
            return None

        history = list(self.ppe_history[ppe_class])
        present_ratio = sum(history) / len(history)
        return present_ratio >= threshold

    def evidence(self, ppe_class: str) -> tuple[int, float]:
        """Return (number of observations in window, present ratio). (0, 0.0) if none."""
        history = self.ppe_history.get(ppe_class)
        if not history:
            return 0, 0.0
        return len(history), sum(history) / len(history)

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
            "last_seen_frame": self.last_seen_frame,
            "ppe_present_ratio": {
                ppe_class: round(self.evidence(ppe_class)[1], 4)
                for ppe_class in self.ppe_history.keys()
            },
            "ppe_observations": {
                ppe_class: self.evidence(ppe_class)[0]
                for ppe_class in self.ppe_history.keys()
            },
            "ppe_avg_confidence": {
                ppe_class: round(conf, 4) if (conf := self.get_avg_confidence(ppe_class)) else None
                for ppe_class in self.confidence_history.keys()
            },
            "violation_duration": dict(self.violation_duration),
        }


class TemporalSafetyMemory:
    """Maintains a bounded temporal PPE history per tracked worker."""

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.5):
        """Initialize memory.

        Args:
            window_size: Number of most recent observations kept per worker and class.
            confidence_threshold: Min confidence to consider a detection as "present".
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.states: dict[int, WorkerSafetyState] = {}

    def update(
        self,
        track_id: int,
        ppe_detections: dict[str, float],  # {"helmet": 0.92, "vest": 0.87}
        frame_index: Optional[int] = None,
    ) -> WorkerSafetyState:
        """Record one frame's PPE evidence for a tracked worker.

        Args:
            track_id: Persistent worker ID.
            ppe_detections: {"ppe_class": confidence} for this frame. Classes not listed
                are recorded as absent if they were already tracked for this worker.
            frame_index: Pipeline frame index, used for stale-track cleanup. If omitted,
                the worker's own update count is used.

        Returns:
            Updated WorkerSafetyState.
        """
        if track_id not in self.states:
            self.states[track_id] = WorkerSafetyState(track_id=track_id, frame_count=0)

        state = self.states[track_id]
        state.frame_count += 1
        state.last_seen_frame = frame_index if frame_index is not None else state.frame_count

        for ppe_class in set(ppe_detections.keys()) | set(state.ppe_history.keys()):
            if ppe_class not in state.ppe_history:
                state.ppe_history[ppe_class] = deque(maxlen=self.window_size)
                state.confidence_history[ppe_class] = deque(maxlen=self.window_size)

            confidence = ppe_detections.get(ppe_class, 0.0)
            is_present = confidence >= self.confidence_threshold

            state.ppe_history[ppe_class].append(is_present)
            state.confidence_history[ppe_class].append(confidence)

            if not is_present:
                state.violation_duration[ppe_class] = state.violation_duration.get(ppe_class, 0) + 1
            else:
                state.violation_duration[ppe_class] = 0

        return state

    def get_state(self, track_id: int) -> Optional[WorkerSafetyState]:
        """Retrieve current state for a worker."""
        return self.states.get(track_id)

    def remove_old_tracks(self, current_frame: int, max_age: int = 50) -> None:
        """Remove states not updated for more than ``max_age`` frames.

        Args:
            current_frame: Current pipeline frame index.
            max_age: Frames a worker may go unseen before its memory is dropped.
        """
        stale = [
            track_id for track_id, state in self.states.items()
            if current_frame - state.last_seen_frame > max_age
        ]
        for track_id in stale:
            del self.states[track_id]

    def reset(self) -> None:
        """Clear all memory (new video)."""
        self.states.clear()
