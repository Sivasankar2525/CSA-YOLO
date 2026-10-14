"""Temporal Safety Memory Module (TSMM).

Maintains per-worker temporal history of PPE status to reduce false alarms
from occlusion or transient detections. Uses sliding-window majority voting
and confidence averaging.
"""

from .memory import TemporalSafetyMemory, WorkerSafetyState

__all__ = ["TemporalSafetyMemory", "WorkerSafetyState"]
