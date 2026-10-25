"""Worker safety compliance reasoning (SCRM).

Decision rules (see docs/MODULES.md, "SCRM"). For each required PPE class, from the
worker's TSMM window of ``n`` observations with present ratio ``r``:

* ``n < min_evidence_frames``            -> class is UNKNOWN (too little evidence)
* ``r >= present_threshold``             -> class is PRESENT (positive evidence)
* ``r <= absent_threshold``              -> class is ABSENT  (evidence of absence)
* otherwise                              -> class is UNKNOWN (conflicting evidence)

Worker status:

* any required class ABSENT                -> NON_COMPLIANT (violation evidenced)
* every required class PRESENT             -> COMPLIANT     (positive evidence for all)
* otherwise                                -> UNKNOWN       (some class lacks evidence)

Absence is never inferred from a single frame or from a never-observed class: it needs
``min_evidence_frames`` observations and a ratio at or below ``absent_threshold``.
Limitation: a PPE item hidden by occlusion looks identical to a missing one, so an
occluded worker can be reported NON_COMPLIANT. Measuring this is future work.
"""

from dataclasses import dataclass, field
from typing import Optional

from csa_yolo.tsmm.memory import WorkerSafetyState

COMPLIANT = "COMPLIANT"
NON_COMPLIANT = "NON_COMPLIANT"
UNKNOWN = "UNKNOWN"

PRESENT = "PRESENT"
ABSENT = "ABSENT"
UNDECIDED = "UNDECIDED"


@dataclass
class ComplianceResult:
    """Compliance determination for one worker."""
    track_id: int
    status: str  # COMPLIANT, NON_COMPLIANT or UNKNOWN
    confidence: float  # [0, 1]: strength of the weakest required class's evidence
    missing_ppe: list[str]  # required classes with ABSENT evidence
    required_ppe: list[str]
    reason: str
    unknown_ppe: list[str] = field(default_factory=list)  # required classes with insufficient evidence

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "missing_ppe": self.missing_ppe,
            "unknown_ppe": self.unknown_ppe,
            "required_ppe": self.required_ppe,
            "reason": self.reason,
        }


class ComplianceEngine:
    """Determines worker compliance from temporal PPE evidence."""

    def __init__(
        self,
        required_ppe: Optional[list[str]] = None,
        confirmation_threshold: float = 0.7,
        absent_threshold: Optional[float] = None,
        min_evidence_frames: int = 5,
    ):
        """Initialize compliance checker.

        Args:
            required_ppe: Required PPE classes (default: ["helmet", "vest"]).
            confirmation_threshold: Present-ratio at or above which a class is PRESENT.
            absent_threshold: Present-ratio at or below which a class is ABSENT.
                Default: ``1 - confirmation_threshold``.
            min_evidence_frames: Observations needed before a class can be decided.
        """
        if absent_threshold is None:
            absent_threshold = round(1.0 - confirmation_threshold, 10)
        if not 0.0 <= absent_threshold < confirmation_threshold <= 1.0:
            raise ValueError(
                "require 0 <= absent_threshold < confirmation_threshold <= 1, got "
                f"{absent_threshold} and {confirmation_threshold}"
            )
        if min_evidence_frames < 1:
            raise ValueError(f"min_evidence_frames must be >= 1, got {min_evidence_frames}")
        self.required_ppe = list(required_ppe) if required_ppe else ["helmet", "vest"]
        self.confirmation_threshold = confirmation_threshold
        self.absent_threshold = absent_threshold
        self.min_evidence_frames = min_evidence_frames

    def classify_class(self, worker_state: WorkerSafetyState, ppe_class: str) -> tuple[str, float]:
        """Return (PRESENT | ABSENT | UNDECIDED, evidence strength in [0, 1])."""
        count, ratio = worker_state.evidence(ppe_class)
        if count < self.min_evidence_frames:
            return UNDECIDED, 0.0
        strength = max(ratio, 1.0 - ratio)
        if ratio >= self.confirmation_threshold:
            return PRESENT, strength
        if ratio <= self.absent_threshold:
            return ABSENT, strength
        return UNDECIDED, 0.0

    def check_compliance(self, worker_state: WorkerSafetyState) -> ComplianceResult:
        """Determine compliance status for one worker from its temporal state."""
        present, absent, undecided = [], [], []
        strengths = []
        for ppe_class in self.required_ppe:
            label, strength = self.classify_class(worker_state, ppe_class)
            strengths.append(strength)
            if label == PRESENT:
                present.append(ppe_class)
            elif label == ABSENT:
                absent.append(ppe_class)
            else:
                undecided.append(ppe_class)

        if absent:
            status = NON_COMPLIANT
            reason = f"Missing: {', '.join(absent)}."
            if undecided:
                reason += f" Unknown: {', '.join(undecided)}."
        elif undecided:
            status = UNKNOWN
            reason = f"Insufficient evidence for: {', '.join(undecided)}."
        else:
            status = COMPLIANT
            reason = f"All required PPE detected: {', '.join(present)}."

        return ComplianceResult(
            track_id=worker_state.track_id,
            status=status,
            confidence=min(strengths) if strengths else 0.0,
            missing_ppe=absent,
            required_ppe=list(self.required_ppe),
            reason=reason,
            unknown_ppe=undecided,
        )

    def check_compliance_batch(
        self,
        worker_states: dict[int, WorkerSafetyState],
    ) -> dict[int, ComplianceResult]:
        """Check compliance for multiple workers, keyed by track ID."""
        return {
            track_id: self.check_compliance(state)
            for track_id, state in worker_states.items()
        }
