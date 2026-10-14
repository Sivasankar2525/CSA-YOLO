"""Worker safety compliance reasoning."""

from dataclasses import dataclass
from typing import Optional, Sequence

from csa_yolo.tsmm.memory import WorkerSafetyState


@dataclass
class ComplianceResult:
    """Compliance determination for one worker."""
    track_id: int
    status: str  # "COMPLIANT", "NON_COMPLIANT", "UNKNOWN"
    confidence: float  # [0, 1] how confident in the status
    missing_ppe: list[str]
    required_ppe: list[str]
    reason: str

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "missing_ppe": self.missing_ppe,
            "required_ppe": self.required_ppe,
            "reason": self.reason,
        }


class ComplianceEngine:
    """Determines worker compliance based on temporal PPE history.

    Configurable rules define required PPE per zone or globally.
    """

    def __init__(
        self,
        required_ppe: list[str] = None,
        confirmation_threshold: float = 0.7,
    ):
        """Initialize compliance checker.

        Args:
            required_ppe: List of required PPE classes (e.g., ["helmet", "vest"]).
                Default: ["helmet", "vest"].
            confirmation_threshold: Threshold for confirmed PPE presence (0-1).
        """
        self.required_ppe = required_ppe or ["helmet", "vest"]
        self.confirmation_threshold = confirmation_threshold

    def check_compliance(self, worker_state: WorkerSafetyState) -> ComplianceResult:
        """Determine if a worker is compliant based on temporal state.

        Args:
            worker_state: WorkerSafetyState from TSMM.

        Returns:
            ComplianceResult with status, missing PPE, and reason.
        """
        if not worker_state.ppe_history:
            return ComplianceResult(
                track_id=worker_state.track_id,
                status="UNKNOWN",
                confidence=0.0,
                missing_ppe=self.required_ppe.copy(),
                required_ppe=self.required_ppe.copy(),
                reason="No PPE detections recorded.",
            )

        missing = []
        unknown = []
        confirmed_present = []
        confidence_sum = 0.0
        count = 0

        for ppe_class in self.required_ppe:
            confirmed = worker_state.get_confirmed_status(ppe_class, self.confirmation_threshold)
            avg_conf = worker_state.get_avg_confidence(ppe_class) or 0.0

            if confirmed is None:
                unknown.append(ppe_class)
            elif confirmed:
                confirmed_present.append(ppe_class)
                confidence_sum += avg_conf
                count += 1
            else:
                missing.append(ppe_class)
                confidence_sum += 1.0 - avg_conf
                count += 1

        # Determine overall status
        if missing:
            status = "NON_COMPLIANT"
            reason = f"Missing: {', '.join(missing)}."
            if unknown:
                reason += f" Unknown: {', '.join(unknown)}."
        elif unknown:
            status = "UNKNOWN"
            reason = f"Uncertain about: {', '.join(unknown)}."
        else:
            status = "COMPLIANT"
            reason = f"All required PPE detected: {', '.join(confirmed_present)}."

        confidence = confidence_sum / max(count, 1)

        return ComplianceResult(
            track_id=worker_state.track_id,
            status=status,
            confidence=confidence,
            missing_ppe=missing,
            required_ppe=self.required_ppe.copy(),
            reason=reason,
        )

    def check_compliance_batch(
        self,
        worker_states: dict[int, WorkerSafetyState],
    ) -> dict[int, ComplianceResult]:
        """Check compliance for multiple workers.

        Args:
            worker_states: Dict {track_id: WorkerSafetyState}.

        Returns:
            Dict {track_id: ComplianceResult}.
        """
        return {
            track_id: self.check_compliance(state)
            for track_id, state in worker_states.items()
        }
