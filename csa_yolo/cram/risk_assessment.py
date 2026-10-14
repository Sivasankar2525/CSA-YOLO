"""Construction risk assessment — rule-based prototype.

This is a TRANSPARENT PROTOTYPE risk model, NOT an ML predictor.
It grades individual workers and the overall site based on PPE violations.
"""

from dataclasses import dataclass

from csa_yolo.scrm.compliance import ComplianceResult
from csa_yolo.tsmm.memory import WorkerSafetyState


@dataclass
class RiskResult:
    """Risk assessment result."""
    track_id: int
    worker_risk_score: float  # 0-100
    worker_risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_factors: list[str]
    reason: str

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "track_id": self.track_id,
            "worker_risk_score": round(self.worker_risk_score, 2),
            "worker_risk_level": self.worker_risk_level,
            "risk_factors": self.risk_factors,
            "reason": self.reason,
        }


@dataclass
class SiteRiskSummary:
    """Aggregated site-level risk."""
    total_workers: int
    compliant_workers: int
    non_compliant_workers: int
    site_risk_score: float  # 0-100
    site_risk_level: str
    main_hazards: list[str]

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "total_workers": self.total_workers,
            "compliant_workers": self.compliant_workers,
            "non_compliant_workers": self.non_compliant_workers,
            "site_risk_score": round(self.site_risk_score, 2),
            "site_risk_level": self.site_risk_level,
            "main_hazards": self.main_hazards,
        }


class RiskAssessmentEngine:
    """Transparent risk scoring based on PPE violations.

    DISCLAIMER: This is a PROTOTYPE rule-based model for demonstration.
    It is NOT OSHA-certified, legally binding, or suitable for real safety decisions.
    Use only for research and educational purposes.
    """

    def __init__(
        self,
        helmet_weight: float = 25.0,
        vest_weight: float = 15.0,
        violation_duration_weight: float = 10.0,
        confidence_penalty: float = 5.0,
    ):
        """Initialize risk engine.

        Args:
            helmet_weight: Risk points for missing helmet.
            vest_weight: Risk points for missing vest.
            violation_duration_weight: Additional points per 10 frames of violation.
            confidence_penalty: Penalty for low-confidence detections.
        """
        self.helmet_weight = helmet_weight
        self.vest_weight = vest_weight
        self.violation_duration_weight = violation_duration_weight
        self.confidence_penalty = confidence_penalty

    def assess_worker_risk(
        self,
        worker_state: WorkerSafetyState,
        compliance_result: ComplianceResult,
    ) -> RiskResult:
        """Assess risk for one worker.

        Args:
            worker_state: Temporal state from TSMM.
            compliance_result: Compliance determination from SCRM.

        Returns:
            RiskResult with score, level, and factors.
        """
        score = 0.0
        factors = []

        # Base risk from missing PPE
        if "helmet" in compliance_result.missing_ppe:
            score += self.helmet_weight
            factors.append("Missing helmet")

            # Add duration penalty
            duration = worker_state.violation_duration.get("helmet", 0)
            if duration > 10:
                duration_penalty = (duration / 10) * self.violation_duration_weight
                score += duration_penalty
                factors.append(f"Helmet missing for {duration} frames")

        if "vest" in compliance_result.missing_ppe:
            score += self.vest_weight
            factors.append("Missing safety vest")

            duration = worker_state.violation_duration.get("vest", 0)
            if duration > 10:
                duration_penalty = (duration / 10) * self.violation_duration_weight
                score += duration_penalty
                factors.append(f"Vest missing for {duration} frames")

        # Confidence penalty
        for ppe_class in compliance_result.required_ppe:
            avg_conf = worker_state.get_avg_confidence(ppe_class) or 0.0
            if avg_conf < 0.5:
                score += self.confidence_penalty
                factors.append(f"Low confidence in {ppe_class} detection")

        # Clamp to 0-100
        score = min(100.0, max(0.0, score))

        # Determine risk level
        if score < 25:
            risk_level = "LOW"
        elif score < 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        reason = f"Score {score:.1f}: {'; '.join(factors) if factors else 'No hazards detected'}"

        return RiskResult(
            track_id=worker_state.track_id,
            worker_risk_score=score,
            worker_risk_level=risk_level,
            risk_factors=factors,
            reason=reason,
        )

    def assess_site_risk(
        self,
        worker_risks: dict[int, RiskResult],
        total_workers: int,
    ) -> SiteRiskSummary:
        """Aggregate site-level risk from all workers.

        Args:
            worker_risks: Dict {track_id: RiskResult}.
            total_workers: Total workers on site.

        Returns:
            SiteRiskSummary with aggregate metrics.
        """
        if not worker_risks:
            return SiteRiskSummary(
                total_workers=total_workers,
                compliant_workers=total_workers,
                non_compliant_workers=0,
                site_risk_score=0.0,
                site_risk_level="LOW",
                main_hazards=[],
            )

        # Count risk levels
        high_risk = sum(1 for r in worker_risks.values() if r.worker_risk_level == "HIGH")
        medium_risk = sum(1 for r in worker_risks.values() if r.worker_risk_level == "MEDIUM")

        # Average score across all active workers
        avg_score = sum(r.worker_risk_score for r in worker_risks.values()) / len(worker_risks)

        # Aggregate hazards
        all_factors = []
        for risk in worker_risks.values():
            all_factors.extend(risk.risk_factors)

        # Count frequency
        factor_count = {}
        for factor in all_factors:
            # Extract base factor (remove frame counts)
            base = factor.split(" for ")[0]
            factor_count[base] = factor_count.get(base, 0) + 1

        # Top hazards
        main_hazards = sorted(
            [(f, c) for f, c in factor_count.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        hazard_strs = [f"{name} ({count} workers)" for name, count in main_hazards]

        # Site risk level
        if avg_score < 20:
            site_risk_level = "LOW"
        elif avg_score < 50:
            site_risk_level = "MEDIUM"
        else:
            site_risk_level = "HIGH"

        return SiteRiskSummary(
            total_workers=total_workers,
            compliant_workers=total_workers - len(worker_risks),
            non_compliant_workers=len(worker_risks),
            site_risk_score=avg_score,
            site_risk_level=site_risk_level,
            main_hazards=hazard_strs,
        )
