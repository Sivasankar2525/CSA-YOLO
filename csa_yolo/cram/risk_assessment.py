"""Construction risk assessment — rule-based prototype.

This is a TRANSPARENT PROTOTYPE risk model, NOT an ML predictor and NOT a validated
risk model. It grades individual workers and the overall site from evidenced PPE
violations using fixed, documented weights. No ground-truth risk labels exist yet, so
its outputs are not experimentally validated.

Scoring is applied only to workers whose compliance status is COMPLIANT or
NON_COMPLIANT. An UNKNOWN worker has no evidenced violation and is therefore not scored;
the pipeline does not call ``assess_worker_risk`` for UNKNOWN workers.
"""

from dataclasses import dataclass, field

from csa_yolo.scrm.compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN, ComplianceResult
from csa_yolo.tsmm.memory import WorkerSafetyState

SITE_UNKNOWN = "UNKNOWN"  # no worker has a scored risk (includes zero workers)


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
    """Site-level summary. Compliance counts are mutually exclusive:

    ``compliant_workers + non_compliant_workers + unknown_workers == total_workers``.
    """
    total_workers: int
    compliant_workers: int
    non_compliant_workers: int
    unknown_workers: int
    site_risk_score: float  # mean score over scored workers, 0-100
    site_risk_level: str  # LOW, MEDIUM, HIGH, or SITE_UNKNOWN when no worker is scored
    main_hazards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "total_workers": self.total_workers,
            "compliant_workers": self.compliant_workers,
            "non_compliant_workers": self.non_compliant_workers,
            "unknown_workers": self.unknown_workers,
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
            violation_duration_weight: Additional points per 10 frames of violation
                beyond the first 10 frames.
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
        """Assess risk for one scored worker (COMPLIANT or NON_COMPLIANT).

        Raises:
            ValueError: If the compliance status is UNKNOWN (not scored by design).
        """
        if compliance_result.status == UNKNOWN:
            raise ValueError("UNKNOWN workers are not scored; check compliance status first")

        score = 0.0
        factors = []

        if "helmet" in compliance_result.missing_ppe:
            score += self.helmet_weight
            factors.append("Missing helmet")
            duration = worker_state.violation_duration.get("helmet", 0)
            if duration > 10:
                score += (duration / 10) * self.violation_duration_weight
                factors.append(f"Helmet missing for {duration} frames")

        if "vest" in compliance_result.missing_ppe:
            score += self.vest_weight
            factors.append("Missing safety vest")
            duration = worker_state.violation_duration.get("vest", 0)
            if duration > 10:
                score += (duration / 10) * self.violation_duration_weight
                factors.append(f"Vest missing for {duration} frames")

        # Low-confidence penalty applies only to PPE that is present but weakly detected.
        # A missing item already carries its own penalty; counting its 0.0 confidence
        # again would double-count the same violation.
        for ppe_class in compliance_result.required_ppe:
            if ppe_class in compliance_result.missing_ppe:
                continue
            count, _ = worker_state.evidence(ppe_class)
            if count == 0:
                continue
            avg_conf = worker_state.get_avg_confidence(ppe_class) or 0.0
            if avg_conf < 0.5:
                score += self.confidence_penalty
                factors.append(f"Low confidence in {ppe_class} detection")

        score = min(100.0, max(0.0, score))

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
        compliance_results: dict[int, ComplianceResult],
        worker_risks: dict[int, RiskResult],
    ) -> SiteRiskSummary:
        """Aggregate site-level summary.

        Compliance counts come from ``compliance_results``, one entry per observed worker,
        so the three categories are mutually exclusive and sum to ``total_workers``.
        The score and hazards come from ``worker_risks``, which should contain only the
        scored (COMPLIANT / NON_COMPLIANT) workers.

        Args:
            compliance_results: {track_id: ComplianceResult} for every observed worker.
            worker_risks: {track_id: RiskResult} for every scored worker.
        """
        statuses = [r.status for r in compliance_results.values()]
        compliant = statuses.count(COMPLIANT)
        non_compliant = statuses.count(NON_COMPLIANT)
        unknown = statuses.count(UNKNOWN)
        total = len(compliance_results)
        assert compliant + non_compliant + unknown == total, "unexpected compliance status"

        scored = list(worker_risks.values())
        if not scored:
            return SiteRiskSummary(
                total_workers=total,
                compliant_workers=compliant,
                non_compliant_workers=non_compliant,
                unknown_workers=unknown,
                site_risk_score=0.0,
                site_risk_level=SITE_UNKNOWN,
            )

        avg_score = sum(r.worker_risk_score for r in scored) / len(scored)

        factor_count: dict[str, int] = {}
        for risk in scored:
            for factor in set(f.split(" for ")[0] for f in risk.risk_factors):
                factor_count[factor] = factor_count.get(factor, 0) + 1
        top = sorted(factor_count.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        hazard_strs = [f"{name} ({count} workers)" for name, count in top]

        if avg_score < 20:
            site_risk_level = "LOW"
        elif avg_score < 50:
            site_risk_level = "MEDIUM"
        else:
            site_risk_level = "HIGH"

        return SiteRiskSummary(
            total_workers=total,
            compliant_workers=compliant,
            non_compliant_workers=non_compliant,
            unknown_workers=unknown,
            site_risk_score=avg_score,
            site_risk_level=site_risk_level,
            main_hazards=hazard_strs,
        )
