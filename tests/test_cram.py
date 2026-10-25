"""CRAM rule-based risk prototype tests (CPU-only, deterministic).

CRAM is a transparent PROTOTYPE rule model, not a trained or validated risk predictor.
These tests check that the coded rules behave as documented, not that they are correct
for real sites.
"""

import json
from collections import deque

import pytest

from csa_yolo.cram.risk_assessment import SITE_UNKNOWN, RiskAssessmentEngine
from csa_yolo.scrm.compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN, ComplianceResult
from csa_yolo.tsmm.memory import WorkerSafetyState


def make_state(
    helmet_conf: float = 0.9,
    vest_conf: float = 0.9,
    helmet_duration: int = 0,
    vest_duration: int = 0,
) -> WorkerSafetyState:
    state = WorkerSafetyState(track_id=1, frame_count=5)
    for ppe_class, conf in (("helmet", helmet_conf), ("vest", vest_conf)):
        state.ppe_history[ppe_class] = deque([conf >= 0.5] * 5, maxlen=5)
        state.confidence_history[ppe_class] = deque([conf] * 5, maxlen=5)
    state.violation_duration = {"helmet": helmet_duration, "vest": vest_duration}
    return state


def make_compliance(missing: list[str], status: str | None = None) -> ComplianceResult:
    status = status or (NON_COMPLIANT if missing else COMPLIANT)
    return ComplianceResult(
        track_id=1,
        status=status,
        confidence=1.0,
        missing_ppe=missing,
        required_ppe=["helmet", "vest"],
        reason="",
    )


def assess(missing, **state_kwargs):
    return RiskAssessmentEngine().assess_worker_risk(
        make_state(**state_kwargs), make_compliance(missing)
    )


# --- worker-level rules ----------------------------------------------------

def test_compliant_worker_has_zero_risk() -> None:
    result = assess([])
    assert result.worker_risk_score == 0.0
    assert result.worker_risk_level == "LOW"
    assert result.risk_factors == []


def test_missing_helmet_alone_scores_25_and_is_medium() -> None:
    result = assess(["helmet"])
    assert result.worker_risk_score == pytest.approx(25.0)
    assert result.worker_risk_level == "MEDIUM"  # boundary: 25 is not < 25
    assert "Missing helmet" in result.risk_factors


def test_missing_vest_alone_scores_15_and_is_low() -> None:
    result = assess(["vest"])
    assert result.worker_risk_score == pytest.approx(15.0)
    assert result.worker_risk_level == "LOW"


def test_missing_both_scores_40_and_is_medium() -> None:
    result = assess(["helmet", "vest"])
    assert result.worker_risk_score == pytest.approx(40.0)
    assert result.worker_risk_level == "MEDIUM"


def test_duration_penalty_starts_after_10_frames() -> None:
    assert assess(["helmet"], helmet_duration=10).worker_risk_score == pytest.approx(25.0)
    assert assess(["helmet"], helmet_duration=20).worker_risk_score == pytest.approx(45.0)


def test_long_helmet_violation_reaches_high() -> None:
    result = assess(["helmet"], helmet_duration=40)  # 25 + 40 = 65
    assert result.worker_risk_score == pytest.approx(65.0)
    assert result.worker_risk_level == "HIGH"


def test_high_boundary_at_60_is_high() -> None:
    result = assess(["helmet"], helmet_duration=35)  # 25 + 35 = 60
    assert result.worker_risk_score == pytest.approx(60.0)
    assert result.worker_risk_level == "HIGH"


def test_score_is_clamped_to_100() -> None:
    result = assess(["helmet", "vest"], helmet_duration=60, vest_duration=60)
    assert result.worker_risk_score == 100.0
    assert result.worker_risk_level == "HIGH"


def test_low_confidence_present_ppe_adds_penalty() -> None:
    result = assess([], helmet_conf=0.3)
    assert result.worker_risk_score == pytest.approx(5.0)
    assert any("Low confidence in helmet" in f for f in result.risk_factors)


def test_missing_ppe_is_not_double_counted_as_low_confidence() -> None:
    # Regression: the 0.0 confidence of a missing helmet must not add a second penalty.
    state = make_state(helmet_conf=0.0)
    result = RiskAssessmentEngine().assess_worker_risk(state, make_compliance(["helmet"]))
    assert result.worker_risk_score == pytest.approx(25.0)
    assert not any("Low confidence" in f for f in result.risk_factors)


def test_no_penalty_for_class_without_any_evidence() -> None:
    state = make_state()
    del state.ppe_history["helmet"]
    del state.confidence_history["helmet"]
    result = RiskAssessmentEngine().assess_worker_risk(state, make_compliance([]))
    assert result.worker_risk_score == 0.0


def test_level_boundary_at_25_is_configurable_via_weights() -> None:
    state, comp = make_state(), make_compliance(["helmet"])
    assert RiskAssessmentEngine(helmet_weight=24.9).assess_worker_risk(state, comp).worker_risk_level == "LOW"
    assert RiskAssessmentEngine(helmet_weight=25.0).assess_worker_risk(state, comp).worker_risk_level == "MEDIUM"


def test_unknown_workers_are_not_scored() -> None:
    # Regression for audit defect 6 (and the scoring rule): no evidenced violation, no score.
    with pytest.raises(ValueError):
        RiskAssessmentEngine().assess_worker_risk(make_state(), make_compliance([], status=UNKNOWN))


# --- site-level summary ----------------------------------------------------

def _site(statuses: dict[int, str], scored: dict[int, float] | None = None):
    compliance = {
        tid: make_compliance(["helmet"] if s == NON_COMPLIANT else [], status=s)
        for tid, s in statuses.items()
    }
    engine = RiskAssessmentEngine()
    risks = {
        tid: engine.assess_worker_risk(make_state(), compliance[tid])
        for tid in (scored if scored is not None else statuses)
        if statuses[tid] != UNKNOWN
    }
    return engine.assess_site_risk(compliance, risks)


def test_site_with_zero_workers_is_unknown_with_zero_counts() -> None:
    summary = RiskAssessmentEngine().assess_site_risk({}, {})
    assert summary.total_workers == 0
    assert summary.compliant_workers == 0
    assert summary.non_compliant_workers == 0
    assert summary.unknown_workers == 0
    assert summary.site_risk_level == SITE_UNKNOWN


def test_site_with_only_unknown_workers_is_unknown() -> None:
    summary = _site({1: UNKNOWN, 2: UNKNOWN})
    assert summary.unknown_workers == 2
    assert summary.site_risk_level == SITE_UNKNOWN
    assert summary.site_risk_score == 0.0


def test_compliance_categories_sum_to_total_workers() -> None:
    # Regression for audit defect 6: categories must be mutually exclusive and exhaustive.
    statuses = {1: COMPLIANT, 2: NON_COMPLIANT, 3: UNKNOWN, 4: NON_COMPLIANT, 5: COMPLIANT}
    summary = _site(statuses)
    assert summary.compliant_workers + summary.non_compliant_workers + summary.unknown_workers == summary.total_workers
    assert summary.total_workers == 5
    assert (summary.compliant_workers, summary.non_compliant_workers, summary.unknown_workers) == (2, 2, 1)


def test_compliant_workers_do_not_count_as_non_compliant() -> None:
    # Regression for audit defect 6: the old code counted every scored worker as non-compliant.
    summary = _site({1: COMPLIANT, 2: NON_COMPLIANT})
    assert summary.non_compliant_workers == 1
    assert summary.compliant_workers == 1


def test_site_score_is_mean_over_scored_workers_only() -> None:
    engine = RiskAssessmentEngine()
    compliance = {1: make_compliance([], COMPLIANT), 2: make_compliance(["helmet"], NON_COMPLIANT),
                  3: make_compliance([], UNKNOWN)}
    risks = {1: engine.assess_worker_risk(make_state(), compliance[1]),
             2: engine.assess_worker_risk(make_state(), compliance[2])}
    summary = engine.assess_site_risk(compliance, risks)
    assert summary.site_risk_score == pytest.approx(12.5)  # (0 + 25) / 2, UNKNOWN excluded
    assert summary.site_risk_level == "LOW"


def test_main_hazards_count_workers() -> None:
    engine = RiskAssessmentEngine()
    compliance = {i: make_compliance(["helmet"], NON_COMPLIANT) for i in (1, 2)}
    compliance[3] = make_compliance(["vest"], NON_COMPLIANT)
    risks = {tid: engine.assess_worker_risk(make_state(), c) for tid, c in compliance.items()}
    summary = engine.assess_site_risk(compliance, risks)
    assert summary.main_hazards[0] == "Missing helmet (2 workers)"


def test_to_dict_is_json_serialisable() -> None:
    json.dumps(assess(["helmet", "vest"], helmet_duration=15).to_dict())
    json.dumps(_site({1: NON_COMPLIANT, 2: UNKNOWN}).to_dict())
