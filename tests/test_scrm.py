"""SCRM compliance tests: COMPLIANT / NON_COMPLIANT / UNKNOWN (CPU-only, deterministic)."""

import json

import pytest

from csa_yolo.scrm.compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN, ComplianceEngine
from csa_yolo.tsmm.memory import TemporalSafetyMemory, WorkerSafetyState


def build_state(helmet: list[float | None], vest: list[float | None], window: int = 5):
    """Feed per-frame confidences through TSMM. None = class not reported that frame."""
    mem = TemporalSafetyMemory(window_size=window)
    state = None
    for h, v in zip(helmet, vest):
        ppe = {}
        if h is not None:
            ppe["helmet"] = h
        if v is not None:
            ppe["vest"] = v
        state = mem.update(1, ppe)
    return state


def check(state, **engine_kwargs):
    return ComplianceEngine(**engine_kwargs).check_compliance(state)


# --- COMPLIANT: sufficient positive evidence for every required class -----

def test_compliant_with_positive_evidence_for_all_required_ppe() -> None:
    result = check(build_state([0.9] * 5, [0.9] * 5))
    assert result.status == COMPLIANT
    assert result.missing_ppe == []
    assert result.unknown_ppe == []
    assert result.confidence == pytest.approx(1.0)  # all frames agree


def test_compliant_requires_presence_at_threshold_not_just_one_frame() -> None:
    # 4/5 present (0.8 >= 0.7) is enough; 3/5 (0.6) is not decisive either way.
    assert check(build_state([0.9, 0.9, 0.9, 0.9, 0.0], [0.9] * 5)).status == COMPLIANT
    assert check(build_state([0.9, 0.9, 0.9, 0.0, 0.0], [0.9] * 5)).status == UNKNOWN


# --- NON_COMPLIANT: sufficient evidence that required PPE is absent --------

def test_absent_helmet_with_enough_evidence_is_non_compliant() -> None:
    result = check(build_state([0.0] * 5, [0.9] * 5))
    assert result.status == NON_COMPLIANT
    assert result.missing_ppe == ["helmet"]


def test_absent_vest_with_enough_evidence_is_non_compliant() -> None:
    result = check(build_state([0.9] * 5, [0.0] * 5))
    assert result.status == NON_COMPLIANT
    assert result.missing_ppe == ["vest"]


def test_multiple_absent_classes_are_all_reported() -> None:
    result = check(build_state([0.0] * 5, [0.0] * 5))
    assert result.status == NON_COMPLIANT
    assert result.missing_ppe == ["helmet", "vest"]


def test_absence_from_too_few_frames_is_not_a_violation() -> None:
    # Regression for audit defect 4: one missing frame must not produce NON_COMPLIANT.
    result = check(build_state([0.0, 0.0], [0.9, 0.9]))
    assert result.status == UNKNOWN
    assert result.missing_ppe == []


def test_violation_is_reported_even_when_other_class_is_undecided() -> None:
    result = check(build_state([0.0] * 5, [None] * 5))
    assert result.status == NON_COMPLIANT
    assert result.missing_ppe == ["helmet"]
    assert result.unknown_ppe == ["vest"]
    assert "Unknown: vest" in result.reason


# --- UNKNOWN: insufficient or conflicting evidence ------------------------

def test_no_observations_is_unknown_with_zero_confidence() -> None:
    empty = WorkerSafetyState(track_id=1, frame_count=0)
    result = check(empty)
    assert result.status == UNKNOWN
    assert result.confidence == 0.0
    assert result.unknown_ppe == ["helmet", "vest"]


def test_never_observed_class_is_unknown_not_absent() -> None:
    # Regression for audit defect 4: a class never detected is UNKNOWN, not absent.
    result = check(build_state([0.9] * 5, [None] * 5))
    assert result.status == UNKNOWN
    assert result.missing_ppe == []
    assert result.unknown_ppe == ["vest"]


def test_conflicting_evidence_is_unknown() -> None:
    # 2 of 5 present = 0.4: neither >= 0.7 nor <= 0.3.
    result = check(build_state([0.9, 0.9, 0.0, 0.0, 0.0], [0.9] * 5))
    assert result.status == UNKNOWN
    assert result.unknown_ppe == ["helmet"]


def test_unknown_with_too_few_frames_even_if_all_absent() -> None:
    result = check(build_state([0.0] * 4, [0.9] * 4), min_evidence_frames=5)
    assert result.status == UNKNOWN


# --- configuration and boundaries -----------------------------------------

def test_required_ppe_is_configurable() -> None:
    result = check(build_state([0.9] * 5, [0.0] * 5), required_ppe=["helmet"])
    assert result.status == COMPLIANT
    assert result.required_ppe == ["helmet"]


def test_present_threshold_is_respected() -> None:
    state = build_state([0.9, 0.9, 0.9, 0.9, 0.0], [0.9] * 5)  # helmet 0.8
    assert check(state, confirmation_threshold=0.7).status == COMPLIANT
    # With present >= 0.9 and absent <= 0.1, a 0.8 ratio is conflicting: UNKNOWN.
    assert check(state, confirmation_threshold=0.9).status == UNKNOWN


def test_absent_threshold_defaults_to_complement_of_present() -> None:
    engine = ComplianceEngine(confirmation_threshold=0.8)
    assert engine.absent_threshold == pytest.approx(0.2)


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        ComplianceEngine(confirmation_threshold=0.7, absent_threshold=0.8)
    with pytest.raises(ValueError):
        ComplianceEngine(min_evidence_frames=0)


def test_confidence_is_weakest_class_strength() -> None:
    result = check(build_state([0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.0]))
    # vest ratio 0.8 -> strength 0.8; helmet ratio 1.0 -> strength 1.0; min = 0.8
    assert result.confidence == pytest.approx(0.8)
    assert 0.0 <= result.confidence <= 1.0


def test_reason_is_human_readable() -> None:
    result = check(build_state([0.9] * 5, [0.9] * 5))
    assert result.reason == "All required PPE detected: helmet, vest."


def test_batch_check_is_keyed_by_track_id() -> None:
    states = {1: build_state([0.9] * 5, [0.9] * 5), 2: build_state([0.0] * 5, [0.9] * 5)}
    results = ComplianceEngine().check_compliance_batch(states)
    assert results[1].status == COMPLIANT
    assert results[2].status == NON_COMPLIANT


def test_to_dict_is_json_serialisable() -> None:
    result = check(build_state([0.0] * 5, [0.9] * 5))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["status"] == NON_COMPLIANT
    assert payload["unknown_ppe"] == []
