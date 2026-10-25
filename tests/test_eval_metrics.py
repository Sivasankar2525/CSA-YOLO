"""Metric definitions, verified on synthetic inputs (arithmetic only; no measured results)."""

import pytest

from evaluation.metrics import (
    COMPLIANT,
    NON_COMPLIANT,
    UNKNOWN,
    association_prf,
    compliance_metrics,
    flicker_reduction,
    id_switches,
    state_transitions,
)


# --- association -----------------------------------------------------------

def test_association_perfect_match() -> None:
    truth = {"img1": {1: {"helmet": "h1", "vest": "v1"}}}
    result = association_prf(truth, truth)
    assert (result["precision"], result["recall"], result["f1"]) == (1.0, 1.0, 1.0)


def test_association_counts_wrong_and_missing_links() -> None:
    truth = {"img1": {1: {"helmet": "h1", "vest": "v1"}, 2: {"helmet": "h2", "vest": None}}}
    predicted = {"img1": {1: {"helmet": "h1", "vest": None}, 2: {"helmet": "h1", "vest": None}}}
    result = association_prf(predicted, truth)
    assert result["tp"] == 1  # worker 1 helmet
    assert result["fp"] == 1  # worker 2 assigned to h1 (wrong item)
    assert result["fn"] == 2  # worker 1 vest missed, worker 2 helmet h2 missed
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1 / 3)


def test_association_with_no_predictions_has_undefined_precision() -> None:
    truth = {"img1": {1: {"helmet": "h1", "vest": None}}}
    result = association_prf({}, truth)
    assert result["precision"] is None
    assert result["recall"] == 0.0
    assert result["f1"] is None


def test_association_with_no_truth_has_undefined_recall() -> None:
    result = association_prf({"img1": {1: {"helmet": "h1", "vest": None}}}, {})
    assert result["recall"] is None
    assert result["precision"] == 0.0


def test_association_empty_inputs_are_undefined() -> None:
    result = association_prf({}, {})
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None


# --- compliance ------------------------------------------------------------

def test_compliance_accuracy_and_confusion() -> None:
    truth = {"a": COMPLIANT, "b": NON_COMPLIANT, "c": NON_COMPLIANT, "d": COMPLIANT}
    predicted = {"a": COMPLIANT, "b": NON_COMPLIANT, "c": COMPLIANT, "d": NON_COMPLIANT}
    result = compliance_metrics(predicted, truth)
    assert result["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_unknown_is_abstention_not_error() -> None:
    truth = {"a": NON_COMPLIANT, "b": COMPLIANT}
    predicted = {"a": UNKNOWN, "b": COMPLIANT}
    result = compliance_metrics(predicted, truth)
    assert result["accuracy"] == 1.0  # only b is decided, and it is correct
    assert result["coverage"] == pytest.approx(0.5)
    assert result["unknown_count"] == 1


def test_compliance_all_unknown_has_no_accuracy() -> None:
    result = compliance_metrics({"a": UNKNOWN}, {"a": NON_COMPLIANT})
    assert result["accuracy"] is None
    assert result["coverage"] == 0.0


def test_compliance_ignores_workers_without_ground_truth() -> None:
    result = compliance_metrics({"a": COMPLIANT, "z": NON_COMPLIANT}, {"a": COMPLIANT})
    assert result["evaluated_workers"] == 1


# --- identity switches -----------------------------------------------------

def test_stable_identity_has_no_switches() -> None:
    frames = [{"w1": 5}, {"w1": 5}, {"w1": 5}]
    assert id_switches(frames)["id_switches_total"] == 0


def test_identity_change_is_counted_once() -> None:
    frames = [{"w1": 5}, {"w1": 9}, {"w1": 9}]
    result = id_switches(frames)
    assert result["id_switches_total"] == 1
    assert result["id_switches_per_worker"] == {"w1": 1}


def test_gap_does_not_reset_identity_comparison() -> None:
    frames = [{"w1": 5}, {}, {}, {"w1": 9}]
    assert id_switches(frames)["id_switches_total"] == 1


def test_separate_workers_are_counted_separately() -> None:
    frames = [{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 2, "b": 3}]
    result = id_switches(frames)
    assert result["id_switches_per_worker"] == {"a": 1, "b": 1}
    assert result["gt_workers"] == 2


# --- flicker / smoothing ---------------------------------------------------

def test_state_transitions_counts_changes() -> None:
    assert state_transitions(["C", "N", "N", "C", "C"]) == 2
    assert state_transitions(["C"]) == 0
    assert state_transitions([]) == 0


def test_flicker_reduction_on_a_single_sequence() -> None:
    raw = ["C", "N", "C", "N", "C", "N"]     # 5 transitions
    smoothed = ["C", "C", "C", "N", "N", "N"]  # 1 transition (C -> N once)
    result = flicker_reduction(raw, smoothed)
    assert result["raw_transitions"] == 5
    assert result["smoothed_transitions"] == 1
    assert result["reduction"] == pytest.approx(4 / 5)


def test_flicker_reduction_undefined_without_raw_transitions() -> None:
    result = flicker_reduction(["C", "C"], ["C", "C"])
    assert result["reduction"] is None


def test_flicker_requires_aligned_sequences() -> None:
    with pytest.raises(ValueError):
        flicker_reduction(["C"], ["C", "N"])
