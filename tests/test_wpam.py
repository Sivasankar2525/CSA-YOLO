"""WPAM worker-PPE association tests (CPU-only, deterministic, no model).

Covers spatial gating, one-to-one (Hungarian) assignment, configuration, and
regression tests for the three defects found in the pre-training audit.
"""

import itertools
import json

import numpy as np
import pytest

from csa_yolo.detection import Detection
from csa_yolo.wpam.associate import (
    WPAMConfig,
    associate_ppe_to_workers,
    candidate_costs,
    get_head_region,
    get_torso_region,
    match_one_to_one,
)


def det(box, name, conf=0.9, class_id=0):
    return Detection(xyxy=box, confidence=conf, class_id=class_id, class_name=name)


# A 100x100 worker: head region ~ y in [0, 25], torso region ~ y in [20, 75].
WORKER = (0.0, 0.0, 100.0, 100.0)
HELMET_IN_HEAD = (35.0, 5.0, 65.0, 20.0)
VEST_IN_TORSO = (10.0, 40.0, 90.0, 70.0)


# --- region geometry -------------------------------------------------------

def test_region_geometry_matches_documented_fractions() -> None:
    _, _, _, head_bottom = get_head_region(WORKER)
    assert head_bottom == pytest.approx(25.0)  # top 25% of a 100px box
    _, torso_top, _, torso_bottom = get_torso_region(WORKER)
    assert (torso_top, torso_bottom) == (pytest.approx(20.0), pytest.approx(75.0))


# --- basic association -----------------------------------------------------

def test_helmet_in_head_region_is_associated() -> None:
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(HELMET_IN_HEAD, "helmet")])
    assert [d.xyxy for d in assoc.associated_ppe["helmet"]] == [HELMET_IN_HEAD]
    assert "vest" not in assoc.associated_ppe


def test_vest_in_torso_region_is_associated() -> None:
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(VEST_IN_TORSO, "vest")])
    assert [d.xyxy for d in assoc.associated_ppe["vest"]] == [VEST_IN_TORSO]
    assert "helmet" not in assoc.associated_ppe


def test_worker_with_helmet_and_vest() -> None:
    ppe = [det(HELMET_IN_HEAD, "helmet"), det(VEST_IN_TORSO, "vest")]
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], ppe)
    assert set(assoc.associated_ppe) == {"helmet", "vest"}


def test_no_ppe_gives_empty_association() -> None:
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [])
    assert assoc.associated_ppe == {}


def test_no_workers_gives_empty_list() -> None:
    assert associate_ppe_to_workers([], [det(HELMET_IN_HEAD, "helmet")]) == []


def test_unrelated_class_is_ignored() -> None:
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(HELMET_IN_HEAD, "boot")])
    assert assoc.associated_ppe == {}


def test_class_name_matching_is_case_insensitive() -> None:
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(HELMET_IN_HEAD, "Helmet")])
    assert "helmet" in assoc.associated_ppe


def test_one_association_per_worker_in_input_order() -> None:
    workers = [det(WORKER, "person"), det((300.0, 0.0, 400.0, 100.0), "person")]
    assocs = associate_ppe_to_workers(workers, [])
    assert [a.worker_id for a in assocs] == [0, 1]
    assert assocs[1].worker_box == (300.0, 0.0, 400.0, 100.0)


def test_association_is_deterministic() -> None:
    workers = [det(WORKER, "person")]
    ppe = [det(HELMET_IN_HEAD, "helmet"), det((40.0, 2.0, 70.0, 18.0), "helmet")]
    first = [a.to_dict() for a in associate_ppe_to_workers(workers, ppe)]
    second = [a.to_dict() for a in associate_ppe_to_workers(workers, ppe)]
    assert first == second


def test_to_dict_is_json_serialisable() -> None:
    ppe = [det(HELMET_IN_HEAD, "helmet"), det(VEST_IN_TORSO, "vest")]
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], ppe)
    json.dumps(assoc.to_dict())


# --- (a) valid nearby association / (b) far-away rejection ----------------

def test_nearby_helmet_is_accepted() -> None:
    nearby = (30.0, -4.0, 60.0, 8.0)  # just above the worker's top edge
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(nearby, "helmet")])
    assert "helmet" in assoc.associated_ppe


def test_far_away_helmet_is_rejected() -> None:
    # Regression for audit defect 1: no distance gate existed.
    helmet_on_floor = (35.0, 80.0, 65.0, 95.0)
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(helmet_on_floor, "helmet")])
    assert "helmet" not in assoc.associated_ppe


def test_helmet_of_distant_worker_is_rejected() -> None:
    workers = [det(WORKER, "person"), det((300.0, 0.0, 400.0, 100.0), "person")]
    assocs = associate_ppe_to_workers(workers, [det((335.0, 5.0, 365.0, 20.0), "helmet")])
    assert "helmet" not in assocs[0].associated_ppe
    assert assocs[1].associated_ppe["helmet"][0].xyxy == (335.0, 5.0, 365.0, 20.0)


def test_gate_margin_is_configurable() -> None:
    # Helmet sits 15px above the worker's top edge (outside the head region).
    just_outside = (35.0, -25.0, 65.0, -10.0)
    strict = WPAMConfig(gate_margin_ratio=0.0)
    loose = WPAMConfig(gate_margin_ratio=0.5)  # margin 50px for a 100px worker
    [a_strict] = associate_ppe_to_workers([det(WORKER, "person")], [det(just_outside, "helmet")], config=strict)
    [a_loose] = associate_ppe_to_workers([det(WORKER, "person")], [det(just_outside, "helmet")], config=loose)
    assert "helmet" not in a_strict.associated_ppe
    assert "helmet" in a_loose.associated_ppe


def test_negative_gate_margin_is_rejected() -> None:
    with pytest.raises(ValueError):
        WPAMConfig(gate_margin_ratio=-0.1)


# --- (c) multiple candidate PPE objects -----------------------------------

def test_multiple_candidates_choose_closest_to_region_centre() -> None:
    centred = (35.0, 5.0, 65.0, 20.0)     # centre (50, 12.5) = head-region centre
    off_centre = (5.0, 5.0, 25.0, 15.0)   # still inside the gate, further from centre
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(off_centre, "helmet"), det(centred, "helmet")])
    assert [d.xyxy for d in assoc.associated_ppe["helmet"]] == [centred]


def test_at_most_one_helmet_per_worker() -> None:
    ppe = [det(HELMET_IN_HEAD, "helmet"), det((40.0, 2.0, 70.0, 18.0), "helmet")]
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], ppe)
    assert len(assoc.associated_ppe["helmet"]) == 1


# --- (d) no valid candidate ------------------------------------------------

def test_no_valid_candidate_leaves_worker_without_ppe() -> None:
    far_vest = (400.0, 400.0, 450.0, 450.0)
    [assoc] = associate_ppe_to_workers([det(WORKER, "person")], [det(far_vest, "vest")])
    assert assoc.associated_ppe == {}


# --- one-worker-per-PPE (audit defect 2) -----------------------------------

def test_single_helmet_cannot_belong_to_two_overlapping_workers() -> None:
    workers = [det(WORKER, "person"), det((60.0, 0.0, 160.0, 100.0), "person")]
    assocs = associate_ppe_to_workers(workers, [det((40.0, 5.0, 80.0, 20.0), "helmet")])
    owners = [a.worker_id for a in assocs if "helmet" in a.associated_ppe]
    assert len(owners) <= 1


def test_each_worker_gets_its_own_helmet() -> None:
    workers = [det(WORKER, "person"), det((300.0, 0.0, 400.0, 100.0), "person")]
    helmets = [det((35.0, 5.0, 65.0, 20.0), "helmet"), det((335.0, 5.0, 365.0, 20.0), "helmet")]
    assocs = associate_ppe_to_workers(workers, helmets)
    assigned = [a.associated_ppe["helmet"][0].xyxy for a in assocs]
    assert assigned == [helmets[0].xyxy, helmets[1].xyxy]


# --- Hungarian matching: executed and optimal -----------------------------

def _brute_force_best(costs: np.ndarray) -> tuple[int, float]:
    """Max feasible cardinality, then min total cost, over all injective assignments."""
    if costs.shape[0] > costs.shape[1]:
        costs = costs.T  # enumerate over the smaller side
    n, m = costs.shape
    best = (0, 0.0)
    for cols in itertools.permutations(range(m), n):
        pairs = [(i, c) for i, c in enumerate(cols) if costs[i, c] < 1e6]
        count = len(pairs)
        total = sum(costs[i, c] for i, c in pairs)
        if count > best[0] or (count == best[0] and total < best[1]):
            best = (count, total)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_hungarian_matching_is_optimal_against_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    costs = rng.uniform(0.0, 2.0, size=(n, m))
    costs[rng.uniform(size=(n, m)) < 0.4] = 1e6  # ungated pairs
    pairs = match_one_to_one(costs)
    count = len(pairs)
    total = sum(costs[i, j] for i, j in pairs)
    best_count, best_total = _brute_force_best(costs)
    assert count == best_count
    assert total == pytest.approx(best_total)


@pytest.mark.parametrize("seed", range(20))
def test_matching_is_one_to_one(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    costs = rng.uniform(0.0, 1.0, size=(4, 4))
    pairs = match_one_to_one(costs)
    assert len({i for i, _ in pairs}) == len(pairs)
    assert len({j for _, j in pairs}) == len(pairs)


def test_matching_on_empty_matrix() -> None:
    assert match_one_to_one(np.zeros((0, 3))) == []
    assert match_one_to_one(np.zeros((2, 0))) == []


def test_candidate_costs_mark_ungated_pairs_as_infeasible() -> None:
    costs = candidate_costs(
        [det(WORKER, "person")],
        [det(HELMET_IN_HEAD, "helmet"), det((400.0, 400.0, 420.0, 420.0), "helmet")],
        get_head_region,
        WPAMConfig(),
    )
    assert costs[0, 0] < 1e6
    assert costs[0, 1] >= 1e6
