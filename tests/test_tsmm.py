"""TSMM temporal safety memory tests (CPU-only, deterministic)."""

import json

import pytest

from csa_yolo.tsmm.memory import TemporalSafetyMemory


def feed(mem: TemporalSafetyMemory, track_id: int, confidences: list[dict]):
    state = None
    for ppe in confidences:
        state = mem.update(track_id, ppe)
    return state


def helmet_seq(*confs: float) -> list[dict]:
    return [{"helmet": c} for c in confs]


def test_new_worker_creates_state() -> None:
    mem = TemporalSafetyMemory()
    state = mem.update(7, {"helmet": 0.9})
    assert state.track_id == 7
    assert state.frame_count == 1
    assert list(state.ppe_history["helmet"]) == [True]


def test_sliding_window_is_bounded() -> None:
    mem = TemporalSafetyMemory(window_size=5)
    state = feed(mem, 1, helmet_seq(*([0.9] * 8)))
    assert len(state.ppe_history["helmet"]) == 5


def test_majority_vote_confirms_presence_at_70_percent() -> None:
    mem = TemporalSafetyMemory(window_size=5)
    state = feed(mem, 1, helmet_seq(0.9, 0.9, 0.9, 0.9, 0.1))  # 4/5 = 0.8
    assert state.get_confirmed_status("helmet", threshold=0.7) is True


def test_majority_vote_rejects_presence_below_70_percent() -> None:
    mem = TemporalSafetyMemory(window_size=5)
    state = feed(mem, 1, helmet_seq(0.9, 0.9, 0.9, 0.1, 0.1))  # 3/5 = 0.6
    assert state.get_confirmed_status("helmet", threshold=0.7) is False


def test_exact_threshold_counts_as_present() -> None:
    mem = TemporalSafetyMemory(window_size=10)
    state = feed(mem, 1, helmet_seq(*([0.9] * 7 + [0.1] * 3)))  # 7/10 = 0.7
    assert state.get_confirmed_status("helmet", threshold=0.7) is True


def test_transient_missing_frame_is_smoothed_out() -> None:
    mem = TemporalSafetyMemory(window_size=5)
    state = feed(mem, 1, helmet_seq(0.9, 0.9, 0.0, 0.9, 0.9))  # one dropped frame
    assert state.get_confirmed_status("helmet") is True


def test_confidence_below_presence_threshold_counts_as_absent() -> None:
    mem = TemporalSafetyMemory(confidence_threshold=0.5)
    state = mem.update(1, {"helmet": 0.49})
    assert list(state.ppe_history["helmet"]) == [False]


def test_class_missing_from_frame_is_recorded_as_absent_once_seen() -> None:
    mem = TemporalSafetyMemory()
    mem.update(1, {"helmet": 0.9})
    state = mem.update(1, {})
    assert list(state.ppe_history["helmet"]) == [True, False]


def test_violation_duration_counts_consecutive_absence_and_resets() -> None:
    mem = TemporalSafetyMemory()
    feed(mem, 1, helmet_seq(0.9))
    state = feed(mem, 1, helmet_seq(0.0, 0.0, 0.0))
    assert state.violation_duration["helmet"] == 3
    state = mem.update(1, {"helmet": 0.9})
    assert state.violation_duration["helmet"] == 0


def test_never_observed_class_is_unknown() -> None:
    mem = TemporalSafetyMemory()
    state = mem.update(1, {"helmet": 0.9})
    assert state.get_confirmed_status("vest") is None


def test_workers_have_independent_memory() -> None:
    mem = TemporalSafetyMemory()
    feed(mem, 1, helmet_seq(0.9, 0.9))
    other = feed(mem, 2, helmet_seq(0.0, 0.0))
    assert other.get_confirmed_status("helmet") is False
    assert mem.get_state(1).get_confirmed_status("helmet") is True


def test_average_confidence_over_window() -> None:
    mem = TemporalSafetyMemory()
    state = feed(mem, 1, helmet_seq(0.8, 0.6))
    assert state.get_avg_confidence("helmet") == pytest.approx(0.7)
    assert state.get_avg_confidence("vest") is None


def test_to_dict_is_json_serialisable() -> None:
    mem = TemporalSafetyMemory()
    state = feed(mem, 1, [{"helmet": 0.9, "vest": 0.8}, {"helmet": 0.0}])
    json.dumps(state.to_dict())


def test_remove_old_tracks_drops_stale_worker() -> None:
    mem = TemporalSafetyMemory()
    mem.update(1, {"helmet": 0.9})
    mem.remove_old_tracks(current_frame=100, max_age=50)
    assert mem.get_state(1) is None


def test_remove_old_tracks_keeps_recently_seen_worker() -> None:
    # Regression for audit defect 5: a worker first seen late in a long video was dropped.
    mem = TemporalSafetyMemory()
    for frame in (98, 99, 100):
        mem.update(5, {"helmet": 0.9}, frame_index=frame)
    mem.remove_old_tracks(current_frame=100, max_age=50)
    assert mem.get_state(5) is not None


def test_remove_old_tracks_boundary_is_strictly_greater_than_max_age() -> None:
    mem = TemporalSafetyMemory()
    mem.update(1, {"helmet": 0.9}, frame_index=50)
    mem.remove_old_tracks(current_frame=100, max_age=50)  # 50 frames unseen: kept
    assert mem.get_state(1) is not None
    mem.remove_old_tracks(current_frame=101, max_age=50)  # 51 frames unseen: removed
    assert mem.get_state(1) is None


def test_long_run_cleanup_keeps_active_and_drops_departed_workers() -> None:
    mem = TemporalSafetyMemory()
    mem.update("departed", {"helmet": 0.9}, frame_index=1)
    for frame in range(2, 1001):
        mem.update("active", {"helmet": 0.9}, frame_index=frame)
        mem.remove_old_tracks(current_frame=frame, max_age=50)
        assert len(mem.states) <= 2
    assert mem.get_state("active") is not None
    assert mem.get_state("departed") is None


def test_memory_size_is_bounded_under_churn() -> None:
    # A new worker every frame: memory must stay near max_age, not grow with time.
    mem = TemporalSafetyMemory()
    for frame in range(1, 2001):
        mem.update(frame, {"helmet": 0.9}, frame_index=frame)
        mem.remove_old_tracks(current_frame=frame, max_age=50)
    assert len(mem.states) <= 51


def test_reset_clears_memory() -> None:
    mem = TemporalSafetyMemory()
    mem.update(1, {"helmet": 0.9})
    mem.reset()
    assert mem.get_state(1) is None
