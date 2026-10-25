"""IoU worker tracker tests (CPU-only, deterministic, no model)."""

import pytest

from csa_yolo.detection import Detection
from csa_yolo.tracking.tracker import SimpleIOUTracker

WORKER_A = (0.0, 0.0, 100.0, 100.0)
WORKER_B = (300.0, 0.0, 400.0, 100.0)


def det(box):
    return Detection(xyxy=box, confidence=0.9, class_id=0, class_name="person")


def shift(box, dx):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1, x2 + dx, y2)


def test_first_frame_assigns_sequential_ids() -> None:
    tracker = SimpleIOUTracker()
    ids = tracker.update([det(WORKER_A), det(WORKER_B)])
    assert sorted(ids) == [0, 1]


def test_same_worker_keeps_id_across_frames() -> None:
    tracker = SimpleIOUTracker()
    first = tracker.update([det(WORKER_A)])
    second = tracker.update([det(shift(WORKER_A, 5.0))])
    assert list(first) == list(second) == [0]


def test_multiple_workers_keep_unique_ids_across_frames() -> None:
    tracker = SimpleIOUTracker()
    tracker.update([det(WORKER_A), det(WORKER_B)])
    ids = tracker.update([det(shift(WORKER_B, 3.0)), det(shift(WORKER_A, 3.0))])
    assert sorted(ids) == [0, 1]
    assert ids[1].xyxy == shift(WORKER_B, 3.0)  # track 1 is WORKER_B


def test_new_worker_gets_new_id() -> None:
    tracker = SimpleIOUTracker()
    tracker.update([det(WORKER_A)])
    ids = tracker.update([det(WORKER_A), det((700.0, 0.0, 800.0, 100.0))])
    assert sorted(ids) == [0, 1]


def test_low_iou_starts_new_track() -> None:
    tracker = SimpleIOUTracker(iou_threshold=0.3)
    tracker.update([det(WORKER_A)])
    # Shifting 80px gives IoU ~0.11, below the 0.3 threshold.
    ids = tracker.update([det(shift(WORKER_A, 80.0))])
    assert list(ids) == [1]


def test_empty_frame_returns_no_active_detections() -> None:
    tracker = SimpleIOUTracker()
    tracker.update([det(WORKER_A)])
    assert tracker.update([]) == {}


def test_track_survives_short_occlusion() -> None:
    tracker = SimpleIOUTracker()
    tracker.update([det(WORKER_A)])
    for _ in range(5):
        tracker.update([])
    ids = tracker.update([det(WORKER_A)])
    assert list(ids) == [0]


def test_track_kept_for_max_age_frames_then_dropped() -> None:
    tracker = SimpleIOUTracker(max_age=30)
    tracker.update([det(WORKER_A)])
    for _ in range(30):
        tracker.update([])
    assert 0 in tracker.tracks  # 30 missed frames: still within max_age

    tracker.update([])  # 31st missed frame
    assert 0 not in tracker.tracks
    ids = tracker.update([det(WORKER_A)])
    assert list(ids) == [1]  # reappearance after expiry gets a new ID


def test_frame_counter_increments_each_update() -> None:
    tracker = SimpleIOUTracker()
    for _ in range(4):
        tracker.update([])
    assert tracker.frame_count == 4


def test_empty_frames_age_out_tracks() -> None:
    # Regression for audit defect 5: stale tracks must be removed on frames with no
    # workers too, which the pipeline now relies on.
    tracker = SimpleIOUTracker(max_age=30)
    tracker.update([det(WORKER_A)])
    for _ in range(31):
        tracker.update([])
    assert tracker.tracks == {}


def test_long_run_churn_keeps_track_table_bounded() -> None:
    tracker = SimpleIOUTracker(max_age=30)
    for frame in range(2000):
        tracker.update([det((frame * 500.0, 0.0, frame * 500.0 + 100.0, 100.0))])
        assert len(tracker.tracks) <= 31
    assert tracker.next_track_id == 2000


def test_active_worker_keeps_id_over_long_run_with_short_gaps() -> None:
    tracker = SimpleIOUTracker(max_age=30)
    first = tracker.update([det(WORKER_A)])
    track_id = next(iter(first))
    for frame in range(1, 1000):
        if frame % 20 == 0:
            tracker.update([])  # gaps shorter than max_age
        else:
            ids = tracker.update([det(shift(WORKER_A, 0.0))])
            assert list(ids) == [track_id]
    assert tracker.next_track_id == 1


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        SimpleIOUTracker(iou_threshold=1.5)
    with pytest.raises(ValueError):
        SimpleIOUTracker(max_age=-1)


def test_reset_clears_state() -> None:
    tracker = SimpleIOUTracker()
    tracker.update([det(WORKER_A)])
    tracker.reset()
    assert tracker.tracks == {}
    assert tracker.next_track_id == 0
    assert list(tracker.update([det(WORKER_A)])) == [0]
