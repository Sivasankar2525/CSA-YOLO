"""Full-pipeline runner tests: edge cases with a scripted detector (no model, CPU-only)."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from csa_yolo.detection import Detection, FrameDetections
from csa_yolo.pipeline import SafetyPipeline
from csa_yolo.runner import FrameInput, open_source, run_source

WORKER = (0.0, 0.0, 100.0, 100.0)
HELMET = (35.0, 5.0, 65.0, 20.0)
VEST = (10.0, 40.0, 90.0, 70.0)
FRAME = np.full((100, 120, 3), 128, dtype=np.uint8)


def det(box, name, conf=0.9):
    return Detection(xyxy=box, confidence=conf, class_id={"person": 0, "helmet": 1, "vest": 2}.get(name, 9), class_name=name)


class ScriptedDetector:
    """Returns scripted detections per call; an entry that is an Exception is raised instead."""

    def __init__(self, script):
        self._script = list(script)
        self._calls = 0

    def detect(self, image, frame_index: int = 0):
        entry = self._script[self._calls] if self._calls < len(self._script) else []
        self._calls += 1
        if isinstance(entry, Exception):
            raise entry
        return FrameDetections(frame_index=self._calls, detections=tuple(entry))


def frames_of(arrays) -> list[FrameInput]:
    return [FrameInput(index=i, frame=f, timestamp_s=(i - 1) * 0.1) for i, f in enumerate(arrays, start=1)]


def run(script, arrays, tmp_path: Path, **kwargs):
    pipeline = SafetyPipeline(ScriptedDetector(script))
    return run_source(pipeline, frames_of(arrays), tmp_path / "out", source_name="test",
                      kind=kwargs.pop("kind", "video"), **kwargs), pipeline


WITH_PPE = [det(WORKER, "person"), det(HELMET, "helmet"), det(VEST, "vest")]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- normal operation -----------------------------------------------------

def test_outputs_are_written_and_summary_is_honest(tmp_path: Path) -> None:
    summary, _ = run([WITH_PPE] * 6, [FRAME] * 6, tmp_path)
    out = tmp_path / "out"
    assert (out / "frames.jsonl").exists() and (out / "workers.csv").exists()
    assert (out / "summary.json").exists() and (out / "annotated.mp4").exists()
    assert summary["frames_processed"] == 6
    assert summary["frames_seen"] == 6
    assert summary["compliance_status_counts"].get("COMPLIANT", 0) > 0
    assert "NOT EXPERIMENTALLY VALIDATED" in summary["validation_status"]


def test_one_jsonl_record_per_frame_with_source_index(tmp_path: Path) -> None:
    run([WITH_PPE] * 3, [FRAME] * 3, tmp_path)
    records = read_jsonl(tmp_path / "out" / "frames.jsonl")
    assert [r["frame_index"] for r in records] == [1, 2, 3]
    assert all(r["status"] == "ok" for r in records)
    assert records[0]["timestamp_s"] == pytest.approx(0.0)


def test_worker_csv_has_one_row_per_worker_per_frame(tmp_path: Path) -> None:
    run([WITH_PPE] * 2, [FRAME] * 2, tmp_path)
    rows = list(csv.DictReader((tmp_path / "out" / "workers.csv").open()))
    assert len(rows) == 2
    assert {r["track_id"] for r in rows} == {"0"}
    assert rows[0]["helmet_matched"] == "True"


# --- empty and unusual frames ---------------------------------------------

def test_zero_worker_frames_do_not_crash(tmp_path: Path) -> None:
    summary, _ = run([[]] * 3, [FRAME] * 3, tmp_path)
    assert summary["frames_processed"] == 3
    assert summary["frames_empty"] == 3
    assert summary["max_workers_in_frame"] == 0
    assert (tmp_path / "out" / "workers.csv").read_text() == ""


def test_worker_without_ppe_is_processed(tmp_path: Path) -> None:
    summary, _ = run([[det(WORKER, "person")]] * 6, [FRAME] * 6, tmp_path)
    assert summary["frames_processed"] == 6
    # Minimum-evidence rule (5 observations): frames 1-4 are UNKNOWN, frames 5-6 are NON_COMPLIANT.
    assert summary["compliance_status_counts"] == {"UNKNOWN": 4, "NON_COMPLIANT": 2}


def test_multiple_workers_summary(tmp_path: Path) -> None:
    two = WITH_PPE + [det((300.0, 0.0, 400.0, 100.0), "person")]
    summary, _ = run([two] * 2, [FRAME] * 2, tmp_path)
    assert summary["max_workers_in_frame"] == 2


# --- failures ---------------------------------------------------------------

def test_unreadable_frame_is_recorded_and_skipped(tmp_path: Path) -> None:
    summary, _ = run([WITH_PPE, WITH_PPE], [FRAME, None, FRAME], tmp_path)
    assert summary["frame_errors"] == 1
    assert summary["frames_processed"] == 2
    records = read_jsonl(tmp_path / "out" / "frames.jsonl")
    assert any(r.get("status") == "frame_error" for r in records)


def test_transient_detector_failure_is_retried(tmp_path: Path) -> None:
    summary, _ = run([RuntimeError("cuda hiccup"), WITH_PPE], [FRAME, FRAME], tmp_path)
    assert summary["detector_errors"] == 0
    assert summary["frames_processed"] == 2


def test_persistent_detector_failure_is_recorded_not_guessed(tmp_path: Path) -> None:
    script = [RuntimeError("boom"), RuntimeError("boom again"), WITH_PPE]
    summary, _ = run(script, [FRAME, FRAME], tmp_path)
    assert summary["detector_errors"] == 1
    records = read_jsonl(tmp_path / "out" / "frames.jsonl")
    failed = [r for r in records if r.get("status") == "detector_error"]
    assert len(failed) == 1 and "boom again" in failed[0]["error"]


def test_max_frames_limits_processing(tmp_path: Path) -> None:
    summary, _ = run([WITH_PPE] * 10, [FRAME] * 10, tmp_path, max_frames=4)
    assert summary["frames_seen"] == 4


def test_annotated_images_for_image_input(tmp_path: Path) -> None:
    run([WITH_PPE] * 2, [FRAME] * 2, tmp_path, kind="images")
    assert len(list((tmp_path / "out").glob("annotated_*.jpg"))) == 2


# --- input handling -------------------------------------------------------

def test_missing_input_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input not found"):
        open_source(tmp_path / "nope.mp4")


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(ValueError, match="unsupported input type"):
        open_source(f)


def test_empty_folder_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no images"):
        open_source(tmp_path)


def test_corrupt_video_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "broken.mp4"
    f.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="cannot open video"):
        open_source(f)


def test_unreadable_image_in_folder_is_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "imgs"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), FRAME)
    (folder / "b.jpg").write_bytes(b"corrupt")
    kind, frames, _ = open_source(folder)
    items = list(frames)
    assert kind == "images"
    assert sum(1 for i in items if i.frame is None) == 1


def test_video_input_end_to_end_with_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (120, 100))
    for _ in range(5):
        writer.write(FRAME)
    writer.release()
    kind, frames, fps = open_source(video)
    assert kind == "video"
    assert fps == pytest.approx(10.0, rel=0.05)
    pipeline = SafetyPipeline(ScriptedDetector([WITH_PPE] * 5))
    summary = run_source(pipeline, frames, tmp_path / "out", source_name=str(video),
                         kind=kind, fps=fps)
    assert summary["frames_processed"] == 5
    records = read_jsonl(tmp_path / "out" / "frames.jsonl")
    assert records[1]["timestamp_s"] == pytest.approx(0.1, abs=0.05)


def test_summary_is_json_serialisable(tmp_path: Path) -> None:
    summary, _ = run([WITH_PPE], [FRAME], tmp_path)
    json.dumps(summary)
