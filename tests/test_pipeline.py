"""End-to-end SafetyPipeline tests with a scripted detector (no model, CPU-only)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from csa_yolo.detection import Detection, FrameDetections
from csa_yolo.pipeline import DEFAULT_SETTINGS, SafetyPipeline, settings_from_config
from csa_yolo.scrm.compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = (0.0, 0.0, 100.0, 100.0)
HELMET = (35.0, 5.0, 65.0, 20.0)
VEST = (10.0, 40.0, 90.0, 70.0)
DUMMY_FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def det(box, name, conf=0.9):
    class_id = {"person": 0, "helmet": 1, "vest": 2}.get(name, 9)
    return Detection(xyxy=box, confidence=conf, class_id=class_id, class_name=name)


class ScriptedDetector:
    """Stands in for the YOLO backbone: returns pre-scripted detections per call."""

    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = frames
        self._calls = 0

    def detect(self, image, frame_index: int = 0) -> FrameDetections:
        dets = self._frames[self._calls]
        self._calls += 1
        return FrameDetections(frame_index=self._calls, detections=tuple(dets))


def run(frames: list[list[Detection]], **kwargs):
    pipeline = SafetyPipeline(ScriptedDetector(frames), **kwargs)
    results = [pipeline.process_frame(DUMMY_FRAME) for _ in frames]
    return pipeline, results


WITH_PPE = [det(WORKER, "person"), det(HELMET, "helmet"), det(VEST, "vest")]
NO_HELMET = [det(WORKER, "person"), det(VEST, "vest")]
NOTHING = []


def test_result_exposes_every_stage() -> None:
    _, results = run([WITH_PPE] * 5)
    result = results[-1]
    assert result.detections is not None and len(result.detections.detections) == 3
    assert len(result.associations) == 1
    assert list(result.tracked_workers) == [0]
    assert list(result.temporal_states) == [0]
    assert list(result.compliance_results) == [0]
    assert list(result.worker_risks) == [0]


def test_single_frame_is_unknown_not_decided() -> None:
    # Regression for audit defect 4: one frame is not enough evidence for a decision.
    _, [result] = run([WITH_PPE])
    assert result.compliance_results[0].status == UNKNOWN
    assert result.worker_risks == {}  # UNKNOWN is not scored


def test_sustained_ppe_gives_compliant_low_risk() -> None:
    _, results = run([WITH_PPE] * 5)
    final = results[-1]
    assert final.compliance_results[0].status == COMPLIANT
    assert final.worker_risks[0].worker_risk_level == "LOW"


def test_helmet_removed_for_five_frames_is_non_compliant() -> None:
    _, results = run([WITH_PPE] * 2 + [NO_HELMET] * 5)
    final = results[-1]
    assert final.compliance_results[0].status == NON_COMPLIANT
    assert final.compliance_results[0].missing_ppe == ["helmet"]
    assert final.worker_risks[0].worker_risk_score == pytest.approx(25.0)


def test_brief_helmet_drop_is_not_a_violation() -> None:
    # One helmet-less frame inside sustained PPE: the window smooths it out.
    _, results = run([WITH_PPE] * 4 + [NO_HELMET] + [WITH_PPE] * 2)
    assert results[-1].compliance_results[0].status == COMPLIANT


def test_worker_never_wearing_helmet_is_non_compliant() -> None:
    # Regression for audit defect 4: a helmet that is never seen is recorded as absent
    # evidence once the worker is tracked, so it is a violation after enough frames.
    _, results = run([NO_HELMET] * 5)
    assert results[-1].compliance_results[0].status == NON_COMPLIANT


def test_frame_without_workers_creates_no_worker_state() -> None:
    _, [result] = run([[det(HELMET, "helmet")]])
    assert result.associations == []
    assert result.tracked_workers == {}
    assert result.temporal_states == {}


def test_empty_frame_returns_empty_result() -> None:
    _, [result] = run([NOTHING])
    assert result.detections.detections == ()
    assert result.compliance_results == {}


def test_two_workers_get_distinct_tracks_and_helmets() -> None:
    worker_b = (300.0, 0.0, 400.0, 100.0)
    helmet_b = (335.0, 5.0, 365.0, 20.0)
    frame = [
        det(WORKER, "person"), det(HELMET, "helmet"),
        det(worker_b, "person"), det(helmet_b, "helmet"),
    ]
    _, [result] = run([frame])
    assert sorted(result.tracked_workers) == [0, 1]
    assert len(result.compliance_results) == 2
    owners = [a.worker_id for a in result.associations if "helmet" in a.associated_ppe]
    assert owners == [0, 1]


def test_result_is_json_serialisable() -> None:
    _, [result] = run([WITH_PPE])
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["frame_index"] == 1
    assert set(payload) == {
        "frame_index", "detections", "associations", "tracked_workers",
        "temporal_states", "compliance_results", "worker_risks",
    }


# --- stale-state cleanup (audit defect 5) ----------------------------------

def test_departed_worker_temporal_state_is_removed() -> None:
    frames = [WITH_PPE] * 3 + [NOTHING] * 60
    pipeline, _ = run(frames, track_max_age=30)
    assert pipeline.temporal_memory.states == {}
    assert pipeline.tracker.tracks == {}


def test_reappearing_worker_after_long_gap_gets_new_track_id() -> None:
    frames = [WITH_PPE] * 3 + [NOTHING] * 40 + [WITH_PPE]
    _, results = run(frames, track_max_age=30)
    assert list(results[-1].tracked_workers) == [1]


def test_long_run_pipeline_memory_is_bounded() -> None:
    frames = [WITH_PPE] * 500
    pipeline, _ = run(frames, track_max_age=30)
    assert len(pipeline.temporal_memory.states) <= 31


# --- configuration ---------------------------------------------------------

def test_default_yaml_matches_code_defaults() -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text())
    assert settings_from_config(raw) == DEFAULT_SETTINGS


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        settings_from_config({"compliance": {"present_treshold": 0.7}})


def test_required_ppe_must_be_helmet_or_vest() -> None:
    with pytest.raises(ValueError, match="subset"):
        SafetyPipeline(ScriptedDetector([]), required_ppe=["helmet", "gloves"])


def test_from_config_applies_yaml_values() -> None:
    raw = {"compliance": {"min_evidence_frames": 3}, "temporal": {"window_size": 3}}
    pipeline = SafetyPipeline.from_config(ScriptedDetector([WITH_PPE] * 3), raw)
    assert pipeline.compliance_engine.min_evidence_frames == 3
    assert pipeline.temporal_memory.window_size == 3
    _, results = run([WITH_PPE] * 3, min_evidence_frames=3, window_size=3)
    assert results[-1].compliance_results[0].status == COMPLIANT
