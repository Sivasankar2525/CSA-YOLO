"""Module toggles for ablation: each ladder rung must run exactly its stages."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from csa_yolo.detection import Detection, FrameDetections
from csa_yolo.pipeline import ALL_MODULES, SafetyPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
WORKER = (0.0, 0.0, 100.0, 100.0)
HELMET = (35.0, 5.0, 65.0, 20.0)
VEST = (10.0, 40.0, 90.0, 70.0)


def det(box, name):
    return Detection(xyxy=box, confidence=0.9, class_id=0, class_name=name)


class ScriptedDetector:
    def __init__(self, frames):
        self._frames = frames
        self._calls = 0

    def detect(self, image, frame_index: int = 0):
        dets = self._frames[self._calls]
        self._calls += 1
        return FrameDetections(frame_index=self._calls, detections=tuple(dets))


SCENE = [det(WORKER, "person"), det(HELMET, "helmet"), det(VEST, "vest")]


def run_stack(modules, frames=5):
    pipeline = SafetyPipeline(ScriptedDetector([SCENE] * frames), enabled_modules=modules)
    result = None
    for _ in range(frames):
        result = pipeline.process_frame(FRAME)
    return result


@pytest.mark.parametrize(
    ("modules", "populated"),
    [
        ([], set()),
        (["wpam"], {"associations"}),
        (["wpam", "tracking"], {"associations", "tracked_workers"}),
        (["wpam", "tracking", "tsmm"], {"associations", "tracked_workers", "temporal_states"}),
        (
            ["wpam", "tracking", "tsmm", "scrm"],
            {"associations", "tracked_workers", "temporal_states", "compliance_results"},
        ),
        (
            list(ALL_MODULES),
            {"associations", "tracked_workers", "temporal_states",
             "compliance_results", "worker_risks"},
        ),
    ],
    ids=["detector_only", "wpam", "tracking", "tsmm", "scrm", "full"],
)
def test_each_rung_populates_only_its_stages(modules, populated) -> None:
    result = run_stack(modules)
    assert result.detections.detections, "detector always runs"
    stages = ["associations", "tracked_workers", "temporal_states",
              "compliance_results", "worker_risks"]
    for stage in stages:
        value = getattr(result, stage)
        assert bool(value) == (stage in populated), stage


def test_default_enables_all_modules_and_matches_explicit_full() -> None:
    default = SafetyPipeline(ScriptedDetector([SCENE]))
    assert default.enabled_modules == frozenset(ALL_MODULES)
    default_out = default.process_frame(FRAME).to_dict()
    explicit_out = run_stack(list(ALL_MODULES), frames=1).to_dict()
    assert default_out == explicit_out


def test_tsmm_requires_wpam_and_tracking() -> None:
    with pytest.raises(ValueError, match="requires"):
        SafetyPipeline(ScriptedDetector([]), enabled_modules=["tsmm"])
    with pytest.raises(ValueError, match="requires"):
        SafetyPipeline(ScriptedDetector([]), enabled_modules=["wpam", "tsmm"])


def test_cram_requires_scrm() -> None:
    with pytest.raises(ValueError, match="requires"):
        SafetyPipeline(ScriptedDetector([]), enabled_modules=["wpam", "tracking", "tsmm", "cram"])


def test_unknown_module_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown module"):
        SafetyPipeline(ScriptedDetector([]), enabled_modules=["wpam", "gan"])


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "run_ablation", PROJECT_ROOT / "evaluation" / "run_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_config_has_six_cumulative_rungs() -> None:
    ablation = _load_ablation_module()
    configs = ablation.load_ablation_configs(PROJECT_ROOT / "configs" / "ablation.yaml")
    assert list(configs) == [
        "detector_only",
        "detector_wpam",
        "detector_wpam_tracking",
        "detector_wpam_tracking_tsmm",
        "detector_wpam_tracking_tsmm_scrm",
        "full_csa_yolo",
    ]
    sizes = [len(modules) for modules in configs.values()]
    assert sizes == [0, 1, 2, 3, 4, 5]
    # Every rung must be a valid pipeline stack.
    for modules in configs.values():
        SafetyPipeline(ScriptedDetector([]), enabled_modules=modules)
