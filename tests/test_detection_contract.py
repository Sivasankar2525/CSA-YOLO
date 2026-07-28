"""Unit tests for the detection data contract (no model inference required)."""

import torch

from csa_yolo.detection import Detection, FrameDetections, parse_ultralytics_result, resolve_device


class _FakeBoxes:
    """Minimal stand-in for ultralytics Results.boxes."""

    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[float]) -> None:
        self.xyxy = torch.tensor(xyxy)
        self.conf = torch.tensor(conf)
        self.cls = torch.tensor(cls)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None, names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


NAMES = {0: "person", 1: "helmet", 2: "vest"}


def test_parse_maps_classes_and_boxes() -> None:
    result = _FakeResult(
        _FakeBoxes(
            xyxy=[[10.0, 20.0, 110.0, 220.0], [30.0, 15.0, 60.0, 45.0]],
            conf=[0.91, 0.72],
            cls=[0.0, 1.0],
        ),
        names=NAMES,
    )
    frame = parse_ultralytics_result(result, frame_index=7)
    assert frame.frame_index == 7
    assert len(frame.detections) == 2
    person, helmet = frame.detections
    assert person.class_name == "person" and helmet.class_name == "helmet"
    assert person.xyxy == (10.0, 20.0, 110.0, 220.0)
    assert abs(person.confidence - 0.91) < 1e-6


def test_parse_empty_boxes() -> None:
    frame = parse_ultralytics_result(_FakeResult(None, NAMES))
    assert frame.detections == ()


def test_by_class_filters() -> None:
    dets = (
        Detection((0, 0, 1, 1), 0.9, 0, "person"),
        Detection((0, 0, 1, 1), 0.8, 1, "helmet"),
        Detection((0, 0, 1, 1), 0.7, 2, "vest"),
    )
    frame = FrameDetections(frame_index=0, detections=dets)
    assert [d.class_name for d in frame.by_class(["helmet", "vest"])] == ["helmet", "vest"]


def test_detection_geometry() -> None:
    det = Detection((10.0, 20.0, 30.0, 60.0), 0.5, 0, "person")
    assert det.center == (20.0, 40.0)
    assert det.area == 20.0 * 40.0


def test_resolve_device_explicit_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}
