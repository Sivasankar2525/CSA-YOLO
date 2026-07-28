"""Detection backbone wrapper and the typed detection data contract."""

from csa_yolo.detection.detector import (
    Detection,
    Detector,
    FrameDetections,
    parse_ultralytics_result,
    resolve_device,
)

__all__ = ["Detection", "Detector", "FrameDetections", "parse_ultralytics_result", "resolve_device"]
