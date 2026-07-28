"""Typed wrapper around the YOLO/ICSD-YOLO detector.

This module defines the *data contract* between the detection backbone and
every downstream CSA-YOLO module: WPAM, TSMM, SCRM and CRAM never touch raw
Ultralytics result objects — they consume :class:`Detection` /
:class:`FrameDetections`, which keeps the research modules independent of the
backbone implementation (stock YOLO now, fine-tuned ICSD-YOLO in Phase 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame.

    Attributes:
        xyxy: Bounding box ``(x1, y1, x2, y2)`` in absolute pixel coordinates.
        confidence: Detector confidence in ``[0, 1]``.
        class_id: Integer class index from the detector.
        class_name: Human-readable class label (e.g. ``person``, ``helmet``).
    """

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        """Box center ``(cx, cy)`` in pixels."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Box area in square pixels (non-negative)."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation used by scripts and the dashboard."""
        return {
            "xyxy": [round(v, 2) for v in self.xyxy],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True)
class FrameDetections:
    """All detections for one frame, plus frame metadata."""

    frame_index: int
    detections: tuple[Detection, ...] = field(default_factory=tuple)

    def by_class(self, class_names: Sequence[str]) -> tuple[Detection, ...]:
        """Return detections whose class name is in ``class_names``."""
        wanted = set(class_names)
        return tuple(d for d in self.detections if d.class_name in wanted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "detections": [d.to_dict() for d in self.detections],
        }


def resolve_device(requested: str = "auto") -> str:
    """Resolve the compute device.

    Args:
        requested: ``auto``, ``cpu``, ``cuda``, or ``mps``.

    Returns:
        The concrete device string. ``auto`` prefers CUDA, then Apple MPS,
        then CPU.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    """Runs the detection backbone and returns typed :class:`FrameDetections`.

    Args:
        model_path: Ultralytics model name or path to a ``.pt`` checkpoint.
        confidence_threshold: Minimum confidence for a detection to be kept.
        iou_threshold: NMS IoU threshold.
        device: ``auto`` (default), ``cpu``, ``cuda`` or ``mps``.
        image_size: Inference resolution (square).
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        image_size: int = 640,
    ) -> None:
        # Imported lazily so unit tests of the data contract don't need the
        # (heavy) ultralytics runtime.
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self._model = YOLO(str(model_path))
        logger.info(
            "Detector initialized: model=%s device=%s conf=%.2f iou=%.2f imgsz=%d",
            model_path, self.device, confidence_threshold, iou_threshold, image_size,
        )

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping from class id to class name for the loaded model."""
        return dict(self._model.names)

    def detect(self, image: np.ndarray | str | Path, frame_index: int = 0) -> FrameDetections:
        """Run detection on a single image or frame.

        Args:
            image: BGR image array (OpenCV convention) or path to an image.
            frame_index: Index of this frame within the source video/stream.

        Returns:
            Typed detections for the frame.
        """
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        return parse_ultralytics_result(results[0], frame_index=frame_index)


def parse_ultralytics_result(result: Any, frame_index: int = 0) -> FrameDetections:
    """Convert one Ultralytics ``Results`` object into :class:`FrameDetections`.

    Kept as a standalone function so it can be unit-tested with lightweight
    stand-in objects instead of running a real model.
    """
    names: dict[int, str] = dict(result.names)
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is not None:
        for xyxy, conf, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):
            class_id = int(cls)
            detections.append(
                Detection(
                    xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(conf),
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                )
            )
    return FrameDetections(frame_index=frame_index, detections=tuple(detections))
