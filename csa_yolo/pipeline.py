"""End-to-end CSA-YOLO safety pipeline orchestrator."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from csa_yolo.cram.risk_assessment import RiskAssessmentEngine, RiskResult
from csa_yolo.detection import Detector, FrameDetections
from csa_yolo.scrm.compliance import UNKNOWN, ComplianceEngine, ComplianceResult
from csa_yolo.tracking.tracker import SimpleIOUTracker
from csa_yolo.tsmm.memory import TemporalSafetyMemory, WorkerSafetyState
from csa_yolo.wpam.associate import WorkerPPEAssociation, WPAMConfig, associate_ppe_to_workers

ALL_MODULES = ("wpam", "tracking", "tsmm", "scrm", "cram")

# Each module's prerequisites. Enforced so that every ablation configuration is a valid
# stack: TSMM smooths per-track PPE evidence, which needs WPAM associations and persistent
# track IDs; SCRM decides on TSMM states; CRAM scores SCRM results.
MODULE_REQUIREMENTS = {
    "wpam": (),
    "tracking": (),
    "tsmm": ("wpam", "tracking"),
    "scrm": ("tsmm",),
    "cram": ("scrm",),
}

# Defaults. Mirrored in configs/default.yaml; tests check the two stay equal.
DEFAULT_SETTINGS: dict[str, Any] = {
    "association": {"gate_margin_ratio": 0.10},
    "tracking": {"iou_threshold": 0.3, "max_age": 30},
    "temporal": {"window_size": 5, "presence_confidence": 0.5},
    "compliance": {
        "required_ppe": ["helmet", "vest"],
        "present_threshold": 0.7,
        "absent_threshold": 0.3,
        "min_evidence_frames": 5,
    },
}


@dataclass
class PipelineResult:
    """Complete output from one frame of the pipeline."""
    frame_index: int
    detections: FrameDetections = None
    associations: list[WorkerPPEAssociation] = field(default_factory=list)
    tracked_workers: dict = field(default_factory=dict)  # {track_id: Detection}
    temporal_states: dict[int, WorkerSafetyState] = field(default_factory=dict)
    compliance_results: dict[int, ComplianceResult] = field(default_factory=dict)
    worker_risks: dict[int, RiskResult] = field(default_factory=dict)  # scored workers only
    annotated_frame: np.ndarray = None

    def to_dict(self) -> dict:
        """JSON-serializable output."""
        return {
            "frame_index": self.frame_index,
            "detections": self.detections.to_dict(),
            "associations": [a.to_dict() for a in self.associations],
            "tracked_workers": {
                str(tid): d.to_dict() for tid, d in self.tracked_workers.items()
            },
            "temporal_states": {
                str(tid): state.to_dict() for tid, state in self.temporal_states.items()
            },
            "compliance_results": {
                str(tid): result.to_dict()
                for tid, result in self.compliance_results.items()
            },
            "worker_risks": {
                str(tid): result.to_dict()
                for tid, result in self.worker_risks.items()
            },
        }


class SafetyPipeline:
    """Complete CSA-YOLO safety analysis pipeline."""

    def __init__(
        self,
        detector: Detector,
        worker_class_names: Optional[list[str]] = None,
        helmet_class_names: Optional[list[str]] = None,
        vest_class_names: Optional[list[str]] = None,
        required_ppe: Optional[list[str]] = None,
        enabled_modules: Iterable[str] = ALL_MODULES,
        wpam_config: Optional[WPAMConfig] = None,
        iou_threshold: float = DEFAULT_SETTINGS["tracking"]["iou_threshold"],
        track_max_age: int = DEFAULT_SETTINGS["tracking"]["max_age"],
        window_size: int = DEFAULT_SETTINGS["temporal"]["window_size"],
        presence_confidence: float = DEFAULT_SETTINGS["temporal"]["presence_confidence"],
        present_threshold: float = DEFAULT_SETTINGS["compliance"]["present_threshold"],
        absent_threshold: Optional[float] = DEFAULT_SETTINGS["compliance"]["absent_threshold"],
        min_evidence_frames: int = DEFAULT_SETTINGS["compliance"]["min_evidence_frames"],
    ):
        """Initialize pipeline.

        Args:
            detector: YOLO detector instance (or any object with ``detect(frame)``).
            worker_class_names: Class names for workers (default: ["person"]).
            helmet_class_names: Class names for helmets (default: ["helmet", "hat", "hardhat"]).
            vest_class_names: Class names for vests (default: ["vest", "safety_vest"]).
            required_ppe: Required PPE classes; subset of {"helmet", "vest"}
                (default: ["helmet", "vest"]).
            enabled_modules: Subset of ``ALL_MODULES`` to run. Disabled stages produce
                empty outputs. The detector always runs. Used by the ablation study.
            wpam_config: Association parameters (default: ``WPAMConfig()``).
            iou_threshold: Tracker IoU threshold.
            track_max_age: Frames a worker may go unmatched before its track and TSMM
                state are removed.
            window_size: TSMM observations kept per worker and class.
            presence_confidence: PPE confidence at or above which an observation is "present".
            present_threshold: Present ratio for a class to be PRESENT (SCRM).
            absent_threshold: Present ratio at or below which a class is ABSENT (SCRM).
            min_evidence_frames: Observations needed before SCRM decides a class.

        Raises:
            ValueError: If an unknown module is named, a module is enabled without its
                prerequisites, or ``required_ppe`` contains a class WPAM does not produce.
        """
        self.detector = detector
        self.enabled_modules = self._validate_modules(enabled_modules)

        self.worker_class_names = worker_class_names or ["person"]
        self.helmet_class_names = helmet_class_names or ["helmet", "hat", "hardhat"]
        self.vest_class_names = vest_class_names or ["vest", "safety_vest"]
        self.required_ppe = list(required_ppe) if required_ppe else ["helmet", "vest"]
        if not set(self.required_ppe) <= {"helmet", "vest"}:
            raise ValueError(f"required_ppe must be a subset of helmet/vest, got {self.required_ppe}")

        self.wpam_config = wpam_config or WPAMConfig()
        self.track_max_age = track_max_age
        self.tracker = SimpleIOUTracker(iou_threshold=iou_threshold, max_age=track_max_age)
        self.temporal_memory = TemporalSafetyMemory(
            window_size=window_size, confidence_threshold=presence_confidence
        )
        self.compliance_engine = ComplianceEngine(
            required_ppe=self.required_ppe,
            confirmation_threshold=present_threshold,
            absent_threshold=absent_threshold,
            min_evidence_frames=min_evidence_frames,
        )
        self.risk_engine = RiskAssessmentEngine()

        self.frame_count = 0

    @classmethod
    def from_config(
        cls,
        detector: Detector,
        raw: dict[str, Any],
        enabled_modules: Iterable[str] = ALL_MODULES,
    ) -> "SafetyPipeline":
        """Build a pipeline from the parsed YAML config. Missing keys use DEFAULT_SETTINGS."""
        s = settings_from_config(raw)
        return cls(
            detector,
            required_ppe=s["compliance"]["required_ppe"],
            enabled_modules=enabled_modules,
            wpam_config=WPAMConfig(**s["association"]),
            iou_threshold=s["tracking"]["iou_threshold"],
            track_max_age=s["tracking"]["max_age"],
            window_size=s["temporal"]["window_size"],
            presence_confidence=s["temporal"]["presence_confidence"],
            present_threshold=s["compliance"]["present_threshold"],
            absent_threshold=s["compliance"]["absent_threshold"],
            min_evidence_frames=s["compliance"]["min_evidence_frames"],
        )

    @staticmethod
    def _validate_modules(enabled_modules: Iterable[str]) -> frozenset[str]:
        enabled = frozenset(enabled_modules)
        unknown = enabled - set(ALL_MODULES)
        if unknown:
            raise ValueError(f"Unknown module(s): {sorted(unknown)}; expected {ALL_MODULES}")
        for module in enabled:
            missing = [req for req in MODULE_REQUIREMENTS[module] if req not in enabled]
            if missing:
                raise ValueError(f"Module '{module}' requires {missing} to be enabled")
        return enabled

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process one frame through the complete pipeline.

        The tracker and TSMM cleanup run on every frame, including frames with no
        detections, so stale tracks age out on schedule.

        Args:
            frame: Input frame (BGR, numpy array).

        Returns:
            PipelineResult containing all outputs.
        """
        self.frame_count += 1
        result = PipelineResult(frame_index=self.frame_count)

        if "tsmm" in self.enabled_modules:
            self.temporal_memory.remove_old_tracks(self.frame_count, max_age=self.track_max_age)

        # 1. Detection (always on)
        result.detections = self.detector.detect(frame)
        workers = result.detections.by_class(self.worker_class_names)
        ppe_items = [
            d for d in result.detections.detections
            if d.class_name.lower() in (
                [c.lower() for c in self.helmet_class_names] +
                [c.lower() for c in self.vest_class_names]
            )
        ]

        # 2. WPAM: spatial gate + one-to-one assignment
        if "wpam" in self.enabled_modules:
            result.associations = associate_ppe_to_workers(
                workers,
                ppe_items,
                helmet_classes=self.helmet_class_names,
                vest_classes=self.vest_class_names,
                config=self.wpam_config,
            )

        # 3. Tracking: persistent worker IDs (updated every frame, even with no workers)
        if "tracking" in self.enabled_modules:
            result.tracked_workers = self.tracker.update(workers)

        # 4. TSMM: per-track PPE evidence (absent unless associated this frame)
        if "tsmm" in self.enabled_modules:
            worker_index = {id(w): i for i, w in enumerate(workers)}
            assoc_by_worker = {a.worker_id: a for a in result.associations}
            for track_id, detection in result.tracked_workers.items():
                ppe_conf = {ppe_class: 0.0 for ppe_class in self.required_ppe}
                assoc = assoc_by_worker.get(worker_index[id(detection)])
                if assoc is not None:
                    for ppe_class, matched in assoc.associated_ppe.items():
                        if ppe_class in ppe_conf and matched:
                            ppe_conf[ppe_class] = max(d.confidence for d in matched)
                result.temporal_states[track_id] = self.temporal_memory.update(
                    track_id, ppe_conf, frame_index=self.frame_count
                )

        # 5. SCRM: compliance from temporal evidence
        if "scrm" in self.enabled_modules:
            result.compliance_results = self.compliance_engine.check_compliance_batch(
                result.temporal_states
            )

        # 6. CRAM: scored workers only (UNKNOWN has no evidenced violation)
        if "cram" in self.enabled_modules:
            for track_id, compliance in result.compliance_results.items():
                if compliance.status == UNKNOWN:
                    continue
                state = result.temporal_states[track_id]
                result.worker_risks[track_id] = self.risk_engine.assess_worker_risk(state, compliance)

        return result

    def reset(self) -> None:
        """Reset for new video."""
        self.tracker.reset()
        self.temporal_memory.reset()
        self.frame_count = 0


def settings_from_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge a parsed YAML config over DEFAULT_SETTINGS (only known keys are read)."""
    merged = {section: dict(values) for section, values in DEFAULT_SETTINGS.items()}
    for section, values in merged.items():
        overrides = (raw or {}).get(section) or {}
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError(f"unknown key(s) in '{section}': {sorted(unknown)}")
        values.update(overrides)
    return merged
