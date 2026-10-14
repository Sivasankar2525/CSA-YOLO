"""End-to-end CSA-YOLO safety pipeline orchestrator."""

from dataclasses import dataclass, field

import numpy as np

from csa_yolo.cram.risk_assessment import RiskAssessmentEngine, RiskResult
from csa_yolo.detection import Detector, FrameDetections
from csa_yolo.scrm.compliance import ComplianceEngine, ComplianceResult
from csa_yolo.tracking.tracker import SimpleIOUTracker
from csa_yolo.tsmm.memory import TemporalSafetyMemory, WorkerSafetyState
from csa_yolo.wpam.associate import associate_ppe_to_workers, WorkerPPEAssociation


@dataclass
class PipelineResult:
    """Complete output from one frame of the pipeline."""
    frame_index: int
    detections: FrameDetections = None
    associations: list[WorkerPPEAssociation] = field(default_factory=list)
    tracked_workers: dict = field(default_factory=dict)  # {track_id: Detection}
    temporal_states: dict[int, WorkerSafetyState] = field(default_factory=dict)
    compliance_results: dict[int, ComplianceResult] = field(default_factory=dict)
    worker_risks: dict[int, RiskResult] = field(default_factory=dict)
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
        worker_class_names: list[str] = None,
        helmet_class_names: list[str] = None,
        vest_class_names: list[str] = None,
        required_ppe: list[str] = None,
        **kwargs,
    ):
        """Initialize pipeline.

        Args:
            detector: YOLO detector instance.
            worker_class_names: Class names for workers (default: ["person"]).
            helmet_class_names: Class names for helmets (default: ["helmet", "hat"]).
            vest_class_names: Class names for vests (default: ["vest", "safety_vest"]).
            required_ppe: Required PPE list (default: ["helmet", "vest"]).
        """
        self.detector = detector

        self.worker_class_names = worker_class_names or ["person"]
        self.helmet_class_names = helmet_class_names or ["helmet", "hat", "hardhat"]
        self.vest_class_names = vest_class_names or ["vest", "safety_vest"]
        self.required_ppe = required_ppe or ["helmet", "vest"]

        self.tracker = SimpleIOUTracker(iou_threshold=0.3)
        self.temporal_memory = TemporalSafetyMemory(window_size=5)
        self.compliance_engine = ComplianceEngine(
            required_ppe=self.required_ppe,
            confirmation_threshold=0.7,
        )
        self.risk_engine = RiskAssessmentEngine()

        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process one frame through the complete pipeline.

        Args:
            frame: Input frame (BGR, numpy array).

        Returns:
            PipelineResult containing all outputs.
        """
        self.frame_count += 1
        result = PipelineResult(frame_index=self.frame_count)

        # 1. Detection
        result.detections = self.detector.detect(frame)

        if not result.detections.detections:
            return result

        # 2. Separate workers and PPE
        workers = result.detections.by_class(self.worker_class_names)
        ppe_items = [
            d for d in result.detections.detections
            if d.class_name.lower() in (
                [c.lower() for c in self.helmet_class_names] +
                [c.lower() for c in self.vest_class_names]
            )
        ]

        if not workers:
            return result

        # 3. WPAM: Associate PPE to workers
        result.associations = associate_ppe_to_workers(
            workers,
            ppe_items,
            helmet_classes=self.helmet_class_names,
            vest_classes=self.vest_class_names,
        )

        # 4. Tracking: Assign persistent worker IDs
        result.tracked_workers = self.tracker.update(workers)

        # 5. TSMM: Temporal safety memory
        for track_id, detection in result.tracked_workers.items():
            # Find corresponding association
            ppe_conf = {}
            for assoc in result.associations:
                if assoc.worker_id == list(workers).index(detection):
                    for ppe_class, detections in assoc.associated_ppe.items():
                        if detections:
                            ppe_conf[ppe_class] = max(d.confidence for d in detections)
                        else:
                            ppe_conf[ppe_class] = 0.0
                    break

            state = self.temporal_memory.update(track_id, ppe_conf)
            result.temporal_states[track_id] = state

        # 6. SCRM: Compliance reasoning
        result.compliance_results = self.compliance_engine.check_compliance_batch(
            result.temporal_states
        )

        # 7. CRAM: Risk assessment
        for track_id, compliance in result.compliance_results.items():
            state = result.temporal_states[track_id]
            risk = self.risk_engine.assess_worker_risk(state, compliance)
            result.worker_risks[track_id] = risk

        return result

    def reset(self) -> None:
        """Reset for new video."""
        self.tracker.reset()
        self.temporal_memory.reset()
        self.frame_count = 0
