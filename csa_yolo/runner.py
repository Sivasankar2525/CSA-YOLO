"""Run the full CSA-YOLO pipeline over a video, an image, or a folder of images.

Outputs (under ``out_dir``):
  frames.jsonl         one JSON record per processed frame (all stage outputs, timestamp, camera)
  workers.csv          one row per worker per frame: track, PPE, compliance, risk
  summary.json         counts and status totals for the run (measured from this run only)
  annotated.mp4        annotated video (video input) or annotated_<name>.jpg per image (image input)

Failure handling:
  - missing or unsupported input raises a clear error before any work starts
  - an unreadable frame is recorded as ``frame_error`` and skipped
  - a detector exception is retried once; if it fails again the frame is recorded
    as ``detector_error`` with no detections (it is NOT replaced by a guess)
  - a frame with no workers, no PPE, or no detections is processed normally
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import cv2
import numpy as np

from csa_yolo.pipeline import PipelineResult, SafetyPipeline

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STATUS_COLOURS = {  # BGR
    "COMPLIANT": (80, 200, 80),
    "NON_COMPLIANT": (50, 50, 230),
    "UNKNOWN": (160, 160, 160),
}


@dataclass
class FrameInput:
    index: int                      # 1-based frame number within the source
    frame: Optional[np.ndarray]     # None if the frame could not be read
    timestamp_s: Optional[float]    # seconds from start; None when unknown (image folders)


def open_source(source: Path, fps_override: Optional[float] = None) -> tuple[str, Iterator[FrameInput], Optional[float]]:
    """Return (kind, frame iterator, fps). Raises for missing or unsupported input."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"input not found: {source}")
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {source}")
        return "images", _image_frames(images), None
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image", _image_frames([source]), None
    if suffix in VIDEO_SUFFIXES:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"cannot open video (corrupt or unsupported codec): {source}")
        fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or None
        return "video", _video_frames(cap), fps
    raise ValueError(f"unsupported input type {suffix!r}; expected a video, an image, or a folder of images")


def _image_frames(paths: list[Path]) -> Iterator[FrameInput]:
    for i, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("unreadable image skipped: %s", path)
        yield FrameInput(index=i, frame=frame, timestamp_s=None)


def _video_frames(cap: "cv2.VideoCapture") -> Iterator[FrameInput]:
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield FrameInput(index=index, frame=frame if frame is not None and frame.size else None,
                             timestamp_s=ts if ts >= 0 else None)
    finally:
        cap.release()


def _detect_with_retry(pipeline: SafetyPipeline, frame: np.ndarray) -> tuple[Optional[PipelineResult], Optional[str]]:
    last_error = None
    for attempt in (1, 2):
        try:
            return pipeline.process_frame(frame), None
        except Exception as exc:  # detector or stage failure: record, do not crash the run
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            logger.warning("frame processing failed (%s)", last_error)
    return None, last_error


def _worker_rows(result: PipelineResult, camera_id: str, frame_index: int, timestamp_s: Optional[float]) -> list[dict]:
    rows = []
    assoc_by_worker = {a.worker_id: a for a in result.associations}
    worker_ids = {d for d in result.tracked_workers} | set(result.compliance_results)
    for track_id in sorted(worker_ids):
        compliance = result.compliance_results.get(track_id)
        state = result.temporal_states.get(track_id)
        risk = result.worker_risks.get(track_id)
        detection = result.tracked_workers.get(track_id)
        assoc = None
        if detection is not None:
            for worker_idx, det in enumerate(result.detections.by_class(["person"])):
                if det is detection:
                    assoc = assoc_by_worker.get(worker_idx)
                    break
        helmet_conf = max((d.confidence for d in assoc.associated_ppe.get("helmet", [])), default=0.0) if assoc else 0.0
        vest_conf = max((d.confidence for d in assoc.associated_ppe.get("vest", [])), default=0.0) if assoc else 0.0
        rows.append({
            "camera_id": camera_id,
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "track_id": track_id,
            "box_xyxy": [round(v, 1) for v in detection.xyxy] if detection is not None else None,
            "helmet_matched": bool(assoc and assoc.associated_ppe.get("helmet")),
            "helmet_confidence": round(helmet_conf, 4),
            "vest_matched": bool(assoc and assoc.associated_ppe.get("vest")),
            "vest_confidence": round(vest_conf, 4),
            "helmet_present_ratio": round(state.evidence("helmet")[1], 4) if state else None,
            "vest_present_ratio": round(state.evidence("vest")[1], 4) if state else None,
            "compliance_status": compliance.status if compliance else None,
            "missing_ppe": ";".join(compliance.missing_ppe) if compliance else "",
            "unknown_ppe": ";".join(compliance.unknown_ppe) if compliance else "",
            "risk_score": round(risk.worker_risk_score, 2) if risk else None,
            "risk_level": risk.worker_risk_level if risk else None,
        })
    return rows


def _annotate(frame: np.ndarray, result: PipelineResult) -> np.ndarray:
    out = frame.copy()
    for det in result.detections.detections if result.detections else ():
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(out, f"{det.class_name} {det.confidence:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for track_id, det in result.tracked_workers.items():
        comp = result.compliance_results.get(track_id)
        status = comp.status if comp else "UNKNOWN"
        colour = STATUS_COLOURS.get(status, STATUS_COLOURS["UNKNOWN"])
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"ID {track_id} {status}"
        if comp and comp.missing_ppe:
            label += " missing:" + ",".join(comp.missing_ppe)
        cv2.putText(out, label, (x1, max(12, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return out


def run_source(
    pipeline: SafetyPipeline,
    frames: Iterable[FrameInput],
    out_dir: Path,
    *,
    source_name: str,
    camera_id: str = "camera-0",
    fps: Optional[float] = None,
    kind: str = "video",
    write_annotated: bool = True,
    max_frames: Optional[int] = None,
) -> dict[str, Any]:
    """Process frames and write the structured outputs. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path = out_dir / "frames.jsonl", out_dir / "workers.csv"
    writer: Optional[cv2.VideoWriter] = None
    counts: Counter = Counter()
    status_totals: Counter = Counter()
    risk_totals: Counter = Counter()
    max_workers = 0
    csv_fields = None
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with jsonl_path.open("w") as jf, csv_path.open("w", newline="") as cf:
        for item in frames:
            if max_frames is not None and counts["frames_seen"] >= max_frames:
                break
            counts["frames_seen"] += 1
            if item.frame is None:
                counts["frame_error"] += 1
                jf.write(json.dumps({"frame_index": item.index, "status": "frame_error"}) + "\n")
                continue

            result, error = _detect_with_retry(pipeline, item.frame)
            if result is None:
                counts["detector_error"] += 1
                jf.write(json.dumps({"frame_index": item.index, "status": "detector_error", "error": error}) + "\n")
                continue

            counts["frames_processed"] += 1
            detections = result.detections.detections if result.detections else ()
            workers = result.tracked_workers
            if not detections:
                counts["frames_empty"] += 1
            max_workers = max(max_workers, len(workers))
            for comp in result.compliance_results.values():
                status_totals[comp.status] += 1
            for risk in result.worker_risks.values():
                risk_totals[risk.worker_risk_level] += 1

            # Source frame index wins over the pipeline's internal counter (which skips failed frames).
            record = {
                **result.to_dict(),
                "frame_index": item.index,
                "timestamp_s": item.timestamp_s,
                "camera_id": camera_id,
                "status": "ok",
                "worker_count": len(workers),
            }
            jf.write(json.dumps(record) + "\n")

            rows = _worker_rows(result, camera_id, item.index, item.timestamp_s)
            if rows:
                if csv_fields is None:
                    csv_fields = list(rows[0].keys())
                    csv.DictWriter(cf, fieldnames=csv_fields).writeheader()
                csv.DictWriter(cf, fieldnames=csv_fields).writerows(rows)

            if write_annotated:
                annotated = _annotate(item.frame, result)
                if kind == "video":
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (w, h))
                    writer.write(annotated)
                else:
                    cv2.imwrite(str(out_dir / f"annotated_{item.index:06d}.jpg"), annotated)

    if writer is not None:
        writer.release()
    if csv_fields is None:
        csv_path.write_text("")  # header-less file when no worker was ever seen
    summary = {
        "source": source_name,
        "input_kind": kind,
        "camera_id": camera_id,
        "fps": fps,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames_seen": counts["frames_seen"],
        "frames_processed": counts["frames_processed"],
        "frames_empty": counts["frames_empty"],
        "frame_errors": counts["frame_error"],
        "detector_errors": counts["detector_error"],
        "max_workers_in_frame": max_workers,
        "compliance_status_counts": dict(status_totals),
        "risk_level_counts": dict(risk_totals),
        "enabled_modules": sorted(pipeline.enabled_modules),
        "outputs": {"frames": jsonl_path.name, "workers": csv_path.name,
                    "annotated": "annotated.mp4" if kind == "video" else "annotated_*.jpg"
                    if write_annotated else None},
        "validation_status": "NOT EXPERIMENTALLY VALIDATED (software output only)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
