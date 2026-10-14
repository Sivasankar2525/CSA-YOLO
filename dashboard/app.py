#!/usr/bin/env python3
"""CSA-YOLO Streamlit Dashboard."""

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from csa_yolo.detection import Detector, resolve_device
from csa_yolo.pipeline import SafetyPipeline
from csa_yolo.utils.config import load_config

st.set_page_config(page_title="CSA-YOLO Safety Dashboard", layout="wide")
st.title("🏗️ CSA-YOLO Construction Safety Dashboard")

st.markdown(
    """
This is a prototype demonstration of the **Construction Safety-Aware YOLO** pipeline.
It performs real-time safety analysis including worker detection, PPE verification,
temporal memory, compliance checking, and risk assessment.

**DISCLAIMER:** This is a research prototype for educational purposes only.
It should NOT be used for real safety-critical decisions without proper validation.
"""
)

# Sidebar configuration
st.sidebar.header("Configuration")

# Load default config
try:
    config = load_config("configs/default.yaml")
except Exception as e:
    st.error(f"Failed to load config: {e}")
    st.stop()

device = resolve_device(config["detector"]["device"])
st.sidebar.write(f"**Device:** {device}")

# Initialize detector and pipeline
@st.cache_resource
def init_pipeline():
    detector = Detector(
        model_path=str(
            Path("weights") / config["detector"]["model"]
        ),
        confidence_threshold=config["detector"]["confidence_threshold"],
        iou_threshold=config["detector"]["iou_threshold"],
        device=device,
        image_size=config["detector"]["image_size"],
    )
    pipeline = SafetyPipeline(detector)
    return pipeline


pipeline = init_pipeline()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "📊 Analytics"])

# TAB 1: Image Analysis
with tab1:
    st.header("Single Image Analysis")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_image is not None:
        # Read image
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process
        with st.spinner("Processing..."):
            result = pipeline.process_frame(image)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detections")
            st.write(f"Total objects detected: {len(result.detections.detections)}")
            for detection in result.detections.detections:
                st.write(
                    f"- {detection.class_name}: {detection.confidence:.3f}"
                )

        with col2:
            st.subheader("Tracked Workers")
            st.write(f"Active workers: {len(result.tracked_workers)}")

        # Compliance and Risk
        st.subheader("Compliance & Risk Analysis")

        if result.compliance_results:
            for track_id, compliance in result.compliance_results.items():
                with st.expander(f"Worker #{track_id}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        status_color = "🟢" if compliance.status == "COMPLIANT" else "🔴"
                        st.metric(
                            "Status",
                            f"{status_color} {compliance.status}",
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f"{compliance.confidence:.1%}",
                        )

                    with col3:
                        if track_id in result.worker_risks:
                            risk = result.worker_risks[track_id]
                            risk_color = (
                                "🟢" if risk.worker_risk_level == "LOW"
                                else "🟡" if risk.worker_risk_level == "MEDIUM"
                                else "🔴"
                            )
                            st.metric(
                                "Risk Level",
                                f"{risk_color} {risk.worker_risk_level}",
                            )

                    if compliance.missing_ppe:
                        st.warning(f"Missing: {', '.join(compliance.missing_ppe)}")

                    st.write(f"**Reason:** {compliance.reason}")

                    if track_id in result.worker_risks:
                        risk = result.worker_risks[track_id]
                        st.progress(
                            min(risk.worker_risk_score / 100, 1.0),
                            text=f"Risk Score: {risk.worker_risk_score:.1f}/100",
                        )
                        if risk.risk_factors:
                            st.write(
                                f"**Risk Factors:** {'; '.join(risk.risk_factors)}"
                            )
        else:
            st.info("No workers detected in this image.")

        # JSON export
        if st.button("📋 Export as JSON"):
            json_data = result.to_dict()
            st.json(json_data)
            st.download_button(
                "Download JSON",
                json.dumps(json_data, indent=2),
                "csa_yolo_result.json",
                "application/json",
            )

# TAB 2: Video Analysis
with tab2:
    st.header("Video Stream Analysis")
    st.info(
        "Upload a video to process frame-by-frame through the CSA-YOLO pipeline. "
        "For best results, use short videos (< 2 min)."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        # Save temporarily
        temp_video_path = Path("/tmp") / uploaded_video.name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info(f"Processing video: {uploaded_video.name}")

        # Process video
        cap = cv2.VideoCapture(str(temp_video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Limit frames for performance
        max_frames = st.slider("Max frames to process", 1, frame_count, min(frame_count, 50))

        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        results_placeholder = st.empty()

        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for display
            display_frame = cv2.resize(frame, (800, 600))

            # Process
            result = pipeline.process_frame(frame)

            # Display
            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            # Summary
            with results_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Frame", f"{frame_idx} / {max_frames}")
                with col2:
                    st.metric("Workers", len(result.tracked_workers))
                with col3:
                    non_compliant = sum(
                        1 for c in result.compliance_results.values()
                        if c.status == "NON_COMPLIANT"
                    )
                    st.metric("Non-Compliant", non_compliant)

            progress_bar.progress((frame_idx + 1) / max_frames)
            frame_idx += 1

        cap.release()
        temp_video_path.unlink()
        st.success("Video processing complete!")

# TAB 3: Analytics
with tab3:
    st.header("System Overview")

    st.subheader("Pipeline Architecture")
    st.write(
        """
    **CSA-YOLO Pipeline:**

    1. **Detection:** YOLO-based worker and PPE detection
    2. **WPAM:** Worker–PPE Association Module (assigns PPE to workers)
    3. **Tracking:** Maintains persistent worker IDs across frames
    4. **TSMM:** Temporal Safety Memory (reduces false alarms)
    5. **SCRM:** Safety Compliance Reasoning (checks required PPE)
    6. **CRAM:** Construction Risk Assessment (rule-based risk scoring)
    """
    )

    st.subheader("Module Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**WPAM**")
        st.write("Associates detected PPE items with individual workers using geometric proximity and IoU matching.")

    with col2:
        st.write("**TSMM**")
        st.write("Maintains sliding-window temporal history per worker to reduce frame-by-frame noise.")

    with col3:
        st.write("**SCRM**")
        st.write("Determines compliance status based on confirmed PPE detection over temporal window.")

    st.divider()

    st.subheader("⚠️ Disclaimer")
    st.warning(
        """
    **THIS IS A RESEARCH PROTOTYPE.**

    This system is NOT:
    - OSHA-certified
    - Legally binding for safety decisions
    - Suitable for real-world safety-critical deployments
    - Guaranteed to detect all PPE violations

    Use only for research, education, and demonstrations.
    """
    )

    st.subheader("Citation")
    st.code(
        """
@article{csa-yolo-2026,
  title={CSA-YOLO: Construction Safety-Aware Detector with
         Worker-PPE Association, Temporal Memory, and Risk Assessment},
  author={Your Name},
  year={2026}
}
        """,
        language="bibtex",
    )


if __name__ == "__main__":
    pass
