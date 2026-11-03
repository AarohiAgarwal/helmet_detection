"""
Streamlit Dashboard Page Renderers for Traffic Helmet Monitoring.
Implements the 5 pages: Overview, Live Detection, Violation History, Analytics, and Video Detection.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, SCREENSHOTS_DIR, ALERTS_DIR
from src.database import db
from src.analytics import AnalyticsEngine
from src.detection import RealtimeHelmetDetector
from src.alert_system import AlertSoundSystem
from src.violation_manager import ViolationManager
from src.tracking import RiderTrackRegistry


def render_overview_page():
    """PAGE 1: Dashboard Overview with KPI Cards and System Metrics."""
    st.markdown("## 🚦 Traffic Safety Monitoring Overview")
    st.markdown("Real-time compliance intelligence and aggregated violation telemetry.")

    metrics = db.get_summary_metrics()

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Detections",
            value=f"{metrics['total_detections']:,}",
            help="Total cumulative object detections recorded."
        )
    with c2:
        st.metric(
            label="Total Motorcycles (BikeWithRider)",
            value=f"{metrics['total_motorcycles']:,}",
            help="Motorcycle vehicle and rider profiles tracked."
        )
    with c3:
        st.metric(
            label="Helmets Detected (Compliant)",
            value=f"{metrics['total_helmets']:,}",
            delta=f"{metrics['helmet_pct']:.1f}% Compliance",
            delta_color="normal"
        )
    with c4:
        st.metric(
            label="No-Helmet Detected (Violations)",
            value=f"{metrics['total_no_helmets']:,}",
            delta=f"{metrics['no_helmet_pct']:.1f}% Infractions",
            delta_color="inverse"
        )

    st.markdown("---")

    # Overview Sub-panels
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("Compliance Snapshot")
        fig_bar = AnalyticsEngine.get_helmet_vs_nohelmet_bar(metrics)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("System Infrastructure Status")
        model_exists = BEST_MODEL_PATH.exists()

        status_color = "#00CC96" if model_exists else "#FFA500"
        status_text = "READY & LOADED" if model_exists else "PENDING TRAINING"

        st.markdown(f"""
        <div style="background-color: #1e1e24; padding: 18px; border-radius: 8px; border-left: 5px solid {status_color};">
            <h4 style="margin:0; color: {status_color};">YOLOv8 Engine: {status_text}</h4>
            <p style="margin: 8px 0 0 0; color: #ccc;"><b>Model Path:</b> <code>{BEST_MODEL_PATH}</code></p>
            <p style="margin: 4px 0 0 0; color: #ccc;"><b>Classes:</b> 0: BikeWithRider | 1: NoHelmet | 2: Helmet</p>
            <p style="margin: 4px 0 0 0; color: #ccc;"><b>Database:</b> <code>database/violations.db</code> ({metrics['total_records']} audit rows)</p>
            <p style="margin: 4px 0 0 0; color: #ccc;"><b>Evidence Storage:</b> <code>alerts/</code> & <code>screenshots/</code></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Recent Incidents Preview")
        recent_df = db.get_detections_df(limit=3)
        if not recent_df.empty:
            for _, r in recent_df.iterrows():
                st.markdown(f"- **{r['timestamp']}** — `{r['class_name']}` (Conf: {r['confidence']:.2f}, Source: {r['source_type']})")
        else:
            st.info("No violations logged yet. Run live webcam or video detection to capture incidents.")


def render_live_detection_page():
    """PAGE 2: Live Webcam Detection with interactive controls and alert audio."""
    st.markdown("## 📹 Live Webcam Helmet Detection")
    st.markdown("High-FPS surveillance with ByteTrack spatial association and rate-limited audio alerts.")

    # Controls row
    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 1.2])

    with col_c1:
        camera_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    with col_c2:
        conf_thresh = st.slider("Confidence", min_value=0.10, max_value=0.95, value=0.40, step=0.05)
    with col_c3:
        sound_enabled = st.checkbox("Enable Audio Alarm", value=True)
    with col_c4:
        st.write("")  # spacing
        st.write("")
        run_camera = st.toggle("Start Camera", value=False)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if not run_camera:
        frame_placeholder.info("Camera is currently stopped. Toggle 'Start Camera' above to initiate live surveillance.")
        return

    # Check model existence
    if not BEST_MODEL_PATH.exists():
        st.error("Custom trained model 'models/best.pt' not found. Please complete Stage 3 training first.")
        return

    try:
        detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=conf_thresh, enable_tracking=True)
        alert_sound = AlertSoundSystem(sound_enabled=sound_enabled, cooldown_seconds=5.0)
        violation_mgr = ViolationManager()
        track_registry = RiderTrackRegistry()

        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(f"Could not open webcam index {camera_index}. Please verify camera connection and Windows privacy permissions.")
            return

        prev_t = time.time()
        fps_val = 0.0

        stop_button = st.button("⏹️ Emergency Stop")

        while run_camera and not stop_button:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Failed to receive frame from webcam. Stream stopped.")
                break

            now = time.time()
            fps_val = 0.8 * fps_val + 0.2 * (1.0 / (now - prev_t + 1e-6))
            prev_t = now

            # YOLOv8 + ByteTrack Detection
            detector.conf_threshold = conf_thresh
            annotated, res_data = detector.process_frame(frame)

            # Check Violations and trigger rate-limited sound + screenshot
            if res_data["violation"]:
                for r in res_data.get("associated_riders", []):
                    if r.get("has_violation", False):
                        tid = r.get("track_id")
                        if track_registry.should_alert_rider(tid):
                            alert_sound.trigger_audio_alert()
                            no_helmet_h = next((h for h in r["heads"] if h["class_id"] == 1), None)
                            conf = no_helmet_h["confidence"] if no_helmet_h else r["confidence"]
                            head_box = no_helmet_h["box"] if no_helmet_h else r["box"]

                            rec = violation_mgr.record_violation(
                                frame=annotated,
                                confidence=conf,
                                track_id=tid,
                                association_status=f"Verified (Rider Track #{tid})",
                                bbox=head_box,
                                bike_count=res_data["bike_count"],
                                no_helmet_count=res_data["no_helmet_count"]
                            )

                            # Log to SQLite
                            db.log_detection(
                                class_name="NoHelmet",
                                confidence=conf,
                                source_type="webcam",
                                track_id=tid,
                                screenshot_path=rec.get("screenshot_path"),
                                bike_count=res_data["bike_count"],
                                helmet_count=res_data["helmet_count"],
                                no_helmet_count=res_data["no_helmet_count"],
                                bbox_coords=str(head_box)
                            )

            # Telemetry HUD
            hud_frame = detector.draw_telemetry_hud(
                annotated,
                res_data,
                fps=fps_val,
                conf_threshold=conf_thresh,
                sound_enabled=sound_enabled
            )

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            status_placeholder.markdown(
                f"**FPS:** `{fps_val:.1f}` | **Bikes:** `{res_data['bike_count']}` | "
                f"**Helmets:** `{res_data['helmet_count']}` | **Violations:** `{res_data['no_helmet_count']}`"
            )

        cap.release()

    except Exception as e:
        st.error(f"Stream error: {e}")


def render_violation_history_page():
    """PAGE 3: Violation History with Search, Filter, Lightbox, and CSV Export."""
    st.markdown("## 📋 Violation History & Incident Logs")
    st.markdown("Searchable database of recorded traffic infractions with screenshot evidence.")

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        search = st.text_input("🔍 Search (Timestamp, Class, or Path):", value="")
    with col2:
        class_filter = st.selectbox("Filter Class:", ["All", "NoHelmet", "Helmet", "BikeWithRider"])
    with col3:
        source_filter = st.selectbox("Source Type:", ["All", "webcam", "uploaded_video"])

    df = db.get_detections_df(
        class_filter=class_filter,
        source_filter=source_filter,
        search_query=search,
        limit=100
    )

    # Export to CSV
    export_col, count_col = st.columns([1, 2])
    with export_col:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Records (CSV)",
            data=csv_bytes,
            file_name="helmet_violations_export.csv",
            mime="text/csv"
        )
    with count_col:
        st.info(f"Displaying **{len(df)}** incident records matching filters.")

    if not df.empty:
        # Table of records
        display_cols = ["id", "timestamp", "class_name", "confidence", "track_id", "source_type", "screenshot_path"]
        st.dataframe(df[display_cols], use_container_width=True)

        st.subheader("🖼️ Evidence Snapshot Inspector")
        valid_screenshots = df[df["screenshot_path"].notna() & (df["screenshot_path"] != "")]

        if not valid_screenshots.empty:
            sel_id = st.selectbox("Select Incident ID to inspect photo evidence:", valid_screenshots["id"].tolist())
            selected_row = valid_screenshots[valid_screenshots["id"] == sel_id].iloc[0]

            snap_path = Path(selected_row["screenshot_path"])
            col_img, col_details = st.columns([1.5, 1])

            with col_img:
                if snap_path.exists():
                    st.image(str(snap_path), caption=f"Evidence for Incident #{sel_id} ({selected_row['class_name']})", use_container_width=True)
                else:
                    st.warning(f"Screenshot file not found at: {snap_path}")

            with col_details:
                st.markdown(f"""
                ### Incident #{sel_id} Details
                - **Timestamp:** `{selected_row['timestamp']}`
                - **Class:** `{selected_row['class_name']}`
                - **Confidence Score:** `{selected_row['confidence']:.4f}`
                - **Tracked Object ID:** `{selected_row['track_id']}`
                - **Source:** `{selected_row['source_type']}`
                - **Evidence File:** `.../{snap_path.name}`
                """)
        else:
            st.info("No screenshot snapshots attached to the current filtered records.")
    else:
        st.info("No detection records found in database matching criteria.")


def render_analytics_page():
    """PAGE 4: Interactive Plotly Visualizations & Trends."""
    st.markdown("## 📊 Traffic Compliance Analytics")
    st.markdown("Interactive analytical dashboards generated from real SQLite detection telemetry.")

    df = db.get_detections_df()
    metrics = db.get_summary_metrics()

    # Chart Grid 1
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.plotly_chart(AnalyticsEngine.get_helmet_vs_nohelmet_bar(metrics), use_container_width=True)
    with r1_col2:
        st.plotly_chart(AnalyticsEngine.get_detection_distribution_pie(df, metrics), use_container_width=True)

    # Chart Grid 2
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.plotly_chart(AnalyticsEngine.get_daily_violation_trend(df), use_container_width=True)
    with r2_col2:
        st.plotly_chart(AnalyticsEngine.get_confidence_distribution(df), use_container_width=True)

    # Class-wise statistics table
    st.subheader("Class-Wise Detection Statistics")
    if not df.empty and "class_name" in df.columns:
        summary_tbl = df.groupby("class_name").agg(
            Total_Incidents=("id", "count"),
            Average_Confidence=("confidence", "mean"),
            Min_Confidence=("confidence", "min"),
            Max_Confidence=("confidence", "max")
        ).reset_index()
        summary_tbl["Average_Confidence"] = summary_tbl["Average_Confidence"].round(4)
        summary_tbl["Min_Confidence"] = summary_tbl["Min_Confidence"].round(4)
        summary_tbl["Max_Confidence"] = summary_tbl["Max_Confidence"].round(4)
        st.dataframe(summary_tbl, use_container_width=True)
    else:
        st.info("No recorded detection data to aggregate. Detections will appear as live traffic is processed.")


def render_video_detection_page():
    """PAGE 5: Video Detection on Uploaded Clips."""
    st.markdown("## 🎞️ Video File Detection & Analytics")
    st.markdown("Upload a traffic video file (.mp4, .avi, .mov) and run custom YOLOv8 inference with tracking.")

    uploaded_video = st.file_uploader("Select Video File:", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            conf_thresh = st.slider("Detection Confidence", 0.10, 0.95, 0.40, 0.05, key="video_conf")
            start_btn = st.button("🚀 Process Video File")

        if start_btn:
            if not BEST_MODEL_PATH.exists():
                st.error("Custom trained model 'models/best.pt' not found. Please train model first.")
                return

            detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=conf_thresh, enable_tracking=True)
            violation_mgr = ViolationManager()
            track_registry = RiderTrackRegistry()

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            progress_bar = st.progress(0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            processed_count = 0
            video_violations = 0
            video_bikes = 0
            video_helmets = 0

            # Temporary output video path
            out_path = Path(tempfile.gettempdir()) / "annotated_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_count += 1
                    annotated, res_data = detector.process_frame(frame)

                    video_bikes = max(video_bikes, res_data["bike_count"])
                    video_helmets += res_data["helmet_count"]

                    # Check Violations & Log (Rate-limited per rider track)
                    if res_data["violation"]:
                        for r in res_data.get("associated_riders", []):
                            if r.get("has_violation", False):
                                tid = r.get("track_id")
                                if track_registry.should_alert_rider(tid):
                                    video_violations += 1
                                    no_helmet_h = next((h for h in r["heads"] if h["class_id"] == 1), None)
                                    conf = no_helmet_h["confidence"] if no_helmet_h else r["confidence"]
                                    head_box = no_helmet_h["box"] if no_helmet_h else r["box"]

                                    rec = violation_mgr.record_violation(
                                        frame=annotated,
                                        confidence=conf,
                                        track_id=tid,
                                        association_status=f"Verified (Rider Track #{tid})",
                                        bbox=head_box,
                                        bike_count=res_data["bike_count"],
                                        no_helmet_count=res_data["no_helmet_count"]
                                    )

                                    db.log_detection(
                                        class_name="NoHelmet",
                                        confidence=conf,
                                        source_type="uploaded_video",
                                        track_id=tid,
                                        screenshot_path=rec.get("screenshot_path"),
                                        bike_count=res_data["bike_count"],
                                        helmet_count=res_data["helmet_count"],
                                        no_helmet_count=res_data["no_helmet_count"],
                                        bbox_coords=str(head_box)
                                    )

                    hud_frame = detector.draw_telemetry_hud(annotated, res_data, fps=fps, conf_threshold=conf_thresh)
                    writer.write(hud_frame)

                    # Update preview every 10 frames
                    if processed_count % 10 == 0 or processed_count == total_frames:
                        progress = min(1.0, processed_count / total_frames)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {processed_count}/{total_frames} ({progress*100:.1f}%)")
                        rgb_preview = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                        preview_placeholder.image(rgb_preview, channels="RGB", use_container_width=True)

            finally:
                cap.release()
                writer.release()

            st.success(f"Video processing complete! Processed {processed_count} frames.")

            # Summary stats
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Total Frames", processed_count)
            with sc2:
                st.metric("Violations Confirmed", video_violations)
            with sc3:
                st.metric("Helmets Detected", video_helmets)

            st.info("Violation snapshots and SQLite records have been synced to the database.")

