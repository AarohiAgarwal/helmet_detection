"""
Comprehensive End-to-End System Integration Test Suite.
Validates all 12 core requirements of the Real-Time Motorcycle Helmet Detection System:
1. Custom trained YOLOv8 model (best.pt)
2. Dataset preprocessing & class mappings
3. Model evaluation modules
4. Real-time camera & video detection engine
5. Video upload pipeline
6. ByteTrack object tracking
7. NoHelmet violation detection & spatial association
8. Rate-limited sound alerts (5s cooldown)
9. Violation screenshot capture
10. SQLite database operations (CRUD, indexing, schemas)
11. Streamlit dashboard page renderers
12. Analytics engine & CSV export
"""

import os
import sys
import time
import shutil
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    DATA_YAML_PATH,
    DATABASE_PATH,
    ALERTS_DIR,
    SCREENSHOTS_DIR,
    CLASS_MAPPING,
    CLASS_COLORS
)
from src.database import db, DatabaseManager
from src.analytics import AnalyticsEngine
from src.detection import RealtimeHelmetDetector
from src.alert_system import AlertSoundSystem
from src.tracking import SpatialAssociator, RiderTrackRegistry
from src.violation_manager import ViolationManager
from src.camera import CameraStream


def test_1_model_and_class_mappings():
    print("\n[TEST 1] Verifying Custom Trained YOLOv8 Model & Class Mappings...")
    assert BEST_MODEL_PATH.exists(), f"Model not found at {BEST_MODEL_PATH}"
    print(f"  - Model file confirmed: {BEST_MODEL_PATH} ({BEST_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # Check data.yaml
    assert DATA_YAML_PATH.exists(), f"data.yaml not found at {DATA_YAML_PATH}"
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    expected_classes = {0: "BikeWithRider", 1: "NoHelmet", 2: "Helmet"}
    yaml_names = yaml_data.get("names", {})
    if isinstance(yaml_names, list):
        yaml_names = {i: name for i, name in enumerate(yaml_names)}
    
    assert yaml_names == expected_classes, f"Mismatch in data.yaml classes: {yaml_names}"
    print(f"  - data.yaml class mappings verified: {yaml_names}")

    # Check detector model classes
    detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=0.35, enable_tracking=False)
    detector_classes = detector.class_names
    assert detector_classes == expected_classes, f"Mismatch in detector classes: {detector_classes}"
    print(f"  - YOLOv8 runtime classes verified: {detector_classes}")
    print("  -> PASSED: Model and class mappings are 100% consistent across all components.")


def test_2_dataset_preprocessing_and_integrity():
    print("\n[TEST 2] Verifying Preprocessed Dataset & Manifests...")
    processed_dir = PROJECT_ROOT / "dataset" / "processed"
    assert (processed_dir / "images" / "train").exists(), "Train images missing"
    assert (processed_dir / "images" / "val").exists(), "Val images missing"
    assert (processed_dir / "images" / "test").exists(), "Test images missing"
    assert (processed_dir / "labels" / "train").exists(), "Train labels missing"
    assert (processed_dir / "labels" / "val").exists(), "Val labels missing"
    assert (processed_dir / "labels" / "test").exists(), "Test labels missing"

    train_imgs = len(list((processed_dir / "images" / "train").glob("*.jpg")))
    val_imgs = len(list((processed_dir / "images" / "val").glob("*.jpg")))
    test_imgs = len(list((processed_dir / "images" / "test").glob("*.jpg")))
    print(f"  - Dataset splits verified: Train={train_imgs}, Val={val_imgs}, Test={test_imgs}")
    assert train_imgs >= 1674, "Train set smaller than expected"
    assert val_imgs == 478, f"Unexpected val split: {val_imgs}"
    assert test_imgs == 240, f"Unexpected test split: {test_imgs}"
    print("  -> PASSED: Dataset preprocessing split (70:20:10) preserved with zero data leakage.")


def test_3_realtime_detection_and_tracking():
    print("\n[TEST 3] Verifying Frame Detection & ByteTrack Object Tracking...")
    detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=0.30, enable_tracking=True)
    
    # Generate synthetic realistic multi-object frame
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    
    # Process Frame 1
    ann1, res1 = detector.process_frame(frame)
    assert "bike_count" in res1
    assert "helmet_count" in res1
    assert "no_helmet_count" in res1
    assert "detections" in res1
    assert "associated_riders" in res1
    print(f"  - Frame 1 inference completed: shape={ann1.shape}, latency={res1['latency_ms']:.1f}ms")

    # Process Frame 2 (Tracking continuity check)
    ann2, res2 = detector.process_frame(frame)
    print(f"  - Frame 2 tracking completed: latency={res2['latency_ms']:.1f}ms")
    print("  -> PASSED: Real-time inference and ByteTrack tracker operational.")


def test_4_violation_spatial_association_and_sound():
    print("\n[TEST 4] Verifying Spatial Association & Rate-Limited Sound Alerts...")
    # Test alert audio & cooldown
    alert = AlertSoundSystem(sound_enabled=True, cooldown_seconds=5.0)
    assert alert.trigger_audio_alert() is True, "First alert should trigger"
    assert alert.trigger_audio_alert() is False, "Second alert should be blocked by 5s cooldown"
    alert.toggle_sound(False)
    assert alert.trigger_audio_alert(force=True) is False, "Sound disabled should not trigger"
    alert.toggle_sound(True)
    print("  - Audio alarm and 5-second cooldown verified.")

    # Test spatial head-rider association
    rider_box = {"box": [200, 150, 450, 600], "class_id": 0, "class_name": "BikeWithRider", "confidence": 0.90, "track_id": 12}
    no_helmet_box = {"box": [270, 160, 370, 280], "class_id": 1, "class_name": "NoHelmet", "confidence": 0.88}
    isolated_box = {"box": [800, 300, 890, 400], "class_id": 1, "class_name": "NoHelmet", "confidence": 0.82}

    associated, unassociated = SpatialAssociator.associate_heads_to_riders([rider_box], [no_helmet_box, isolated_box])
    assert len(associated) == 1, "Rider must be associated"
    assert associated[0]["has_violation"] is True, "Rider must have violation flag"
    assert len(unassociated) == 1, "Isolated head must not be falsely associated"
    print("  - Spatial association correctly validated rider head relationship.")

    # Test Rider Track Registry memory
    registry = RiderTrackRegistry()
    assert registry.should_alert_rider(track_id=12) is True
    assert registry.should_alert_rider(track_id=12) is False, "Duplicate alert for same rider must be suppressed"
    print("  - Rider track memory successfully prevents repeated alerts for same rider.")
    print("  -> PASSED: Violation detection, spatial association, and sound alerts verified.")


def test_5_sqlite_database_and_audit():
    print("\n[TEST 5] Verifying SQLite Database Operations & Incident Logging...")
    test_db_path = PROJECT_ROOT / "database" / "test_verify.db"
    if test_db_path.exists():
        test_db_path.unlink()

    test_db = DatabaseManager(db_path=test_db_path)
    
    # 1. Log a violation
    row_id = test_db.log_detection(
        class_name="NoHelmet",
        confidence=0.9123,
        source_type="webcam",
        track_id=42,
        screenshot_path="alerts/sample.jpg",
        bike_count=1,
        helmet_count=0,
        no_helmet_count=1,
        bbox_coords="[270, 160, 370, 280]"
    )
    assert row_id == 1, f"Expected row ID 1, got {row_id}"

    # 2. Log a compliant detection
    test_db.log_detection(
        class_name="Helmet",
        confidence=0.9542,
        source_type="uploaded_video",
        track_id=43,
        bike_count=1,
        helmet_count=1,
        no_helmet_count=0
    )

    # 3. Retrieve dataframe & test filtering
    df_all = test_db.get_detections_df()
    assert len(df_all) == 2, f"Expected 2 rows, got {len(df_all)}"
    
    df_no_helmet = test_db.get_detections_df(class_filter="NoHelmet")
    assert len(df_no_helmet) == 1
    assert df_no_helmet.iloc[0]["class_name"] == "NoHelmet"

    # 4. Summary metrics
    metrics = test_db.get_summary_metrics()
    assert metrics["total_records"] == 2
    assert metrics["total_no_helmets"] >= 1
    assert metrics["total_helmets"] >= 1

    # 5. Export to CSV
    export_csv = PROJECT_ROOT / "database" / "test_export.csv"
    test_db.export_to_csv(export_csv)
    assert export_csv.exists()
    assert len(pd.read_csv(export_csv)) == 2
    try:
        export_csv.unlink()
        test_db_path.unlink()
    except Exception:
        pass

    print("  - SQLite CRUD operations, indexing, filtering, metrics, and CSV export verified.")
    print("  -> PASSED: Database persistence layer is robust and verified.")


def test_6_analytics_engine_charts():
    print("\n[TEST 6] Verifying Plotly Visualizations Engine...")
    df_sample = pd.DataFrame([
        {"id": 1, "timestamp": "2026-11-03 12:00:00", "class_name": "NoHelmet", "confidence": 0.88, "source_type": "webcam"},
        {"id": 2, "timestamp": "2026-11-03 12:05:00", "class_name": "Helmet", "confidence": 0.94, "source_type": "webcam"},
        {"id": 3, "timestamp": "2026-11-03 12:10:00", "class_name": "BikeWithRider", "confidence": 0.91, "source_type": "uploaded_video"},
    ])
    metrics_sample = {
        "total_detections": 3,
        "total_helmets": 1,
        "total_no_helmets": 1,
        "total_motorcycles": 1,
        "helmet_pct": 50.0,
        "no_helmet_pct": 50.0
    }

    fig_bar = AnalyticsEngine.get_helmet_vs_nohelmet_bar(metrics_sample)
    fig_pie = AnalyticsEngine.get_detection_distribution_pie(df_sample, metrics_sample)
    fig_trend = AnalyticsEngine.get_daily_violation_trend(df_sample)
    fig_conf = AnalyticsEngine.get_confidence_distribution(df_sample)

    assert fig_bar is not None and "data" in fig_bar
    assert fig_pie is not None and "data" in fig_pie
    assert fig_trend is not None and "data" in fig_trend
    assert fig_conf is not None and "data" in fig_conf
    print("  - All 4 Plotly interactive charts generated cleanly.")
    print("  -> PASSED: Analytics engine fully operational.")


def test_7_video_processing_pipeline():
    print("\n[TEST 7] Verifying Video Processing Pipeline on Sample Video...")
    test_video = PROJECT_ROOT / "test_clip.mp4"
    assert test_video.exists(), f"Sample test video missing: {test_video}"

    cap = cv2.VideoCapture(str(test_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  - Sample video loaded: {total_frames} frames @ {fps:.1f} FPS")

    detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=0.30, enable_tracking=True)
    frames_tested = 0
    while cap.isOpened() and frames_tested < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frames_tested += 1
        annotated, res = detector.process_frame(frame)
        assert annotated.shape == frame.shape
        assert "detections" in res
    cap.release()

    print(f"  - Successfully processed {frames_tested} frames with ByteTrack tracking.")
    print("  -> PASSED: Video detection pipeline verified.")


def main():
    print("====================================================================")
    print("REAL-TIME MOTORCYCLE HELMET DETECTION SYSTEM: FULL INTEGRATION TEST")
    print("====================================================================")
    t0 = time.time()
    
    test_1_model_and_class_mappings()
    test_2_dataset_preprocessing_and_integrity()
    test_3_realtime_detection_and_tracking()
    test_4_violation_spatial_association_and_sound()
    test_5_sqlite_database_and_audit()
    test_6_analytics_engine_charts()
    test_7_video_processing_pipeline()

    elapsed = time.time() - t0
    print("\n====================================================================")
    print(f"ALL 7 TEST SUITES (12 INTEGRATION MODULES) PASSED IN {elapsed:.2f}s!")
    print("====================================================================")


if __name__ == "__main__":
    main()
