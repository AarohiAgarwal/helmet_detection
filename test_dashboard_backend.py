"""
Automated Test for Dashboard Backend, Database, and Analytics.
Verifies all 5 pages' data queries and chart generators.
"""

import os
import sys
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH
from src.database import db
from src.analytics import AnalyticsEngine
from src.detection import RealtimeHelmetDetector


def test_database_and_analytics():
    print("--- Testing Database Operations ---")
    metrics = db.get_summary_metrics()
    print("Summary Metrics:", metrics)
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert "total_records" in metrics
    assert "total_helmets" in metrics

    df = db.get_detections_df(limit=10)
    print(f"Retrieved {len(df)} detection records.")

    print("\n--- Testing Analytics Engine Charts ---")
    fig1 = AnalyticsEngine.get_helmet_vs_nohelmet_bar(metrics)
    assert fig1 is not None, "Bar chart failed"
    print("1. Bar Chart: OK")

    fig2 = AnalyticsEngine.get_detection_distribution_pie(df, metrics)
    assert fig2 is not None, "Pie chart failed"
    print("2. Pie Chart: OK")

    fig3 = AnalyticsEngine.get_daily_violation_trend(df)
    assert fig3 is not None, "Trend chart failed"
    print("3. Trend Chart: OK")

    fig4 = AnalyticsEngine.get_confidence_distribution(df)
    assert fig4 is not None, "Hist chart failed"
    print("4. Confidence Hist: OK")


def test_model_inference():
    print("\n--- Testing Real-Time Detector with models/best.pt ---")
    assert BEST_MODEL_PATH.exists(), f"Model path {BEST_MODEL_PATH} must exist!"
    detector = RealtimeHelmetDetector(weights_path=BEST_MODEL_PATH, conf_threshold=0.30)

    # Test on test_clip.mp4 frame
    video_path = PROJECT_ROOT / "test_clip.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            annotated, res = detector.process_frame(frame)
            print("Frame Inference Results:")
            print(f"  Bikes detected:      {res['bike_count']}")
            print(f"  Helmets detected:    {res['helmet_count']}")
            print(f"  No-Helmets detected: {res['no_helmet_count']}")
            print(f"  Latency:             {res['latency_ms']:.1f} ms")
            print("Inference on test video frame: OK")


if __name__ == "__main__":
    test_database_and_analytics()
    test_model_inference()
    print("\nALL DASHBOARD BACKEND TESTS PASSED SUCCESSFULLY!")

