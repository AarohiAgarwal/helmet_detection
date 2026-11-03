"""
Real-Time Helmet Detection & Intelligent No-Helmet Alert Application.
Streams live camera or video input, tracks riders with ByteTrack,
geometrically associates helmets to riders, triggers rate-limited audio alerts,
and captures evidence screenshots.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, SCREENSHOTS_DIR, ALERTS_DIR
from src.camera import CameraStream
from src.detection import RealtimeHelmetDetector
from src.alert_system import AlertSoundSystem
from src.violation_manager import ViolationManager
from src.tracking import RiderTrackRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Motorcycle Helmet Detection Application")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. '0') or video file path (.mp4, .avi)")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom trained best.pt model")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Initial confidence threshold (default: 0.40)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Alert cooldown interval in seconds (default: 5.0)")
    parser.add_argument("--no-sound", action="store_true", help="Disable audio alarm sounds")
    parser.add_argument("--record", type=str, default=None, help="Optional output path to record video (.mp4)")
    parser.add_argument("--test", type=int, default=0, help="Test mode: process N frames headlessly and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 68)
    print("STAGE 6: REAL-TIME HELMET DETECTION & INTELLIGENT ALERT SYSTEM")
    print("=" * 68)
    print(f"Source:         {args.source}")
    print(f"Confidence:     {args.conf:.2f}")
    print(f"Alert Cooldown: {args.cooldown:.1f} seconds")
    print(f"Audio Enabled:  {not args.no_sound}")

    # 1. Initialize Custom-Trained Detector
    try:
        detector = RealtimeHelmetDetector(
            weights_path=Path(args.weights) if args.weights else None,
            conf_threshold=args.conf,
            enable_tracking=True
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Please complete Stage 3 training to generate 'models/best.pt'.")
        sys.exit(1)

    # 2. Initialize Alert & Violation Infrastructure
    alert_sound = AlertSoundSystem(sound_enabled=not args.no_sound, cooldown_seconds=args.cooldown)
    violation_mgr = ViolationManager()
    track_registry = RiderTrackRegistry()

    # 3. Initialize Camera Stream
    print(f"\n[Camera] Opening video stream: {args.source}...")
    try:
        cam = CameraStream(source=args.source)
        cam.start()
    except Exception as e:
        print(f"\n[Camera Error] Could not access video source '{args.source}': {e}")
        print("\nTroubleshooting tips:")
        print("  1. Verify your webcam is plugged in and recognized in Windows Device Manager.")
        print("  2. Check Windows Settings -> Privacy & Security -> Camera (Allow apps to access your camera).")
        print("  3. If using an external USB camera, try '--source 1' instead of 0.")
        print("  4. To test without a physical webcam, pass a video file: 'python webcam.py --source test_clip.mp4'.")
        sys.exit(1)

    print("[Camera] Stream established successfully.")
    print(f"         Resolution: {cam.width}x{cam.height} @ ~{cam.fps:.1f} FPS")

    # 4. Optional Video Recorder
    writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rec_path), fourcc, cam.fps or 25.0, (cam.width, cam.height))
        print(f"[Recorder] Recording stream to: {rec_path}")

    # 5. Interactive Keyboard Controls
    if not args.test:
        print("\n[Controls]")
        print("  'q' or 'ESC': Stop camera and exit")
        print("  '+' or '=':   Increase confidence threshold (+0.05)")
        print("  '-' or '_':   Decrease confidence threshold (-0.05)")
        print("  'm' or 'a':   Toggle warning sound (Mute / Unmute)")
        print("  's':          Save manual snapshot to screenshots/")
        print("  'SPACE':      Pause / Resume playback")
        print("-" * 68)

    window_name = "Real-Time Helmet Detection & Intelligent Alert System"
    fps_smooth = 0.0
    prev_time = time.time()
    frame_count = 0
    paused = False

    try:
        while cam.running:
            if not paused:
                ret, frame = cam.read()
                if not ret or frame is None:
                    if cam.is_file:
                        print("[Stream] Video file playback completed.")
                    time.sleep(0.01)
                    break

                # FPS Calculation
                now = time.time()
                dt = now - prev_time
                prev_time = now
                inst_fps = 1.0 / (dt + 1e-6)
                fps_smooth = (0.85 * fps_smooth) + (0.15 * inst_fps) if fps_smooth > 0 else inst_fps

                # YOLOv8 + ByteTrack Detection & Rider-Head Spatial Association
                annotated, results_data = detector.process_frame(frame)

                # Intelligent Alert & Evidence Triggering
                if results_data["violation"]:
                    # Check for verified violator riders
                    for rider in results_data.get("associated_riders", []):
                        if rider.get("has_violation", False):
                            track_id = rider.get("track_id")
                            # Rule: Avoid playing alerts repeatedly for the same tracked rider!
                            if track_registry.should_alert_rider(track_id):
                                # 1. Play Warning Sound (guarded by 5s cooldown)
                                alert_sound.trigger_audio_alert()

                                # 2. Save Evidence Screenshot and Log to SQLite & CSV
                                no_helmet_head = next((h for h in rider["heads"] if h["class_id"] == 1), None)
                                conf = no_helmet_head["confidence"] if no_helmet_head else rider["confidence"]
                                head_box = no_helmet_head["box"] if no_helmet_head else rider["box"]

                                violation_mgr.record_violation(
                                    frame=annotated,
                                    confidence=conf,
                                    track_id=track_id,
                                    association_status=f"Verified (Rider Track #{track_id})",
                                    bbox=head_box,
                                    bike_count=results_data["bike_count"],
                                    no_helmet_count=results_data["no_helmet_count"]
                                )

                    # Check for unassociated NoHelmet detections (e.g. rider head without tracked bike box)
                    for uhead in results_data.get("unassociated_heads", []):
                        if uhead.get("class_id") == 1:
                            if track_registry.should_alert_rider(track_id=None):
                                alert_sound.trigger_audio_alert()
                                violation_mgr.record_violation(
                                    frame=annotated,
                                    confidence=uhead["confidence"],
                                    track_id=None,
                                    association_status="Unassociated Head (Unverified)",
                                    bbox=uhead["box"],
                                    bike_count=results_data["bike_count"],
                                    no_helmet_count=results_data["no_helmet_count"]
                                )

                # Telemetry HUD
                display_frame = detector.draw_telemetry_hud(
                    annotated,
                    results_data,
                    fps=fps_smooth,
                    conf_threshold=detector.conf_threshold,
                    sound_enabled=alert_sound.sound_enabled
                )

                if writer:
                    writer.write(display_frame)

                frame_count += 1
                if args.test and frame_count >= args.test:
                    print(f"[Test Mode] Processed {frame_count} frames successfully. Exiting.")
                    break

            if not args.test:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), 27]:
                    print("[User] Stop signal received. Exiting...")
                    break
                elif key in [ord('+'), ord('=')]:
                    detector.set_confidence(detector.conf_threshold + 0.05)
                    print(f"[Controls] Confidence threshold set to: {detector.conf_threshold:.2f}")
                elif key in [ord('-'), ord('_')]:
                    detector.set_confidence(detector.conf_threshold - 0.05)
                    print(f"[Controls] Confidence threshold set to: {detector.conf_threshold:.2f}")
                elif key in [ord('m'), ord('a')]:
                    new_state = alert_sound.toggle_sound()
                    print(f"[Controls] Audio Alarm: {'ENABLED' if new_state else 'MUTED'}")
                elif key == ord('s'):
                    snap_name = f"manual_snapshot_{int(time.time())}.jpg"
                    snap_path = SCREENSHOTS_DIR / snap_name
                    cv2.imwrite(str(snap_path), display_frame)
                    print(f"[Snapshot] Saved evidence frame to: {snap_path}")
                elif key == ord(' '):
                    paused = not paused
                    print("[Controls] Playback paused." if paused else "[Controls] Playback resumed.")

    finally:
        print("\n[Cleanup] Releasing camera and video resources...")
        cam.release()
        if writer:
            writer.release()
        if not args.test:
            cv2.destroyAllWindows()
        print("[Cleanup] All resources released cleanly.")


if __name__ == "__main__":
    main()
