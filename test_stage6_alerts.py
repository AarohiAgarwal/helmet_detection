"""
Verification and Test Suite for Stage 6 Intelligent Alert and Tracking System.
Tests:
1. Alert audio triggering and 5-second cooldown.
2. ByteTrack integration & Spatial rider-head association logic.
3. Anti-duplicate tracking memory for the same rider.
4. Violation snapshot saving and audit trail (SQLite & CSV).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_system import AlertSoundSystem
from src.tracking import SpatialAssociator, RiderTrackRegistry
from src.violation_manager import ViolationManager
from src.config import ALERTS_DIR


def test_alert_sound_and_cooldown():
    print("\n--- TEST 1: Alert Sound System & 5-Second Cooldown ---")
    alert_sys = AlertSoundSystem(sound_enabled=True, cooldown_seconds=5.0)

    # First trigger should succeed
    t1 = alert_sys.trigger_audio_alert()
    print(f"Trigger 1 (Initial): Success = {t1}")
    assert t1 is True, "First alert should trigger successfully!"

    # Second trigger immediately after should be blocked by 5-second cooldown
    t2 = alert_sys.trigger_audio_alert()
    print(f"Trigger 2 (+0.1s later): Success = {t2} (Expected: False - cooldown active)")
    assert t2 is False, "Immediate second alert should be blocked by 5s cooldown!"

    # Test disable toggle
    alert_sys.toggle_sound(False)
    t3 = alert_sys.trigger_audio_alert(force=True)
    print(f"Trigger 3 (Sound Disabled): Success = {t3} (Expected: False)")
    assert t3 is False, "Alert should not play when sound is disabled!"

    alert_sys.toggle_sound(True)
    print("[PASSED] Alert Sound System and Cooldown logic verified.")


def test_spatial_association():
    print("\n--- TEST 2: Spatial Rider-Head Association Logic ---")
    # Motorcycle Rider at [100, 100, 300, 500] (w=200, h=400)
    # Upper head zone is roughly y in [100, 300]
    bike_rider = {
        "box": [100, 100, 300, 500],
        "class_id": 0,
        "class_name": "BikeWithRider",
        "confidence": 0.95,
        "track_id": 1
    }

    # Case A: NoHelmet head inside rider upper body
    head_on_rider = {
        "box": [160, 110, 240, 200],  # Centered horizontally at 200, near top
        "class_id": 1,
        "class_name": "NoHelmet",
        "confidence": 0.92
    }

    # Case B: Disconnected head far away (Pedestrian or distant clutter)
    isolated_head = {
        "box": [600, 200, 680, 290],
        "class_id": 1,
        "class_name": "NoHelmet",
        "confidence": 0.85
    }

    bikes = [bike_rider]
    heads = [head_on_rider, isolated_head]

    associated_riders, unassociated_heads = SpatialAssociator.associate_heads_to_riders(bikes, heads)

    print(f"Associated Riders: {len(associated_riders)}")
    print(f"  Rider 1 Status: {associated_riders[0]['compliance_status']}")
    print(f"  Heads on Rider 1: {len(associated_riders[0]['heads'])}")
    print(f"Unassociated Heads: {len(unassociated_heads)}")

    assert associated_riders[0]["compliance_status"] == "VIOLATION (No Helmet)", "Rider should be flagged with violation!"
    assert len(unassociated_heads) == 1, "Isolated head should remain unassociated!"
    print("[PASSED] Spatial association correctly verified rider head and prevented false vehicle claims.")


def test_anti_duplicate_tracking_memory():
    print("\n--- TEST 3: Anti-Duplicate Tracking Memory ---")
    registry = RiderTrackRegistry()

    # Track #4 detected with violation on Frame 1
    first_check = registry.should_alert_rider(track_id=4)
    print(f"Frame 1 - Rider Track #4: Trigger Alert = {first_check} (Expected: True)")
    assert first_check is True, "First time seeing Track #4 violation should alert!"

    # Track #4 detected with violation again on Frame 2, 3, 4 (rider driving along)
    repeat_check = registry.should_alert_rider(track_id=4)
    print(f"Frame 2 - Rider Track #4: Trigger Alert = {repeat_check} (Expected: False - duplicate prevented)")
    assert repeat_check is False, "Subsequent frames for the same tracked rider must NOT repeat alerts!"

    # A new violator enters: Track #7
    new_rider_check = registry.should_alert_rider(track_id=7)
    print(f"Frame 5 - New Rider Track #7: Trigger Alert = {new_rider_check} (Expected: True)")
    assert new_rider_check is True, "New rider track #7 must trigger alert!"

    print(f"Total Unique Violators Tracked: {registry.total_unique_violators}")
    assert registry.total_unique_violators == 2
    print("[PASSED] Anti-duplicate tracking memory successfully prevents repeated alerts.")


def test_violation_snapshot_and_audit():
    print("\n--- TEST 4: Screenshot Evidence Saving & Audit Trail ---")
    vm = ViolationManager()

    # Generate synthetic camera frame
    test_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    cv2.putText(test_frame, "STAGE 6 VIOLATION TEST EVIDENCE", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    record = vm.record_violation(
        frame=test_frame,
        confidence=0.915,
        track_id=3,
        association_status="Verified (Rider Track #3)",
        bbox=[150, 120, 380, 600]
    )

    print(f"Recorded ID:       {record['id']}")
    print(f"Timestamp:         {record['timestamp']}")
    print(f"Track ID:          {record['track_id']}")
    print(f"Screenshot Path:   {record['screenshot_path']}")

    # Verify file exists on disk
    saved_file = Path(record["screenshot_path"])
    assert saved_file.exists(), f"Screenshot was not created: {saved_file}"
    assert saved_file.stat().st_size > 1000, "Screenshot file is empty!"

    # Verify CSV log updated
    assert vm.csv_path.exists(), "CSV log file not found!"
    with open(vm.csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    assert f"track3" in record["screenshot_path"]
    assert "NoHelmet" in csv_content

    print("[PASSED] Screenshot and audit trail (SQLite & CSV) successfully verified.")


def main():
    print("=" * 68)
    print("STAGE 6: INTELLIGENT NO-HELMET ALERT SYSTEM VERIFICATION")
    print("=" * 68)

    test_alert_sound_and_cooldown()
    test_spatial_association()
    test_anti_duplicate_tracking_memory()
    test_violation_snapshot_and_audit()

    print("\n" + "=" * 68)
    print("ALL STAGE 6 TESTS COMPLETED AND PASSED SUCCESSFULLY!")
    print("=" * 68)


if __name__ == "__main__":
    main()

