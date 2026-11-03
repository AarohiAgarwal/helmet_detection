"""
Violation Management and Evidence Audit System.
Captures snapshots of confirmed NoHelmet violations, manages rate-limiting,
and logs structured records to SQLite and CSV.
"""

import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERTS_DIR, SCREENSHOTS_DIR, DATABASE_PATH
from src.database import DatabaseManager


class ViolationManager:
    """
    Manages capturing evidence screenshots, recording violation metadata,
    and syncing audit logs with SQLite and flat CSV files.
    """

    def __init__(
        self,
        alerts_dir: Path = ALERTS_DIR,
        screenshots_dir: Path = SCREENSHOTS_DIR,
        csv_path: Optional[Path] = None
    ):
        self.alerts_dir = Path(alerts_dir)
        self.screenshots_dir = Path(screenshots_dir)
        self.csv_path = Path(csv_path) if csv_path else (self.alerts_dir / "violations_log.csv")

        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseManager(DATABASE_PATH)
        self._init_csv()

    def _init_csv(self):
        """Initializes CSV header if file does not exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID",
                    "Timestamp",
                    "Track_ID",
                    "Class_Name",
                    "Confidence",
                    "Association_Status",
                    "Screenshot_Path",
                    "BBox_XYXY"
                ])

    def record_violation(
        self,
        frame: np.ndarray,
        confidence: float,
        track_id: Optional[int] = None,
        association_status: str = "Verified",
        bbox: Optional[List[int]] = None,
        bike_count: int = 1,
        no_helmet_count: int = 1
    ) -> Dict[str, Any]:
        """
        Saves snapshot evidence image and registers violation entry in SQLite and CSV.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

        # Save Evidence Image
        track_tag = f"track{track_id}_" if track_id is not None else ""
        filename = f"violation_{track_tag}{file_timestamp}.jpg"
        save_path = self.alerts_dir / filename
        cv2.imwrite(str(save_path), frame)

        # Log into SQLite
        db_id = self.db.log_violation(
            violation_type="NoHelmet",
            confidence=confidence,
            bike_count=bike_count,
            no_helmet_count=no_helmet_count,
            screenshot_path=str(save_path)
        )

        # Log into CSV
        bbox_str = str(bbox) if bbox else ""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                db_id,
                timestamp_str,
                track_id if track_id is not None else "N/A",
                "NoHelmet",
                f"{confidence:.4f}",
                association_status,
                str(save_path),
                bbox_str
            ])

        print(f"[Violation Manager] Recorded infraction #{db_id} (Track #{track_id}): {save_path.name}")

        return {
            "id": db_id,
            "timestamp": timestamp_str,
            "track_id": track_id,
            "class_name": "NoHelmet",
            "confidence": confidence,
            "association_status": association_status,
            "screenshot_path": str(save_path),
            "bbox": bbox
        }

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent violation records from SQLite database."""
        return self.db.get_recent_violations(limit=limit)


if __name__ == "__main__":
    vm = ViolationManager()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = vm.record_violation(dummy_frame, confidence=0.88, track_id=5, association_status="Verified (Rider Track #5)")
    print("Test Record:", res)

