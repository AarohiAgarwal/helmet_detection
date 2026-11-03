"""
Database Layer for Helmet Detection Application.
Manages SQLite operations, auto-creates tables, logs detections with provenance,
and exports audit records without coupling to UI code.
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_PATH, DATABASE_DIR


class DatabaseManager:
    """Encapsulates all SQLite database operations for detections, violations, and analytics."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_database(self):
        """Creates the database schema automatically if it does not exist."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    track_id INTEGER,
                    screenshot_path TEXT,
                    source_type TEXT NOT NULL,
                    bike_count INTEGER DEFAULT 0,
                    helmet_count INTEGER DEFAULT 0,
                    no_helmet_count INTEGER DEFAULT 0,
                    bbox_coords TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_class ON detections(class_name);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_track ON detections(track_id);
            """)

    def log_detection(
        self,
        class_name: str,
        confidence: float,
        source_type: str = "webcam",
        track_id: Optional[int] = None,
        screenshot_path: Optional[str] = None,
        bike_count: int = 0,
        helmet_count: int = 0,
        no_helmet_count: int = 0,
        bbox_coords: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """Stores a confirmed detection record."""
        ts = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    timestamp, class_name, confidence, track_id,
                    screenshot_path, source_type, bike_count,
                    helmet_count, no_helmet_count, bbox_coords
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, class_name, float(confidence), track_id,
                screenshot_path, source_type, int(bike_count),
                int(helmet_count), int(no_helmet_count), bbox_coords
            ))
            return cursor.lastrowid

    def log_violation(
        self,
        violation_type: str = "NoHelmet",
        confidence: float = 0.0,
        bike_count: int = 1,
        no_helmet_count: int = 1,
        screenshot_path: Optional[str] = None,
        source_type: str = "webcam",
        track_id: Optional[int] = None,
        bbox_coords: Optional[str] = None
    ) -> int:
        """Convenience method to log infractions into the detections table."""
        return self.log_detection(
            class_name=violation_type,
            confidence=confidence,
            source_type=source_type,
            track_id=track_id,
            screenshot_path=screenshot_path,
            bike_count=bike_count,
            helmet_count=0,
            no_helmet_count=no_helmet_count,
            bbox_coords=bbox_coords
        )

    def get_recent_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent violations as a list of dicts."""
        df = self.get_detections_df(limit=limit)
        return df.to_dict(orient="records")

    def get_detections_df(
        self,
        class_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieves detections as a pandas DataFrame with optional filters."""
        query = "SELECT * FROM detections WHERE 1=1"
        params = []

        if class_filter and class_filter != "All":
            query += " AND class_name = ?"
            params.append(class_filter)

        if source_filter and source_filter != "All":
            query += " AND source_type = ?"
            params.append(source_filter)

        if search_query:
            query += " AND (timestamp LIKE ? OR screenshot_path LIKE ? OR class_name LIKE ?)"
            params.extend([f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"])

        query += " ORDER BY id DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates authentic real-time aggregate statistics from the SQLite database."""
        with self._connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM detections")
            total_records = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(bike_count), SUM(helmet_count), SUM(no_helmet_count) FROM detections")
            row = cursor.fetchone()
            total_bikes = row[0] or 0
            total_helmets = row[1] or 0
            total_no_helmets = row[2] or 0

            # If counts were logged primarily by row class
            cursor.execute("SELECT COUNT(*) FROM detections WHERE class_name = 'NoHelmet'")
            no_helmet_violations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM detections WHERE class_name = 'Helmet'")
            helmet_compliant = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM detections WHERE class_name = 'BikeWithRider'")
            bike_detections = cursor.fetchone()[0]

            # Reconcile counts
            eff_helmets = max(total_helmets, helmet_compliant)
            eff_no_helmets = max(total_no_helmets, no_helmet_violations)
            eff_bikes = max(total_bikes, bike_detections)
            eff_total = eff_helmets + eff_no_helmets + eff_bikes

            head_total = eff_helmets + eff_no_helmets
            helmet_pct = (eff_helmets / head_total * 100.0) if head_total > 0 else 0.0
            no_helmet_pct = (eff_no_helmets / head_total * 100.0) if head_total > 0 else 0.0

            return {
                "total_records": total_records,
                "total_detections": eff_total,
                "total_helmets": eff_helmets,
                "total_no_helmets": eff_no_helmets,
                "total_motorcycles": eff_bikes,
                "helmet_pct": helmet_pct,
                "no_helmet_pct": no_helmet_pct
            }

    def export_to_csv(self, output_path: Path) -> str:
        """Exports the entire detections table to a CSV file."""
        df = self.get_detections_df()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)


# Global singleton instance
db = DatabaseManager()

