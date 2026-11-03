"""
Real-Time Helmet Detection Engine with ByteTrack and Spatial Rider-Head Association.
Processes video frames, tracks riders across frames, and geometrically verifies whether
detected helmets or violations belong to specific motorcycle riders.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES
from src.tracking import SpatialAssociator, RiderTrackRegistry

# BGR Colors:
# Helmet: Green (0, 255, 0)
# NoHelmet: Red (0, 0, 255)
# BikeWithRider: Blue (255, 0, 0)
COLOR_HELMET = (0, 255, 0)
COLOR_NO_HELMET = (0, 0, 255)
COLOR_BIKE = (255, 0, 0)

COLOR_MAP = {
    0: COLOR_BIKE,       # BikeWithRider
    1: COLOR_NO_HELMET,  # NoHelmet
    2: COLOR_HELMET      # Helmet
}


class RealtimeHelmetDetector:
    """
    YOLOv8 Real-Time Inference and ByteTrack Engine with Spatial Association.
    Associates rider and head detections reliably.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = "auto",
        enable_tracking: bool = True
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.enable_tracking = enable_tracking
        self.model = None
        self.model_path = None
        self.track_registry = RiderTrackRegistry()

        self._load_custom_model(weights_path)

    def _load_custom_model(self, explicit_path: Optional[Path]):
        """Locates and loads the custom trained best.pt model."""
        from ultralytics import YOLO

        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))

        candidates.append(BEST_MODEL_PATH)
        runs_dir = PROJECT_ROOT / "runs" / "detect"
        candidates.extend(sorted(list(runs_dir.glob("**/weights/best.pt")), key=os.path.getmtime, reverse=True))

        target = None
        for cand in candidates:
            if cand.exists() and cand.stat().st_size > 1000:
                target = cand
                break

        if target is None:
            raise FileNotFoundError(
                "Custom trained 'best.pt' not found! "
                "Per project requirements, generic COCO weights cannot be used as a substitute. "
                "Please ensure model training has produced 'models/best.pt'."
            )

        print(f"[Detector] Loading custom trained model: {target}")
        self.model = YOLO(str(target))
        self.model_path = target

        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            print(f"[Detector] Verified Model Classes: {self.model.names}")

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns verified dataset class names dictionary."""
        if self.model and hasattr(self.model, "names") and isinstance(self.model.names, dict):
            return self.model.names
        return CLASS_NAMES

    def set_confidence(self, new_conf: float):
        """Dynamically adjusts confidence threshold (clamped between 0.05 and 0.95)."""
        self.conf_threshold = max(0.05, min(0.95, round(new_conf, 2)))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs YOLOv8 + ByteTrack inference, performs spatial rider-head matching,
        and renders color-coded bounding boxes and association badges.
        """
        start_time = time.time()
        h, w = frame.shape[:2]
        annotated = frame.copy()

        results_data = {
            "boxes": [],
            "bike_count": 0,
            "no_helmet_count": 0,
            "helmet_count": 0,
            "violation": False,
            "associated_riders": [],
            "unassociated_heads": [],
            "violator_track_ids": [],
            "latency_ms": 0.0
        }

        if self.model is None or frame is None:
            return frame, results_data

        # Run ByteTrack tracking or standard detection
        try:
            if self.enable_tracking:
                preds = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device if self.device != "auto" else "",
                    verbose=False
                )
            else:
                preds = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device if self.device != "auto" else "",
                    verbose=False
                )
        except Exception:
            # Fallback to standard predict if tracker fails
            preds = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device if self.device != "auto" else "",
                verbose=False
            )

        raw_bikes = []
        raw_heads = []

        if preds and len(preds) > 0 and preds[0].boxes is not None:
            boxes = preds[0].boxes
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                x1, y1, x2, y2 = xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                box_record = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "track_id": track_id,
                    "box": [x1, y1, x2, y2]
                }
                results_data["boxes"].append(box_record)

                if cls_id == 0:
                    results_data["bike_count"] += 1
                    raw_bikes.append(box_record)
                elif cls_id == 1:
                    results_data["no_helmet_count"] += 1
                    raw_heads.append(box_record)
                elif cls_id == 2:
                    results_data["helmet_count"] += 1
                    raw_heads.append(box_record)

        # Spatial Association between Riders and Heads
        associated_riders, unassociated_heads = SpatialAssociator.associate_heads_to_riders(
            raw_bikes, raw_heads
        )

        results_data["associated_riders"] = associated_riders
        results_data["unassociated_heads"] = unassociated_heads

        # 1. Render Associated Riders & Heads
        for r in associated_riders:
            rx1, ry1, rx2, ry2 = r["box"]
            track_tag = f"Rider #{r['track_id']}" if r["track_id"] is not None else "BikeWithRider"
            conf_pct = r["confidence"] * 100.0

            if r["has_violation"]:
                results_data["violation"] = True
                if r["track_id"] is not None:
                    results_data["violator_track_ids"].append(r["track_id"])

                # Violation: Highlight Rider Box in RED
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), COLOR_NO_HELMET, 3)
                r_label = f"{track_tag} - VIOLATION (NO HELMET)"
                badge_color = COLOR_NO_HELMET
            elif r["compliance_status"].startswith("COMPLIANT"):
                # Compliant: Blue rider box
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), COLOR_BIKE, 2)
                r_label = f"{track_tag} (Compliant) {conf_pct:.0f}%"
                badge_color = COLOR_BIKE
            else:
                # Unverified / No head detected
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (180, 180, 180), 2)
                r_label = f"{track_tag} (Unverified) {conf_pct:.0f}%"
                badge_color = (80, 80, 80)

            # Rider label badge
            self._draw_label_badge(annotated, r_label, rx1, ry1, badge_color, text_color=(255, 255, 255))

            # Draw Associated Heads & Connector Lines
            for h_obj in r["heads"]:
                hx1, hy1, hx2, hy2 = h_obj["box"]
                h_cls = h_obj["class_id"]
                h_color = COLOR_MAP.get(h_cls, (255, 255, 255))
                h_conf_pct = h_obj["confidence"] * 100.0
                h_name = h_obj["class_name"]

                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), h_color, 2)
                h_badge = f"{h_name} {h_conf_pct:.1f}%"
                text_col = (0, 0, 0) if h_cls == 2 else (255, 255, 255)
                self._draw_label_badge(annotated, h_badge, hx1, hy1, h_color, text_color=text_col)

                # Connector line between head and rider center
                r_center = (int((rx1 + rx2) / 2), int(ry1 + 0.25 * (ry2 - ry1)))
                h_center = (int((hx1 + hx2) / 2), int((hy1 + hy2) / 2))
                cv2.line(annotated, h_center, r_center, h_color, 1, cv2.LINE_AA)

        # 2. Render Unassociated Heads
        for uh in unassociated_heads:
            ux1, uy1, ux2, uy2 = uh["box"]
            u_cls = uh["class_id"]
            u_color = COLOR_MAP.get(u_cls, (200, 200, 200))
            u_conf = uh["confidence"] * 100.0
            u_name = uh["class_name"]

            # Draw dashed or thin outline
            cv2.rectangle(annotated, (ux1, uy1), (ux2, uy2), u_color, 2)
            u_label = f"{u_name} ({u_conf:.1f}%) [Unassociated]"
            self._draw_label_badge(annotated, u_label, ux1, uy1, (100, 100, 100), text_color=(255, 255, 255))

            # Mark global violation if unassociated NoHelmet detected
            if u_cls == 1:
                results_data["violation"] = True

        results_data["detections"] = results_data["boxes"]
        results_data["latency_ms"] = (time.time() - start_time) * 1000.0
        return annotated, results_data

    @staticmethod
    def _draw_label_badge(image: np.ndarray, text: str, x: int, y: int, bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        y1 = max(0, y - th - baseline - 4)
        y2 = y
        cv2.rectangle(image, (x, y1), (x + tw + 6, y2), bg_color, -1)
        cv2.putText(image, text, (x + 3, y2 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_telemetry_hud(
        image: np.ndarray,
        results_data: Dict[str, Any],
        fps: float,
        conf_threshold: float,
        sound_enabled: bool = True
    ) -> np.ndarray:
        h, w = image.shape[:2]
        annotated = image.copy()

        banner_h = 50
        overlay = annotated[0:banner_h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, annotated[0:banner_h, 0:w], 0.25, 0, annotated[0:banner_h, 0:w])

        bike_cnt = results_data.get("bike_count", 0)
        helmet_cnt = results_data.get("helmet_count", 0)
        no_helmet_cnt = results_data.get("no_helmet_count", 0)

        # Counts
        cv2.putText(annotated, f"Bikes: {bike_cnt}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Helmets: {helmet_cnt}", (160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Violations: {no_helmet_cnt}", (320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2, cv2.LINE_AA)

        # Right Telemetry
        sound_tag = "Audio: ON" if sound_enabled else "Audio: OFF"
        right_info = f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f} | {sound_tag}"
        cv2.putText(annotated, right_info, (w - 360, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Clear Red Warning Banner when NoHelmet is detected
        if results_data.get("violation", False):
            alert_bar = annotated[h - 45:h, 0:w].copy()
            cv2.rectangle(alert_bar, (0, 0), (w, 45), (0, 0, 220), -1)
            cv2.addWeighted(alert_bar, 0.85, annotated[h - 45:h, 0:w], 0.15, 0, annotated[h - 45:h, 0:w])
            cv2.putText(annotated, "WARNING: NO-HELMET VIOLATION DETECTED", (w // 2 - 250, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
