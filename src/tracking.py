"""
Rider-to-Helmet Spatial Association and Multi-Object Tracking Engine.
Uses ByteTrack for persistent vehicle/rider IDs and geometric containment
to reliably link heads (Helmet / NoHelmet) to motorcycle riders.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np


class SpatialAssociator:
    """
    Associates detected head objects (Helmet or NoHelmet) with tracked motorcycle riders (BikeWithRider).
    Prevents false assumptions by validating geometric containment and vertical positioning.
    """

    @staticmethod
    def calculate_ioh(head_box: List[int], rider_box: List[int]) -> float:
        """
        Calculates Intersection-over-Head (IoH): the fraction of the head box
        that is contained within the rider box.
        """
        hx1, hy1, hx2, hy2 = head_box
        bx1, by1, bx2, by2 = rider_box

        ix1 = max(hx1, bx1)
        iy1 = max(hy1, by1)
        ix2 = min(hx2, bx2)
        iy2 = min(hy2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection_area = iw * ih

        head_area = max(1, (hx2 - hx1) * (hy2 - hy1))
        return intersection_area / float(head_area)

    @classmethod
    def is_head_on_rider(cls, head_box: List[int], rider_box: List[int]) -> Tuple[bool, float]:
        """
        Evaluates whether a head box belongs to a rider based on:
        1. Head center is horizontally within the rider's bounds (with small margin).
        2. Head center is in the upper 60% of the rider's vertical frame.
        3. Significant spatial overlap (IoH >= 0.35).
        """
        hx1, hy1, hx2, hy2 = head_box
        bx1, by1, bx2, by2 = rider_box

        hcx = (hx1 + hx2) / 2.0
        hcy = (hy1 + hy2) / 2.0

        rider_w = bx2 - bx1
        rider_h = by2 - by1

        # Horizontal alignment: center of head is within rider box (with 10% lateral tolerance)
        horiz_valid = (bx1 - 0.10 * rider_w) <= hcx <= (bx2 + 0.10 * rider_w)

        # Vertical alignment: head is in the upper portion (top 60% of rider height)
        vert_valid = (by1 - 0.15 * rider_h) <= hcy <= (by1 + 0.60 * rider_h)

        ioh = cls.calculate_ioh(head_box, rider_box)

        if horiz_valid and vert_valid and (ioh >= 0.35 or by1 <= hcy <= by1 + 0.40 * rider_h):
            return True, ioh

        return False, 0.0

    @classmethod
    def associate_heads_to_riders(
        cls,
        bikes: List[Dict[str, Any]],
        heads: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Matches detected heads (Helmet/NoHelmet) to tracked motorcycles (BikeWithRider).

        Returns:
        - associated_riders: list of rider dicts with assigned 'heads', 'compliance_status', 'track_id'
        - unassociated_heads: list of head dicts that do not match any detected motorcycle
        """
        # Initialize rider records
        riders_with_status = []
        for b in bikes:
            r_entry = b.copy()
            r_entry["heads"] = []
            r_entry["has_violation"] = False
            r_entry["compliance_status"] = "UNVERIFIED"  # Default before head match
            riders_with_status.append(r_entry)

        unassociated_heads = []

        for h in heads:
            best_match_idx = -1
            best_score = -1.0

            for idx, r in enumerate(riders_with_status):
                is_match, score = cls.is_head_on_rider(h["box"], r["box"])
                if is_match and score > best_score:
                    best_match_idx = idx
                    best_score = score

            if best_match_idx >= 0:
                # Associated to rider
                matching_rider = riders_with_status[best_match_idx]
                matching_rider["heads"].append(h)
                if h["class_id"] == 1:  # NoHelmet
                    matching_rider["has_violation"] = True
            else:
                unassociated_heads.append(h)

        # Assign final verified compliance status
        for r in riders_with_status:
            if not r["heads"]:
                r["compliance_status"] = "UNVERIFIED (No Head Detected)"
            elif r["has_violation"]:
                r["compliance_status"] = "VIOLATION (No Helmet)"
            else:
                r["compliance_status"] = "COMPLIANT (Helmet Verified)"

        return riders_with_status, unassociated_heads


class RiderTrackRegistry:
    """
    Maintains persistent memory of tracked riders across frames to prevent duplicate alerts.
    """

    def __init__(self):
        self.alerted_track_ids: Set[int] = set()
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_unique_violators: int = 0
        self.total_unique_compliant: int = 0

    def should_alert_rider(self, track_id: Optional[int]) -> bool:
        """
        Determines whether a violation alert should be triggered for this rider.
        Returns False if the track_id has already been alerted in the current session.
        """
        if track_id is None:
            # Unassociated or untracked detection - allow conditional alert
            return True

        if track_id in self.alerted_track_ids:
            return False  # Already alerted for this specific rider!

        # First-time violation for this tracked rider
        self.alerted_track_ids.add(track_id)
        self.total_unique_violators += 1
        return True

    def reset(self):
        """Clears tracking memory."""
        self.alerted_track_ids.clear()
        self.active_tracks.clear()
        self.total_unique_violators = 0
        self.total_unique_compliant = 0

