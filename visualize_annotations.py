"""
Visualization Script for Processed YOLOv8 Dataset Annotations.
Renders and exports sample images with overlaid bounding boxes and class labels.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, CLASS_COLORS, REPORTS_DIR

PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
SAMPLES_DIR = REPORTS_DIR / "samples"


def draw_yolo_annotations(image: np.ndarray, label_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Draws YOLO bounding boxes and labels on an image."""
    annotated = image.copy()
    h, w = image.shape[:2]
    parsed_boxes = []

    if not label_path.exists():
        return annotated, parsed_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            # Clamp pixel coords
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cname = CLASS_NAMES.get(cid, str(cid))
            color = CLASS_COLORS.get(cid, (255, 255, 255))

            parsed_boxes.append({
                "class_id": cid,
                "class_name": cname,
                "bbox_xyxy": [x1, y1, x2, y2],
                "yolo_coords": [xc, yc, bw, bh]
            })

            # Draw rectangle
            thickness = 2 if cid != 1 else 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label text
            label_text = f"{cname}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            txt_thick = 2
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, txt_thick)

            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = y1

            # Background pill
            cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
            text_color = (0, 0, 0) if cid != 1 else (255, 255, 255)
            cv2.putText(annotated, label_text, (x1 + 3, label_y2 - 3), font, scale, text_color, txt_thick, cv2.LINE_AA)

        except Exception as e:
            print(f"Error drawing line '{line}': {e}")

    return annotated, parsed_boxes


def visualize_split_samples(
    split: str = "train",
    num_samples: int = 5,
    save_dir: Path = SAMPLES_DIR,
    processed_dir: Path = PROCESSED_ROOT
) -> List[Dict[str, Any]]:
    """Renders and saves sample images from a specific dataset split."""
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split
    save_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"[Error] Directory not found: {img_dir}")
        return []

    all_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not all_images:
        print(f"[Error] No images found in: {img_dir}")
        return []

    # Pick samples: select ones that have both Helmet and NoHelmet or sample diverse indices
    selected_images = []
    # Try finding images with violations and helmets
    for img_name in all_images:
        stem = Path(img_name).stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "1" in content and "2" in content:  # Has both Helmet and NoHelmet
                    selected_images.append(img_name)
                    if len(selected_images) >= num_samples:
                        break

    if len(selected_images) < num_samples:
        for img_name in all_images:
            if img_name not in selected_images:
                selected_images.append(img_name)
                if len(selected_images) >= num_samples:
                    break

    results = []
    print(f"\n--- VISUALIZING {len(selected_images)} SAMPLES FROM '{split.upper()}' SPLIT ---")
    for idx, img_name in enumerate(selected_images):
        stem = Path(img_name).stem
        img_path = img_dir / img_name
        lbl_path = lbl_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        annotated_img, boxes = draw_yolo_annotations(img, lbl_path)
        save_path = save_dir / f"{split}_sample_{idx+1}_{stem}.jpg"
        cv2.imwrite(str(save_path), annotated_img)

        print(f"\nSample {idx+1}: {img_name} (Resolution: {img.shape[1]}x{img.shape[0]})")
        print(f"  Saved visualization to: {save_path}")
        print(f"  Objects ({len(boxes)}):")
        for b in boxes:
            print(f"    - Class: {b['class_name']} (ID {b['class_id']}), BBox: {b['bbox_xyxy']}")

        results.append({
            "image_name": img_name,
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "save_path": str(save_path),
            "boxes": boxes
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize processed dataset annotations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sample images to render")
    parser.add_argument("--save-dir", type=str, default=str(SAMPLES_DIR), help="Output directory for annotated samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_split_samples(
        split=args.split,
        num_samples=args.num_samples,
        save_dir=Path(args.save_dir)
    )
