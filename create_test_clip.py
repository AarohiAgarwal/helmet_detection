"""
Utility to create a sample test video clip from processed test images.
Used for offline video detection and webcam simulation.
"""

import os
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMG_DIR = PROJECT_ROOT / "dataset" / "processed" / "images" / "test"
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "test_clip.mp4"


def create_sample_video(output_path: Path = OUTPUT_VIDEO_PATH, duration_sec: int = 5, fps: int = 15):
    if not TEST_IMG_DIR.exists():
        print(f"Error: Test directory not found at {TEST_IMG_DIR}")
        return

    images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:20]
    if not images:
        print("No images found in test directory.")
        return

    # Target resolution: 1280x720
    target_w, target_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

    frames_per_img = max(1, (duration_sec * fps) // len(images))

    for img_name in images:
        img_path = TEST_IMG_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize and letterbox to 1280x720
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))

        canvas = cv2.copyMakeBorder(
            resized,
            top=(target_h - nh) // 2,
            bottom=(target_h - nh + 1) // 2,
            left=(target_w - nw) // 2,
            right=(target_w - nw + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(30, 30, 30)
        )
        # Ensure exact dimensions
        canvas = cv2.resize(canvas, (target_w, target_h))

        for _ in range(frames_per_img):
            out.write(canvas)

    out.release()
    print(f"Sample test video created successfully at: {output_path}")


if __name__ == "__main__":
    create_sample_video()

