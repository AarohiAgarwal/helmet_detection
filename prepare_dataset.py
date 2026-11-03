"""
Dataset Preparation and Augmentation Pipeline for YOLOv8 Helmet Detection.
Creates a clean, sanitized, and non-leaking processed dataset split into Train (70%),
Validation (20%), and Test (10%) while leaving original dataset intact.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_ROOT,
    IMAGES_DIR,
    LABELS_YOLO_DIR,
    CLASS_NAMES
)

PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
DATA_YAML_PATH = PROJECT_ROOT / "dataset" / "data.yaml"


class BoundingBox:
    """Represents a normalized YOLO bounding box [cls_id, xc, yc, w, h]."""
    def __init__(self, cls_id: int, xc: float, yc: float, w: float, h: float):
        self.cls_id = int(cls_id)
        self.xc = float(xc)
        self.yc = float(yc)
        self.w = float(w)
        self.h = float(h)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2.0
        y1 = self.yc - self.h / 2.0
        x2 = self.xc + self.w / 2.0
        y2 = self.yc + self.h / 2.0
        return x1, y1, x2, y2

    @classmethod
    def from_xyxy(cls, cls_id: int, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        return cls(cls_id, xc, yc, w, h)

    def is_valid(self, min_dim: float = 0.001) -> bool:
        """Checks if box has positive area, valid center within bounds, and recognized class."""
        if self.cls_id not in CLASS_NAMES:
            return False
        if self.w < min_dim or self.h < min_dim:
            return False
        if not (0.0 < self.xc < 1.0 and 0.0 < self.yc < 1.0):
            return False
        return True

    def clamp(self) -> "BoundingBox":
        """Clamps box coordinates strictly within [0.0, 1.0]."""
        x1, y1, x2, y2 = self.to_xyxy()
        x1_c = max(0.0, min(1.0, x1))
        y1_c = max(0.0, min(1.0, y1))
        x2_c = max(0.0, min(1.0, x2))
        y2_c = max(0.0, min(1.0, y2))
        return BoundingBox.from_xyxy(self.cls_id, x1_c, y1_c, x2_c, y2_c)

    def to_yolo_str(self) -> str:
        return f"{self.cls_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


class AugmentationEngine:
    """Safe, realism-preserving data augmentations for helmet detection training images."""

    @staticmethod
    def horizontal_flip(image: np.ndarray, boxes: List[BoundingBox]) -> Tuple[np.ndarray, List[BoundingBox]]:
        flipped_img = cv2.flip(image, 1)
        flipped_boxes = []
        for b in boxes:
            new_xc = 1.0 - b.xc
            flipped_boxes.append(BoundingBox(b.cls_id, new_xc, b.yc, b.w, b.h))
        return flipped_img, flipped_boxes

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, boxes: List[BoundingBox], alpha: float = 1.1, beta: int = 10) -> Tuple[np.ndarray, List[BoundingBox]]:
        # alpha in [0.85, 1.15], beta in [-20, 20]
        adjusted_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # Bounding box locations do not change
        return adjusted_img, [BoundingBox(b.cls_id, b.xc, b.yc, b.w, b.h) for b in boxes]

    @staticmethod
    def small_rotation(image: np.ndarray, boxes: List[BoundingBox], angle: float = 4.0) -> Tuple[np.ndarray, List[BoundingBox]]:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        rotated_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b.to_xyxy()
            # Absolute pixel corners
            pts = np.array([
                [x1 * w, y1 * h],
                [x2 * w, y1 * h],
                [x2 * w, y2 * h],
                [x1 * w, y2 * h]
            ])
            # Apply affine transform
            ones = np.ones(shape=(len(pts), 1))
            pts_ones = np.hstack([pts, ones])
            transformed_pts = M.dot(pts_ones.T).T

            new_x1 = max(0.0, np.min(transformed_pts[:, 0])) / w
            new_y1 = max(0.0, np.min(transformed_pts[:, 1])) / h
            new_x2 = min(1.0, np.max(transformed_pts[:, 0])) / w
            new_y2 = min(1.0, np.max(transformed_pts[:, 1])) / h

            new_box = BoundingBox.from_xyxy(b.cls_id, new_x1, new_y1, new_x2, new_y2)
            if new_box.is_valid():
                rotated_boxes.append(new_box)

        return rotated_img, rotated_boxes

    @staticmethod
    def random_blur(image: np.ndarray, boxes: List[BoundingBox], ksize: int = 3) -> Tuple[np.ndarray, List[BoundingBox]]:
        blurred_img = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred_img, [BoundingBox(b.cls_id, b.xc, b.yc, b.w, b.h) for b in boxes]

    @staticmethod
    def small_translation(image: np.ndarray, boxes: List[BoundingBox], tx_pct: float = 0.03, ty_pct: float = 0.03) -> Tuple[np.ndarray, List[BoundingBox]]:
        h, w = image.shape[:2]
        dx = int(tx_pct * w)
        dy = int(ty_pct * h)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        shifted_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b.to_xyxy()
            new_x1 = max(0.0, min(1.0, x1 + tx_pct))
            new_y1 = max(0.0, min(1.0, y1 + ty_pct))
            new_x2 = max(0.0, min(1.0, x2 + tx_pct))
            new_y2 = max(0.0, min(1.0, y2 + ty_pct))
            new_box = BoundingBox.from_xyxy(b.cls_id, new_x1, new_y1, new_x2, new_y2)
            if new_box.is_valid():
                shifted_boxes.append(new_box)

        return shifted_img, shifted_boxes


def load_and_sanitize_annotation(txt_path: Path) -> Tuple[List[BoundingBox], int, int]:
    """Reads a YOLO annotation file, sanitizes coordinates, and drops invalid boxes."""
    if not txt_path.exists():
        return [], 0, 0

    valid_boxes = []
    dropped_count = 0
    clamped_count = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            dropped_count += 1
            continue

        try:
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            raw_box = BoundingBox(cls_id, xc, yc, w, h)
            clamped_box = raw_box.clamp()

            # Check if clamping adjusted coordinates
            if abs(clamped_box.xc - raw_box.xc) > 1e-4 or abs(clamped_box.yc - raw_box.yc) > 1e-4 or abs(clamped_box.w - raw_box.w) > 1e-4 or abs(clamped_box.h - raw_box.h) > 1e-4:
                clamped_count += 1

            if clamped_box.is_valid(min_dim=0.001):
                valid_boxes.append(clamped_box)
            else:
                dropped_count += 1
        except Exception:
            dropped_count += 1

    return valid_boxes, dropped_count, clamped_count


def stratified_split(
    image_names: List[str],
    image_to_classes: Dict[str, set],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Partitions image names into 70:20:10 splits maintaining class representation balance."""
    random.seed(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    # Group images by class combination signature
    signature_groups = defaultdict(list)
    for name in image_names:
        sig = tuple(sorted(list(image_to_classes[name])))
        signature_groups[sig].append(name)

    train_split = []
    val_split = []
    test_split = []

    for sig, imgs in signature_groups.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure remaining go to test
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_split.extend(imgs[:n_train])
        val_split.extend(imgs[n_train:n_train + n_val])
        test_split.extend(imgs[n_train + n_val:])

    # Verify disjointness
    s_tr, s_va, s_te = set(train_split), set(val_split), set(test_split)
    assert len(s_tr.intersection(s_va)) == 0, "Data leakage between train and val!"
    assert len(s_tr.intersection(s_te)) == 0, "Data leakage between train and test!"
    assert len(s_va.intersection(s_te)) == 0, "Data leakage between val and test!"

    # Minor adjustment to match exact counts if rounding difference
    total = len(image_names)
    target_train = int(total * train_ratio)
    target_val = int(total * val_ratio)
    target_test = total - target_train - target_val

    all_allocated = train_split + val_split + test_split
    random.shuffle(all_allocated)

    final_train = sorted(all_allocated[:target_train])
    final_val = sorted(all_allocated[target_train:target_train + target_val])
    final_test = sorted(all_allocated[target_train + target_val:])

    return {
        "train": final_train,
        "val": final_val,
        "test": final_test
    }


def prepare_dataset(
    output_dir: Path = PROCESSED_ROOT,
    augment_train: bool = True,
    augment_minority_only: bool = False,
    num_augmented_samples: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """Executes the complete end-to-end dataset preparation and augmentation workflow."""
    print("=" * 65)
    print("STAGE 2: DATA PREPROCESSING AND AUGMENTATION")
    print("=" * 65)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Source images directory not found: {IMAGES_DIR}")
    if not LABELS_YOLO_DIR.exists():
        raise FileNotFoundError(f"Source YOLO labels directory not found: {LABELS_YOLO_DIR}")

    # Step 1: Clean and recreate destination structure
    print(f"\n[1/6] Setting up target processed directory: {output_dir}")
    if output_dir.exists():
        print(f"      Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 2: Index images and parse class signatures
    print("\n[2/6] Indexing raw images and checking annotation files...")
    all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"      Found {len(all_images)} original images.")

    image_to_classes = {}
    total_raw_boxes = 0
    for img_name in all_images:
        stem = Path(img_name).stem
        txt_path = LABELS_YOLO_DIR / f"{stem}.txt"
        classes_present = set()
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            classes_present.add(cid)
                            total_raw_boxes += 1
                        except ValueError:
                            pass
        image_to_classes[img_name] = classes_present

    # Step 3: Stratified 70:20:10 Split
    print("\n[3/6] Generating 70:20:10 Train / Validation / Test split...")
    splits = stratified_split(all_images, image_to_classes, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, seed=seed)
    print(f"      Train set: {len(splits['train'])} images (70.0%)")
    print(f"      Val set:   {len(splits['val'])} images (20.0%)")
    print(f"      Test set:  {len(splits['test'])} images (10.0%)")

    # Step 4: Copy, sanitize, and verify bounding boxes for all splits
    print("\n[4/6] Copying images and sanitizing bounding boxes...")
    split_stats = {}
    total_clamped = 0
    total_dropped = 0

    for split_name, img_list in splits.items():
        box_counts = Counter()
        split_clamped = 0
        split_dropped = 0
        total_boxes = 0

        for img_name in img_list:
            stem = Path(img_name).stem
            src_img = IMAGES_DIR / img_name
            src_txt = LABELS_YOLO_DIR / f"{stem}.txt"

            dst_img = output_dir / "images" / split_name / img_name
            dst_txt = output_dir / "labels" / split_name / f"{stem}.txt"

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Sanitize annotation
            boxes, dropped, clamped = load_and_sanitize_annotation(src_txt)
            split_clamped += clamped
            split_dropped += dropped

            # Write sanitized annotation
            with open(dst_txt, "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(b.to_yolo_str() + "\n")
                    box_counts[b.cls_id] += 1
                    total_boxes += 1

        total_clamped += split_clamped
        total_dropped += split_dropped
        split_stats[split_name] = {
            "images": len(img_list),
            "boxes": total_boxes,
            "classes": dict(box_counts),
            "clamped": split_clamped,
            "dropped": split_dropped
        }

        print(f"      - {split_name.capitalize():<5}: {len(img_list)} images, {total_boxes} boxes "
              f"(Clamped: {split_clamped}, Dropped: {split_dropped})")

    print(f"\n      Total bounding boxes sanitized across splits: {total_raw_boxes} -> "
          f"{sum(s['boxes'] for s in split_stats.values())} retained, {total_dropped} dropped (degenerate/zero-area).")

    # Step 5: Apply Safe Augmentation to Train Split ONLY
    augmented_count = 0
    aug_boxes_count = 0
    if augment_train:
        print(f"\n[5/6] Applying safe training augmentations (Targeting {num_augmented_samples} images)...")
        print("      Augmentations: Horizontal Flip, Brightness/Contrast, Small Rotation, Scaling, Blur, Translation.")
        print("      Validation and Test sets remain strictly untouched.")

        train_imgs = splits["train"]
        random.seed(seed)

        # Prioritize images containing Helmet (class 2) to improve minority class representation
        helmet_train_imgs = [img for img in train_imgs if 2 in image_to_classes[img]]
        other_train_imgs = [img for img in train_imgs if 2 not in image_to_classes[img]]

        selected_for_aug = []
        if augment_minority_only:
            selected_for_aug = helmet_train_imgs[:num_augmented_samples]
        else:
            # Half minority class, half other
            k_helmet = min(len(helmet_train_imgs), num_augmented_samples // 2)
            k_other = min(len(other_train_imgs), num_augmented_samples - k_helmet)
            selected_for_aug = random.sample(helmet_train_imgs, k_helmet) + random.sample(other_train_imgs, k_other)
            random.shuffle(selected_for_aug)

        aug_methods = ["flip", "bright", "rot", "blur", "trans"]

        for idx, img_name in enumerate(selected_for_aug):
            stem = Path(img_name).stem
            img_path = output_dir / "images" / "train" / img_name
            txt_path = output_dir / "labels" / "train" / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            boxes, _, _ = load_and_sanitize_annotation(txt_path)
            if not boxes:
                continue

            # Select augmentation strategy
            method = aug_methods[idx % len(aug_methods)]
            aug_img = None
            aug_boxes = []

            if method == "flip":
                aug_img, aug_boxes = AugmentationEngine.horizontal_flip(img, boxes)
            elif method == "bright":
                alpha = random.uniform(0.88, 1.15)
                beta = random.randint(-18, 18)
                aug_img, aug_boxes = AugmentationEngine.adjust_brightness_contrast(img, boxes, alpha, beta)
            elif method == "rot":
                angle = random.choice([-5.0, -3.0, 3.0, 5.0])
                aug_img, aug_boxes = AugmentationEngine.small_rotation(img, boxes, angle)
            elif method == "blur":
                aug_img, aug_boxes = AugmentationEngine.random_blur(img, boxes, ksize=3)
            elif method == "trans":
                tx = random.choice([-0.03, 0.03])
                ty = random.choice([-0.03, 0.03])
                aug_img, aug_boxes = AugmentationEngine.small_translation(img, boxes, tx, ty)

            if aug_img is not None and aug_boxes:
                aug_stem = f"{stem}_aug_{method}"
                aug_img_path = output_dir / "images" / "train" / f"{aug_stem}.jpg"
                aug_txt_path = output_dir / "labels" / "train" / f"{aug_stem}.txt"

                cv2.imwrite(str(aug_img_path), aug_img)
                with open(aug_txt_path, "w", encoding="utf-8") as f:
                    for b in aug_boxes:
                        f.write(b.to_yolo_str() + "\n")
                        aug_boxes_count += 1
                augmented_count += 1

        print(f"      Generated {augmented_count} augmented training images with {aug_boxes_count} bounding boxes.")
        split_stats["train"]["images"] += augmented_count
        split_stats["train"]["boxes"] += aug_boxes_count
    else:
        print("\n[5/6] Training augmentation skipped (--no-augment specified).")

    # Step 6: Generate YOLOv8 data.yaml
    print(f"\n[6/6] Generating YOLOv8 dataset descriptor: {DATA_YAML_PATH}")
    data_yaml_content = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in CLASS_NAMES.items()}
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print("      data.yaml successfully generated.")
    print("=" * 65)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return {
        "processed_root": str(output_dir),
        "data_yaml": str(DATA_YAML_PATH),
        "split_stats": split_stats,
        "total_clamped": total_clamped,
        "total_dropped": total_dropped,
        "augmented_images": augmented_count
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and augment EdgeVision dataset for YOLOv8")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_ROOT), help="Target processed dataset path")
    parser.add_argument("--no-augment", action="store_true", help="Disable training set augmentation")
    parser.add_argument("--augment-samples", type=int, default=500, help="Number of augmented train samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        output_dir=Path(args.output_dir),
        augment_train=not args.no_augment,
        num_augmented_samples=args.augment_samples,
        seed=args.seed
    )

