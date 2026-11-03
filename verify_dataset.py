"""
Verification Script for Processed YOLOv8 Helmet Detection Dataset.
Performs thorough integrity checks:
1. File pairing & presence (every image has an annotation).
2. Data leakage verification between Train, Validation, and Test splits.
3. Bounding box validity (coordinates strictly within [0, 1], positive dimensions, valid class IDs).
4. Ultralytics YOLOv8 native loader compatibility verification.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES

PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
DATA_YAML_PATH = PROJECT_ROOT / "dataset" / "data.yaml"


class DatasetVerifier:
    def __init__(self, processed_dir: Path = PROCESSED_ROOT, yaml_path: Path = DATA_YAML_PATH):
        self.processed_dir = Path(processed_dir)
        self.yaml_path = Path(yaml_path)
        self.errors = []
        self.warnings = []

    def verify_all(self) -> bool:
        print("=" * 65)
        print("STAGE 2: DATASET VERIFICATION & INTEGRITY CHECK")
        print("=" * 65)

        # 1. Directory Structure Check
        print("\n[Step 1/5] Verifying directory structure...")
        if not self.processed_dir.exists():
            self.errors.append(f"Processed dataset directory not found at: {self.processed_dir}")
            return False

        for split in ["train", "val", "test"]:
            img_dir = self.processed_dir / "images" / split
            lbl_dir = self.processed_dir / "labels" / split
            if not img_dir.exists():
                self.errors.append(f"Missing images directory: {img_dir}")
            if not lbl_dir.exists():
                self.errors.append(f"Missing labels directory: {lbl_dir}")

        if self.errors:
            self._print_results()
            return False
        print("      Directory hierarchy verified successfully.")

        # 2. File Pairing & Image Integrity Check
        print("\n[Step 2/5] Checking file pairing & image integrity...")
        split_image_names = {}
        split_base_names = {}
        class_distribution = {split: Counter() for split in ["train", "val", "test"]}
        box_counts = {split: 0 for split in ["train", "val", "test"]}

        for split in ["train", "val", "test"]:
            img_dir = self.processed_dir / "images" / split
            lbl_dir = self.processed_dir / "labels" / split

            images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            split_image_names[split] = set(images)

            # Base names (stripping _aug suffix to verify provenance)
            base_stems = set()
            for img in images:
                stem = Path(img).stem
                if "_aug_" in stem:
                    stem = stem.split("_aug_")[0]
                base_stems.add(stem)
            split_base_names[split] = base_stems

            for img_name in images:
                img_path = img_dir / img_name
                stem = Path(img_name).stem
                txt_path = lbl_dir / f"{stem}.txt"

                # Check annotation file exists
                if not txt_path.exists():
                    self.errors.append(f"Missing annotation: {txt_path} for image {img_path}")
                    continue

                # Check image can be opened
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    self.errors.append(f"Corrupt image file: {img_path} ({e})")
                    continue

                # Validate bounding boxes
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

                for l_idx, line in enumerate(lines):
                    tokens = line.split()
                    if len(tokens) != 5:
                        self.errors.append(f"Malformed label in {txt_path} line {l_idx}: '{line}'")
                        continue
                    try:
                        cid = int(tokens[0])
                        xc, yc, w, h = map(float, tokens[1:])

                        if cid not in CLASS_NAMES:
                            self.errors.append(f"Invalid class ID {cid} in {txt_path} line {l_idx}")
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                            self.errors.append(f"Out-of-bounds center ({xc}, {yc}) in {txt_path} line {l_idx}")
                        if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                            self.errors.append(f"Invalid dimensions (w={w}, h={h}) in {txt_path} line {l_idx}")

                        class_distribution[split][cid] += 1
                        box_counts[split] += 1
                    except ValueError as e:
                        self.errors.append(f"Unparseable coordinate in {txt_path} line {l_idx}: {e}")

            print(f"      - {split.capitalize():<5}: {len(images):>5} images | {box_counts[split]:>5} boxes")

        # 3. Data Leakage Verification
        print("\n[Step 3/5] Verifying data leakage across splits...")
        leak_train_val = split_base_names["train"].intersection(split_base_names["val"])
        leak_train_test = split_base_names["train"].intersection(split_base_names["test"])
        leak_val_test = split_base_names["val"].intersection(split_base_names["test"])

        if leak_train_val:
            self.errors.append(f"DATA LEAKAGE: {len(leak_train_val)} images overlap between Train and Validation!")
        if leak_train_test:
            self.errors.append(f"DATA LEAKAGE: {len(leak_train_test)} images overlap between Train and Test!")
        if leak_val_test:
            self.errors.append(f"DATA LEAKAGE: {len(leak_val_test)} images overlap between Validation and Test!")

        if not (leak_train_val or leak_train_test or leak_val_test):
            print("      [PASSED] Zero data leakage detected. Splits are strictly disjoint.")

        # 4. YAML Descriptor Check
        print("\n[Step 4/5] Checking dataset configuration (data.yaml)...")
        if not self.yaml_path.exists():
            self.errors.append(f"Missing data.yaml at: {self.yaml_path}")
            return False

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "names" in cfg, "data.yaml missing 'names' dictionary!"
        assert len(cfg["names"]) == 3, f"Expected 3 classes in data.yaml, found {len(cfg['names'])}"
        print(f"      Classes declared: {cfg['names']}")
        print(f"      Base path declared: {cfg.get('path', 'None')}")

        # 5. Native Ultralytics YOLOv8 Loading Verification
        print("\n[Step 5/5] Testing dataset loader compatibility with Ultralytics YOLOv8...")
        try:
            from ultralytics.data.utils import check_det_dataset
            dataset_info = check_det_dataset(str(self.yaml_path))
            print("      [PASSED] Ultralytics check_det_dataset succeeded!")
            print(f"      Ultralytics verified splits: train={dataset_info.get('train')}, val={dataset_info.get('val')}")
        except Exception as e:
            self.errors.append(f"Ultralytics failed to load dataset: {e}")

        # Summary output
        self._print_results(class_distribution, box_counts, split_image_names)
        return len(self.errors) == 0

    def _print_results(self, class_distribution=None, box_counts=None, split_image_names=None):
        print("\n" + "=" * 65)
        print("VERIFICATION RESULTS SUMMARY")
        print("=" * 65)
        if split_image_names and box_counts:
            for split in ["train", "val", "test"]:
                print(f"  Split: {split.upper()}")
                print(f"    Images: {len(split_image_names[split])}")
                print(f"    Bounding Boxes: {box_counts[split]}")
                print("    Class Breakdown:")
                for cid in sorted(class_distribution[split].keys()):
                    cname = CLASS_NAMES.get(cid, str(cid))
                    cnt = class_distribution[split][cid]
                    pct = (cnt / box_counts[split]) * 100 if box_counts[split] > 0 else 0
                    print(f"      - {cid} ({cname}): {cnt} ({pct:.2f}%)")
            print("-" * 65)

        if self.warnings:
            print(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                print(f"  [!] {w}")

        if self.errors:
            print(f"FAILED: Found {len(self.errors)} error(s):")
            for e in self.errors[:10]:
                print(f"  [X] {e}")
            print("\nDataset verification FAILED. Please resolve errors before continuing.")
        else:
            print("SUCCESS: Processed dataset passed all integrity and YOLOv8 loading checks!")
            print("Status: READY FOR MODEL TRAINING (Pending user approval).")
        print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify processed YOLOv8 dataset")
    parser.add_argument("--dir", type=str, default=str(PROCESSED_ROOT), help="Processed dataset root directory")
    parser.add_argument("--yaml", type=str, default=str(DATA_YAML_PATH), help="Path to data.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verifier = DatasetVerifier(processed_dir=Path(args.dir), yaml_path=Path(args.yaml))
    success = verifier.verify_all()
    sys.exit(0 if success else 1)

