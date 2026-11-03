# EdgeVision Dataset Inspection & Quality Report

**Project:** Real-Time Motorcycle Helmet Detection System  
**Inspection Date:** 2026-11-03  
**Status:** Completed & Verified (Model Training Halted Pending Approval)

---

## 1. Dataset Overview & Location

The EdgeVision dataset was automatically identified in the project directory. As instructed, the original dataset has been preserved completely untouched without any file deletion, modification, or duplication.

- **Root Directory:** `dataset/EdgeVision Dataset/EdgeVision-Dataset`
- **Images Directory:** `dataset/EdgeVision Dataset/EdgeVision-Dataset/images/`
- **Annotations Directory:** `dataset/EdgeVision Dataset/EdgeVision-Dataset/labels/`
- **Total Image Files:** 2,392
- **Image Extension & Format:** 100% JPEG (`.jpg`)
- **Corrupted Images:** 0 (All 2,392 images verified via PIL image integrity check)
- **Duplicate Images:** 0 (Verified via MD5 checksum analysis)
- **Image Resolution Metrics:**
  - **Minimum Resolution:** 161 x 225 px
  - **Maximum Resolution:** 4624 x 4096 px
  - **Average Resolution:** 1154 x 1411 px

---

## 2. Annotation Formats & Structure

The dataset provides annotations in **three standard computer vision formats**:

| Format | Directory Path | File Count / Structure | Coordinates Representation |
| :--- | :--- | :--- | :--- |
| **YOLO** | `labels/yolo/` | 2,392 `.txt` files + `classes.txt` | Normalized `[x_center, y_center, width, height]` (0-indexed) |
| **Pascal VOC** | `labels/voc_xml/` | 2,392 `.xml` files | Absolute pixel coordinates `[xmin, ymin, xmax, ymax]` |
| **COCO** | `labels/coco/` | `annotations.json` & `annotations.csv` | Absolute pixel coordinates `[x, y, width, height]` (1-indexed) |

---

## 3. Class Names, IDs, and Distribution

The dataset defines 3 primary target classes for motorcycle helmet compliance detection:

### Class Mapping Verification

| Class ID (YOLO) | Class ID (COCO) | Class Name | Description |
| :--- | :--- | :--- | :--- |
| **0** | 1 | `BikeWithRider` | Motorcycle or scooter with rider/passenger |
| **1** | 2 | `NoHelmet` | Rider or pillion without a safety helmet (Violation) |
| **2** | 3 | `Helmet` | Rider or pillion wearing a safety helmet (Compliant) |

### Object Counts & Frequency Breakdown

| Class ID | Class Name | Total Bounding Boxes | Percentage of Annotations | Images Containing Class |
| :---: | :--- | :---: | :---: | :---: |
| `0` | **BikeWithRider** | 3,793 | 45.84% | 2,373 (99.2%) |
| `1` | **NoHelmet** | 2,810 | 33.96% | 1,652 (69.1%) |
| `2` | **Helmet** | 1,672 | 20.21% | 1,345 (56.2%) |
| **Total** | | **8,275** | **100.00%** | **2,392** |

- **Annotations Per Image:** Minimum = 1, Maximum = 20, Average = 3.46 objects/image.
- **Missing Annotations:** 0 (Every image has a corresponding annotation in all three formats).
- **Empty Annotation Files:** 0 (No background-only images).

---

## 4. Dataset Split Status

- **Existing Split:** The uploaded dataset is currently stored as a single flat directory (`images/` with 2,392 files).
- **Subdirectories:** There are **no pre-existing** `train/`, `val/`, or `test/` folders.
- **Preservation Strategy:** Rather than physically moving or copying files into train/val/test folders, we will use non-destructive file list generation (`train.txt`, `val.txt`, `test.txt`) and a YOLOv8 dataset configuration `data.yaml`. This ensures zero disk duplication and keeps the original dataset in its pristine state.

---

## 5. Dataset Issues & Coordinate Anomalies

During rigorous bounding box validation, the dataset was found to be in very high overall condition (99.93% valid bounding boxes). However, **6 specific bounding box anomalies** were identified in the raw annotations:

1. **Zero-Height Box (Degenerate Box):**
   - **File:** `Image_00566.txt` (Line 6)
   - **Annotation:** `2 0.967712 0.626907 0.001845 0.000000`
   - **Root Cause:** In the corresponding VOC XML (`Image_00566.xml`), `ymin = 452` and `ymax = 452`, resulting in a bounding box with a height of exactly 0 pixels.

2. **Out-of-Bounds Centers (Coordinates Exceeding 1.0):**
   - **`Image_00958.txt`** (Line 3): `yc = 1.004608` (VOC `ymax = 3438` on image height `2604`)
   - **`Image_00960.txt`** (Line 1): `yc = 1.086879` (VOC `ymax = 1474` on image height `1128`)
   - **`Image_01035.txt`** (Line 0): `yc = 1.042627, h = 1.274962` (VOC `ymax = 4375` on image height `2604`)
   - **`Image_01059.txt`** (Line 0): `yc = 1.189708` (VOC `ymax = 3824` on image height `2604`)
   - **`Image_01060.txt`** (Line 2): `yc = 1.115975` (VOC `ymax = 3639` on image height `2604`)
   - **Root Cause:** The motorcycle rider extends past the lower edge of the camera view. The annotator drew boxes extending beyond the physical boundary of the image.

3. **Boundary Truncations:**
   - 31 bounding boxes extend slightly beyond image boundaries (`>5%` border overflow).
   - This is normal in real-world traffic surveillance when vehicles enter or exit the camera field of view.

---

## 6. Recommended Preprocessing & Action Plan

To prepare the dataset for training without touching the original files:

1. **Non-Destructive Coordinate Sanitization:**
   - Implement an automated coordinate clamp `[0.0, 1.0]` in the training data generator.
   - Filter out degenerate bounding boxes where width or height $\le 0$.

2. **Stratified Train / Validation / Test Split:**
   - **Recommended Split:**
     - **Training Set (70%):** ~1,674 images (~5,792 annotations)
     - **Validation Set (15%):** ~359 images (~1,241 annotations)
     - **Test Set (15%):** ~359 images (~1,242 annotations)
   - Perform class-balanced / stratified splitting so that `Helmet` (20.2%) and `NoHelmet` (34.0%) are proportionately represented in each split.

3. **YOLOv8 Dataset Descriptor (`data.yaml`):**
   - Configure paths pointing directly to `train.txt`, `val.txt`, and `test.txt`.
   - Specify class names:
     ```yaml
     names:
       0: BikeWithRider
       1: NoHelmet
       2: Helmet
     ```

4. **Image Preprocessing & Augmentation for YOLOv8:**
   - Resolution target: 640 x 640 px (standard YOLOv8 training input).
   - Augmentations: Random HSV color jitter, Mosaic, Mixup, Horizontal flip (since traffic orientation can be left/right).

---

## 7. Approval Request

The dataset inspection is complete and the modular project scaffolding has been prepared.
**Model training has not been started.**

Please review this report and provide your approval to proceed to the next stage (dataset split configuration and pipeline preparation).

