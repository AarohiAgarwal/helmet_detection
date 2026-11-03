# Stage 2: Data Preprocessing and Augmentation Report

**Project:** Real-Time Motorcycle Helmet Detection System  
**Pipeline Stage:** Stage 2 — Preprocessing, Sanitization, Augmentation & Verification  
**Date:** 2026-11-03  
**Status:** Verification PASSED — Model Training Halted Pending Approval

---

## 1. Overview & Objective

Stage 2 transforms the raw EdgeVision dataset into a production-ready, leak-free, and coordinate-sanitized format optimized for **Ultralytics YOLOv8** training. All operations were executed without altering, renaming, or moving the original dataset files.

---

## 2. Dataset Preservation & Output Architecture

- **Original Dataset (Untouched):**
  - Path: `dataset/EdgeVision Dataset/EdgeVision-Dataset/`
  - Preservation: 100% byte-for-byte identical to the uploaded state.
- **Processed Dataset:**
  - Path: `dataset/processed/`
  - YOLO Directory Standard:
    ```text
    dataset/processed/
    ├── images/
    │   ├── train/     # 2,074 images (1,674 original + 400 augmented)
    │   ├── val/       # 478 images (strictly original)
    │   └── test/      # 240 images (strictly original)
    └── labels/
        ├── train/     # 2,074 YOLO txt files
        ├── val/       # 478 YOLO txt files
        └── test/      # 240 YOLO txt files
    ```

---

## 3. Data Splitting (70:20:10 Stratified Split)

The 2,392 original images were partitioned into **70% Train**, **20% Validation**, and **10% Test** splits using a multilabel-stratified signature algorithm with fixed seed (`seed=42`).

| Split | Original Images | Augmented Images | Total Images | Image % | Bounding Boxes | Bounding Box % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,674 | 400 | **2,074** | 74.3% (eff.) | 7,281 | 75.3% |
| **Validation** | 478 | 0 | **478** | 17.1% (eff.) | 1,593 | 16.5% |
| **Test** | 240 | 0 | **240** | 8.6% (eff.) | 796 | 8.2% |
| **Total** | **2,392** | **400** | **2,792** | **100.0%** | **9,670** | **100.0%** |

### Data Leakage Guarantee
Disjoint set checks confirmed zero data overlap between splits:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
Provenance tracking also verified that all 400 augmented images were derived exclusively from the `Train` split.

---

## 4. Bounding Box Sanitization & Anomaly Resolution

All bounding boxes across the dataset were processed through a coordinate sanitizer:

1. **Degenerate Box Dropped (1 box):**
   - File: `Image_00566.txt` (Line 6: `2 0.967712 0.626907 0.001845 0.000000`)
   - Reason: Height was exactly $0.000000$ (zero area). Box safely removed.
2. **Out-of-Bounds Clamping (78 boxes):**
   - 78 bounding box coordinates that extended outside $[0.0, 1.0]$ (e.g. `yc > 1.0` where motorcycle riders extended beyond the bottom edge of the surveillance frame) were clamped strictly to the $[0.0, 1.0]$ boundary.
3. **Class ID Integrity:**
   - All annotations verified against classes: `0: BikeWithRider`, `1: NoHelmet`, `2: Helmet`. No unknown class IDs exist.

---

## 5. Training Augmentation Pipeline (Train Split ONLY)

To improve generalization without altering helmet recognizability, 6 realistic augmentations were applied to a balanced subset of 400 training images (focusing on minority helmet classes):

| Augmentation Technique | Parameter Range | Bounding Box Adjustment |
| :--- | :--- | :--- |
| **Horizontal Flip** | Mirror along vertical axis | $xc_{new} = 1.0 - xc$ |
| **Brightness & Contrast** | $\alpha \in [0.88, 1.15]$, $\beta \in [-18, 18]$ | Spatial coordinates unchanged |
| **Small Rotation** | $\theta \in [-5.0^\circ, +5.0^\circ]$ | Rotated via affine matrix & re-enclosed |
| **Random Blur** | Gaussian Blur ($k=3$) | Simulates motion blur; boxes unchanged |
| **Small Translation** | Shift by $\pm 3\%$ | $xc_{new} = xc \pm 0.03, yc_{new} = yc \pm 0.03$ (clamped) |

> [!IMPORTANT]
> **No augmentations were applied to Validation or Test sets.** Both splits contain exclusively pristine, original images with sanitized bounding boxes for unbiased benchmarking.

---

## 6. Class Distribution Breakdown Across Splits

Class proportions remained balanced and consistent across all three splits:

```
Split: TRAIN (2,074 images, 7,281 boxes)
  - Class 0 (BikeWithRider): 3,316 (45.54%)
  - Class 1 (NoHelmet):     2,531 (34.76%)
  - Class 2 (Helmet):       1,434 (19.70%)

Split: VAL (478 images, 1,593 boxes)
  - Class 0 (BikeWithRider):   738 (46.33%)
  - Class 1 (NoHelmet):       524 (32.89%)
  - Class 2 (Helmet):         331 (20.78%)

Split: TEST (240 images, 796 boxes)
  - Class 0 (BikeWithRider):   369 (46.36%)
  - Class 1 (NoHelmet):       265 (33.29%)
  - Class 2 (Helmet):         162 (20.35%)
```

---

## 7. YOLOv8 Configuration (`data.yaml`)

The dataset descriptor file `dataset/data.yaml` was generated with exact class indices:

```yaml
path: D:/Projects_Aarohi/helmet detection/dataset/processed
train: images/train
val: images/val
test: images/test

names:
  0: BikeWithRider
  1: NoHelmet
  2: Helmet
```

---

## 8. Verification & Ultralytics Compatibility

The dataset was verified using `verify_dataset.py`:
- **Image Pairing:** 100% (Every image has a matching `.txt` label file).
- **Format Integrity:** 100% valid YOLO bounding box formats.
- **Ultralytics YOLOv8 Loader Check:** `ultralytics.data.utils.check_det_dataset` succeeded without any errors or missing file warnings.

---

## 9. Sample Visualization Artifacts

Representative annotated images were generated by `visualize_annotations.py` and saved to `reports/samples/`:

- `train_sample_1_Image_00001.jpg` (Resolution: $1513 \times 801$, 7 objects: BikeWithRider, NoHelmet, Helmet)
- `train_sample_2_Image_00001_aug_flip.jpg` (Resolution: $1513 \times 801$, Horizontal Flip Augmentation)
- `train_sample_3_Image_00002.jpg` (Resolution: $355 \times 606$, BikeWithRider + Helmet)
- `train_sample_4_Image_00004.jpg` (Resolution: $561 \times 749$, BikeWithRider + NoHelmet)
- `val_sample_1_Image_00009.jpg` (Resolution: $1832 \times 968$, 9 objects)
- `test_sample_1_Image_00003.jpg` (Resolution: $217 \times 309$, BikeWithRider + Helmet)

---

## 10. Status and Next Steps

Stage 2 data preprocessing, augmentation, and verification are complete.
**Model training has not started.**

Awaiting user approval before proceeding to Stage 3 (Model Training and Hyperparameter Tuning).

