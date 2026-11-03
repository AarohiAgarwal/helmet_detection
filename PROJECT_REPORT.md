# Comprehensive Project Report
# Real-Time Motorcycle Helmet Detection System Using YOLOv8

**Date:** March 2026  
**Technology Stack:** Python, Ultralytics YOLOv8, PyTorch, OpenCV, SQLite, Streamlit, Plotly, Pandas, NumPy  
**Dataset:** EdgeVision Motorcycle Helmet Detection Dataset  

---

## 1. Executive Summary

Motorcycle road traffic injuries represent one of the leading global causes of severe vehicular fatalities. While helmet usage mandates significantly mitigate head trauma severity, automated enforcement is routinely hindered by occlusions, high vehicle density, and inaccurate rider-to-head association.

This project delivers an end-to-end, production-grade **Real-Time Motorcycle Helmet Detection and Traffic Enforcement System**. Operating directly on the uploaded **EdgeVision Helmet Detection Dataset**, the application encompasses non-destructive data preprocessing and augmentation, custom YOLOv8 model training via transfer learning, exhaustive test-set benchmark evaluation, high-framerate real-time OpenCV tracking with ByteTrack, an intelligent anti-duplicate alert system, SQLite audit persistence, and an interactive dark-themed Streamlit traffic control station.

---

## 2. Project Objectives

1. **Autonomous Dataset Adaptability:** Automatically discover, sanitize, and validate the uploaded EdgeVision dataset without mutating, duplicating, or corrupting original source assets.
2. **Standardized Class Alignment:** Preserve exact dataset class definitions across all components:
   - `0`: `BikeWithRider` (Two-wheeler context)
   - `1`: `NoHelmet` (Violation trigger)
   - `2`: `Helmet` (Compliant)
3. **Optimized Deep Learning Inference:** Train YOLOv8n (Nano) to maximize mean Average Precision ($mAP@0.50$ and $mAP@0.50:0.95$) while ensuring high inference throughput ($\sim18$ to $30+$ FPS) on commodity CPU and edge GPU hardware.
4. **Spatial Relationship Association:** Guarantee that helmet status is associated specifically with a motorcycle rider using geometric Intersection-over-Head ($IoH$) spatial containment, eliminating false alerts triggered by distant pedestrians or roadside bystanders.
5. **Intelligent Alerting & Audit Logging:** Enforce a 5-second audio cooldown, anti-duplicate rider memory across tracking frames, automatic high-resolution evidence snapshot capture, and structured SQLite/CSV incident logging.
6. **Centralized Surveillance Dashboard:** Deliver an intuitive, responsive Streamlit dashboard featuring live telemetry, searchable violation history, interactive Plotly compliance analytics, and drag-and-drop video file batch inference.

---

## 3. Dataset Characteristics & Preprocessing Pipeline

### 3.1 Raw Dataset Audit
- **Total Images:** 2,392 pristine JPEG images.
- **Total Annotations:** 8,275 bounding box instances.
- **Coordinate Audit:** 6 edge anomalies identified (1 degenerate box with height $0.0$, 5 boxes with center-$y > 1.0$).
- **Sanitization Strategy:** Anomalies safely clamped and filtered in a separate `dataset/processed/` directory; the original `dataset/EdgeVision Dataset/EdgeVision-Dataset` remains 100% untouched.

### 3.2 Non-Destructive Partitioning (70:20:10 Stratified Split)
- **Train Set:** 2,074 images (1,674 raw sanitized images + 400 augmented samples).
- **Validation Set:** 478 pristine images (unaugmented to preserve genuine validation loss).
- **Test Set:** 240 pristine images (unaugmented unseen holdout).
- **Data Leakage:** Verified 0% image hash overlap across splits.

### 3.3 Domain-Specific Augmentation
To improve model resilience without distorting helmet geometry, augmentations were restricted to:
- Horizontal reflection (probabilistic mirror flip)
- Contrast / brightness jitter ($\pm15\%$)
- Mild affine translations ($\pm5\%$) and slight scale variations
- Gaussian blur ($\sigma \le 1.0$)

---

## 4. Model Architecture & Training Methodology

### 4.1 YOLOv8 Nano Architecture
YOLOv8 represents a modern, state-of-the-art anchor-free object detector:
1. **Backbone:** Modified CSPDarknet53 with $C2f$ (Cross-Stage Partial with two convolutions) modules for enhanced gradient flow and multi-scale feature extraction.
2. **Neck:** Path Aggregation Network (PANet) fusing low-level spatial detail with high-level semantic abstractions.
3. **Decoupled Head:** Separates classification and bounding-box regression tasks into independent convolutional branches, accelerating convergence.
4. **Anchor-Free Mechanism:** Predicts distances to bounding box edges from center points directly, removing sensitivity to predefined anchor box ratios.

### 4.2 Loss Formulation
The objective function balances classification and geometric localization:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
- **Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$):** Penalizes overlap distance, aspect ratio deviation, and center offset.
- **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$):** Treats bounding box coordinates as continuous distributions around grid points.
- **Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$):** Multi-label classification loss per detection anchor.

### 4.3 Training Hyperparameters
- **Base Weights:** `yolov8n.pt` (transfer learning)
- **Input Resolution:** $640 \times 640$ pixels
- **Batch Size:** 16 (optimized for multi-threaded CPU/GPU throughput)
- **Optimizer:** AdamW with cosine learning rate scheduling
- **Patience:** 10 epochs early stopping
- **Reproducibility:** Fixed random seed (42)

---

## 5. Model Evaluation & Benchmark Metrics

The trained model was benchmarked on the separate 240-image holdout test split:

| Class | Precision ($P$) | Recall ($R$) | mAP@0.50 | mAP@0.50:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes (Overall)** | **0.658** | **0.621** | **0.638** | **0.372** |
| `BikeWithRider` (Class 0) | 0.812 | 0.795 | 0.824 | 0.521 |
| `NoHelmet` (Class 1) | 0.614 | 0.582 | 0.591 | 0.318 |
| `Helmet` (Class 2) | 0.548 | 0.486 | 0.499 | 0.277 |

### Analysis of Error Patterns:
1. **BikeWithRider:** High precision and recall due to distinctive motorcycle profile and large bounding box area.
2. **Small Scale Head Detections:** Severe camera distance ($< 32 \times 32$ pixels) occasionally caused confusion between dark caps and dark helmets.
3. **Occlusions:** Pillion riders partially obscured by the primary rider presented false negative challenges.

---

## 6. Spatial Association & Intelligent Alert System

### 6.1 Spatial Rider-Head Containment
A fundamental innovation of this implementation is the **SpatialAssociator**. A `NoHelmet` detection alone does not prove a violation has occurred unless it belongs to an active motorcycle rider:
- Let $B_{\text{bike}} = [x_1, y_1, x_2, y_2]$ be the rider bounding box.
- The **Rider Upper Torso / Head Zone** is defined as $Z_{\text{head}} = [x_1, y_1, x_2, y_1 + 0.50 \cdot h]$.
- For any detected head box $B_{\text{head}}$, we compute the **Intersection-over-Head ($IoH$)**:
  $$IoH(B_{\text{head}}, Z_{\text{head}}) = \frac{\text{Area}(B_{\text{head}} \cap Z_{\text{head}})}{\text{Area}(B_{\text{head}})}$$
- If $IoH \ge 0.40$ and the horizontal centroid falls within rider bounds, the head is confirmed as belonging to that rider. Isolated pedestrian heads are ignored.

### 6.2 Anti-Duplicate Rider Tracking
Using Ultralytics ByteTrack, unique `track_id` assignments persist across frames. The `RiderTrackRegistry` logs flagged track IDs in an internal LRU cache, ensuring that an unhelmeted rider passing through the camera view triggers **exactly one audio alert and one screenshot**, rather than dozens of repeated alerts every second.

### 6.3 Rate-Limited Audio & Evidence Capture
- **Cooldown:** 5.0-second global sound cooldown.
- **Audio Output:** Native Windows `winsound.Beep` fallback for universal compatibility without third-party audio drivers.
- **Snapshot Manager:** High-resolution JPEG snapshots saved directly to `alerts/` and indexed in `database/violations.db`.

---

## 7. Streamlit Traffic Monitoring Station

The Streamlit web application (`app.py`) provides five comprehensive surveillance views:
1. **Dashboard Overview:** Real-time KPI summary cards, compliance ratio gauges, infrastructure health status, and live violation previews.
2. **Live Detection:** Real-time webcam stream with start/stop toggle, emergency stop button, audio alarm mute, dynamic confidence slider, and HUD overlay.
3. **Violation History:** Full audit table with search filtering, evidence photo lightbox viewer, and one-click CSV export.
4. **Analytics & Trends:** Interactive Plotly charts including Compliance vs. Infraction bar charts, Detection distribution donuts, Daily incident trends, and Confidence histograms.
5. **Video Detection:** Drag-and-drop batch video file processor (.mp4/.avi/.mov) with frame-by-frame progress, annotated video output, and SQLite syncing.

---

## 8. Conclusion & Future Work

The integrated Real-Time Motorcycle Helmet Detection System demonstrates robust engineering, high responsiveness, and strict adherence to dataset provenance. Future extensions may incorporate:
- Automatic License Plate Recognition (ALPR / ANPR) integration.
- Night-time infrared and low-light enhancement pipelines.
- Multi-camera edge deployment via RTSP streaming.

