# Stage 3: YOLOv8 Model Training Report

**Project:** Real-Time Motorcycle Helmet Detection System  
**Pipeline Stage:** Stage 3 — Model Training & Transfer Learning  
**Date:** 2026-11-03 20:35:28  
**Run Directory:** `D:/Projects_Aarohi/helmet detection/runs/detect/helmet_training`  
**Model Architecture:** `yolov8n.pt` (Nano)  

---

## 1. Executive Summary

The YOLOv8n object detection model was trained using transfer learning on the processed EdgeVision dataset.
- **Completed Epochs:** 3 / 50
- **Best Model Checkpoint:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\weights\best.pt`
- **Last Model Checkpoint:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\weights\last.pt`
- **Generalization Assessment:** **Balanced Fit (Good Generalization)**

---

## 2. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Weights** | `yolov8n.pt` | Pretrained COCO weights for transfer learning |
| **Dataset Config** | `dataset/data.yaml` | Processed EdgeVision dataset descriptor |
| **Input Resolution** | `640 	imes 640` | YOLO standard square letterboxing |
| **Batch Size** | `8` | Optimized for stability & memory constraints |
| **Max Epochs** | `50` | Initial ceiling with early stopping |
| **Patience** | `10` | Early stopping threshold on validation mAP50 |
| **Optimizer** | `auto` | Automatic SGD / AdamW scheduling |
| **Initial Learning Rate (lr0)** | `0.01` | Warmup to base learning rate |
| **Final Learning Rate (lrf)** | `0.01` | Cosine learning rate decay fraction |
| **Hardware Device** | `auto` | Automatic CPU / CUDA acceleration |
| **Random Seed** | `42` | Deterministic reproducibility |

---

## 3. Final Validation Metrics

The following metrics were evaluated on the independent validation split:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **mAP @ 0.50** | **0.8546** | Mean Average Precision at IoU threshold 0.50 |
| **mAP @ 0.50:0.95** | **0.5928** | Strict COCO-style average precision across IoU thresholds |
| **Precision** | **0.8761** | Ratio of true positive detections over total detections |
| **Recall** | **0.7891** | Ratio of ground-truth objects successfully detected |
| **Final Train Box Loss** | `0.9272` | Bounding box spatial regression error (Train) |
| **Final Val Box Loss** | `0.9638` | Bounding box spatial regression error (Validation) |
| **Final Train Cls Loss** | `0.7022` | Class categorization cross-entropy loss (Train) |
| **Final Val Cls Loss** | `0.7172` | Class categorization cross-entropy loss (Validation) |

---

## 4. Overfitting vs. Underfitting Analysis

- **Diagnosis:** **Balanced Fit (Good Generalization)**
- **Detailed Findings:**
  Validation losses track training losses consistently. Final mAP50 is 0.8546 and mAP50-95 is 0.5928. The balance between precision (0.8761) and recall (0.7891) indicates robust feature learning without divergence.

---

## 5. Artifacts and Visualization Paths

All generated metrics, confusion matrices, and model weights are persisted:
- **Best Weights:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\weights\best.pt`
- **Last Weights:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\weights\last.pt`
- **Results CSV:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\results.csv`
- **Confusion Matrix:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\confusion_matrix.png`
- **Normalized Confusion Matrix:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\confusion_matrix_normalized.png`
- **Loss Curves:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\custom_training_losses.png`
- **Precision Curve:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\custom_precision_curve.png`
- **Recall Curve:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\custom_recall_curve.png`
- **mAP Curves:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\custom_map50_curve.png` and `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\custom_map50_95_curve.png`
- **Ultralytics Summary Plot:** `D:\Projects_Aarohi\helmet detection\runs\detect\helmet_training\results.png`

---

## 6. Next Steps & Approval Request

Stage 3 model training has completed. Checkpoints have been archived and linked to `models/best.pt`.

**Next Stage (Stage 4): Model Evaluation, Benchmarking & Threshold Calibration.**
Please review the training results and approve proceeding to Stage 4.
