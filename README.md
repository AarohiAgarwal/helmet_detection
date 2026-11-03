# Real-Time Motorcycle Helmet Detection System Using YOLOv8

An end-to-end Deep Learning and Computer Vision application designed for automated motorcycle helmet compliance monitoring, intelligent violation alerting, and traffic road safety enforcement. Powered by **YOLOv8**, **PyTorch**, **OpenCV**, and **Streamlit**, integrated directly with the **EdgeVision Helmet Detection Dataset**.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [Technologies Used](#-technologies-used)
4. [Dataset Information](#-dataset-information)
5. [Model Architecture Explanation](#-model-architecture-explanation)
6. [YOLOv8 Training Procedure](#-yolov8-training-procedure)
7. [Installation & Dependencies](#-installation--dependencies)
8. [Project Structure](#-project-structure)
9. [How to Run the Application](#-how-to-run-the-application)
10. [How to Use Webcam Detection](#-how-to-use-webcam-detection)
11. [How to Use Video Detection](#-how-to-use-video-detection)
12. [How to View Violation History & Analytics](#-how-to-view-violation-history--analytics)
13. [Verification & Test Results](#-verification--test-results)
14. [License & Attribution](#-license--attribution)

---

## 🌟 Project Overview

Motorcycle riders face severe risks of road traffic trauma. While helmet laws are established internationally, manual surveillance cannot scale across high-density road networks. This project delivers an automated, high-FPS traffic surveillance system that:
- Detects motorcycles with riders (`BikeWithRider`), compliant helmet wearers (`Helmet`), and non-compliant violators (`NoHelmet`).
- Tracks individual vehicles and riders using **ByteTrack** across video frames.
- Establishes geometric **Intersection-over-Head ($IoH$)** spatial association to ensure helmet status is tied to the actual motorcycle rider (eliminating false alerts on distant pedestrians).
- Emits rate-limited audio warnings (5-second cooldown) and captures high-resolution evidence snapshots.
- Stores auditable violation events in a local **SQLite database** and renders real-time telemetry through a dark-themed **Streamlit dashboard**.

---

## 🎯 Objectives

1. **Non-Destructive Dataset Adaptor:** Automatically discover, sanitize, and validate the uploaded EdgeVision dataset without mutating original files.
2. **Precise Class Preservation:** Ensure exact class alignment across all training and inference modules:
   - `0`: `BikeWithRider` (Two-wheeler vehicle with rider)
   - `1`: `NoHelmet` (Infraction trigger)
   - `2`: `Helmet` (Compliant rider)
3. **High-Throughput Edge Inference:** Optimize YOLOv8n to achieve real-time responsiveness ($\sim20+$ FPS) on CPU and edge GPU hardware.
4. **Spatial Relationship Verification:** Prevent false violation alerts by validating that detected unhelmeted heads spatially overlap with the upper body zone of a tracked motorcycle rider ($IoH \ge 0.40$).
5. **Anti-Duplicate Tracking Memory:** Prevent alert flooding by suppressing repeated warnings and duplicate snapshots for the same tracked rider.
6. **Unified Central Surveillance:** Provide traffic operators with a responsive dashboard for live feeds, violation inspection, analytics, and CSV data export.

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | Ultralytics YOLOv8 (YOLOv8n) | Real-time object detection and feature extraction |
| **Framework** | PyTorch (`torch`, `torchvision`) | Deep learning runtime and model execution |
| **Object Tracking** | ByteTrack (`lapx`) | Multi-object cross-frame persistent tracking |
| **Computer Vision** | OpenCV (`cv2`) | Video stream ingestion, HUD rendering, bounding box visualization |
| **Data & Scientific** | NumPy, Pandas, SciPy | Spatial matrix calculations, data cleaning, tabular processing |
| **Visualization** | Matplotlib, Seaborn, Plotly | Training curve generation and interactive dashboard charts |
| **Persistence** | SQLite3 | Local transactional database for violation audit logs |
| **User Interface** | Streamlit | Traffic control station web dashboard |
| **Configuration** | PyYAML | Training and dataset manifest configuration |

---

## 📊 Dataset Information

The system is trained and evaluated on the **EdgeVision Motorcycle Helmet Detection Dataset**:
- **Source:** Mendeley Data, CC BY 4.0 (2026).
- **Total Images:** 2,392 JPEG images (100% verified, 0 corrupted).
- **Total Annotations:** 8,275 bounding box instances.
- **Image Dimensions:** Min: $161 \times 225$, Max: $4624 \times 4096$, Mean: $1154 \times 1411$ px.
- **Dataset Partitioning (70:20:10 Stratified Split):**
  - **Train:** 2,074 images (1,674 raw images + 400 augmented samples)
  - **Validation:** 478 pristine images
  - **Test:** 240 pristine holdout images
- **Data Leakage:** 0% overlap across train, validation, and test splits.
- **Class Distribution:**
  - `0: BikeWithRider`: 3,793 annotations (45.8%)
  - `1: NoHelmet`: 2,810 annotations (34.0%)
  - `2: Helmet`: 1,672 annotations (20.2%)

*Original dataset location:* `dataset/EdgeVision Dataset/EdgeVision-Dataset` (Preserved 100% untouched).  
*Sanitized processed dataset:* `dataset/processed/`.

---

## 🧠 Model Architecture Explanation

The application employs **YOLOv8 Nano (YOLOv8n)**, the modern state-of-the-art anchor-free vision architecture:
1. **Backbone (Modified CSPDarknet53):**
   - Utilizes $C2f$ (Cross-Stage Partial with two convolutions) modules for enhanced gradient propagation and multi-scale feature reuse.
   - Integrates Spatial Pyramid Pooling Fast ($SPPF$) to capture receptive fields of varying dimensions without increasing computational overhead.
2. **Neck (Path Aggregation Network - PANet):**
   - Combines top-down semantic context with bottom-up localized spatial cues across P3, P4, and P5 feature pyramids.
3. **Decoupled Anchor-Free Detection Head:**
   - Separates classification from bounding box regression branches.
   - Removes predefined anchor box configurations, directly predicting bounding box offsets from grid cell centers to eliminate hyperparameter tuning sensitivity.
4. **Loss Function Formulation:**
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
   - $\mathcal{L}_{\text{CIoU}}$: Complete Intersection over Union loss accounting for center alignment, overlap, and aspect ratio.
   - $\mathcal{L}_{\text{DFL}}$: Distribution Focal Loss treating coordinates as continuous distributions.
   - $\mathcal{L}_{\text{BCE}}$: Binary Cross-Entropy loss for robust multi-class separation.

---

## 🏋️ YOLOv8 Training Procedure

Transfer learning was performed using official pretrained weights (`yolov8n.pt`) fine-tuned on the sanitized EdgeVision dataset:

### Hyperparameters (`training_config.yaml`):
- **Base Model:** `yolov8n.pt`
- **Dataset Config:** `dataset/data.yaml`
- **Image Resolution:** $640 \times 640$ pixels
- **Batch Size:** 16
- **Early Stopping Patience:** 10 epochs
- **Optimizer:** Auto (AdamW) with Cosine Annealing
- **Device Support:** Automatic CUDA GPU detection with fallback to multi-threaded CPU (20 threads)
- **Random Seed:** 42 (ensures exact reproducibility)

### Execution Command:
```bash
python train.py --confirm-train --epochs 50 --batch 16 --imgsz 640
```

### Checkpoints and Metrics:
- **Best Weights Saved To:** `models/best.pt`
- **Final Weights Saved To:** `runs/detect/helmet_training/weights/last.pt`
- **Validation Results (Holdout Val Split):**
  - Precision: **0.8761** (87.6%)
  - Recall: **0.7891** (78.9%)
  - mAP @ 0.50: **0.8546** (85.5%)
  - mAP @ 0.50:0.95: **0.5928** (59.3%)

---

## 💻 Installation & Dependencies

### 1. Clone / Open Repository
```bash
cd "d:\Projects_Aarohi\helmet detection"
```

### 2. Install Required Python Packages
```bash
pip install -r requirements.txt
```

*Key packages installed:* `ultralytics`, `torch`, `torchvision`, `opencv-python`, `streamlit`, `plotly`, `lapx`, `pandas`, `numpy`, `matplotlib`, `pyyaml`.

---

## 📁 Project Structure

```text
Helmet_Detection/
├── dataset/
│   ├── EdgeVision Dataset/
│   │   └── EdgeVision-Dataset/        # Untouched original dataset (2,392 images)
│   ├── processed/                     # Sanitized 70:20:10 stratified split
│   │   ├── images/ {train, val, test}
│   │   └── labels/ {train, val, test}
│   └── data.yaml                      # YOLOv8 dataset configuration
├── models/
│   └── best.pt                        # Custom trained YOLOv8 model weights
├── runs/
│   └── detect/helmet_training/        # Training plots, confusion matrix, weights
├── database/
│   └── violations.db                  # SQLite database for audit trail
├── alerts/                            # Captured violation evidence snapshots
├── screenshots/                       # Manual user captures
├── reports/                           # Statistical evaluation reports & sample plots
├── src/                               # Modular application source code
│   ├── __init__.py
│   ├── alert_system.py                # Rate-limited audio alarm (5s cooldown)
│   ├── analytics.py                   # Plotly interactive chart generators
│   ├── camera.py                      # Multi-threaded non-blocking OpenCV stream
│   ├── config.py                      # Central constants, colors, and paths
│   ├── dashboard.py                   # Streamlit 5-page rendering logic
│   ├── database.py                    # SQLite persistence layer & CSV export
│   ├── detection.py                   # YOLOv8 + ByteTrack detection engine
│   ├── tracking.py                    # SpatialAssociator & RiderTrackRegistry
│   └── violation_manager.py           # Snapshot capture & incident logging
├── app.py                             # Streamlit web dashboard entry point
├── webcam.py                          # High-FPS standalone OpenCV monitor
├── train.py                           # YOLOv8 training pipeline
├── evaluate_model.py                  # Test set benchmark & error analysis
├── test.py                            # Standalone validation runner
├── prepare_dataset.py                 # Dataset preprocessing & augmentation
├── verify_dataset.py                  # Dataset manifest & integrity validator
├── visualize_annotations.py           # Annotation visualizer
├── run_full_system_test.py            # Comprehensive 12-module integration test
├── test_dashboard_backend.py          # Dashboard backend integration test
├── test_stage6_alerts.py              # Alert & tracking verification test
├── requirements.txt                   # Dependency manifest
├── PROJECT_REPORT.md                  # Comprehensive architectural report
├── FINAL_TEST_REPORT.md               # Quality assurance test matrix
└── README.md                          # Project documentation
```

---

## 🚀 How to Run the Application

### Launch the Streamlit Web Control Center
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard offers **5 specialized surveillance pages**:
1. **📊 Dashboard Overview:** Real-time KPI summary cards, compliance ratios, infrastructure health, and recent incident logs.
2. **📹 Live Detection:** High-FPS live webcam monitoring, start/stop toggle, audio mute switch, confidence slider, and HUD overlay.
3. **📋 Violation History:** Searchable SQLite incident table, class and source filtering, photo lightbox, and CSV export.
4. **📈 Analytics & Trends:** Interactive Plotly compliance bar charts, distribution donuts, daily violation trends, and confidence histograms.
5. **🎞️ Video Detection:** Upload traffic video files (.mp4, .avi, .mov), run model inference with tracking progress, and sync violations to SQLite.

---

## 📹 How to Use Webcam Detection

### Option A: Via Streamlit Dashboard
1. Run `streamlit run app.py`.
2. Navigate to **"📹 Live Detection"** from the left sidebar.
3. Select your webcam index (default: `0`).
4. Set your desired confidence threshold (default: `0.40`).
5. Toggle **"Start Camera"** to begin real-time surveillance.

### Option B: Standalone High-FPS OpenCV Streaming
Run `webcam.py` directly from your terminal:
```bash
# 1. Live Webcam Feed
python webcam.py --source 0 --conf 0.40

# 2. Feed with Output Video Recording
python webcam.py --source 0 --record recorded_stream.mp4
```

**Interactive Keyboard Controls During Streaming:**
- `q` or `ESC`: Stop camera and exit cleanly.
- `+` or `=`: Increase confidence threshold by $+0.05$.
- `-` or `_`: Decrease confidence threshold by $-0.05$.
- `a`: Toggle sound alerts ON / OFF.
- `s`: Save instant evidence screenshot into `screenshots/`.
- `SPACE`: Pause / Resume playback.

**Bounding Box Color Standard:**
- 🟢 **Green:** `Helmet` (Compliant)
- 🔴 **Red:** `NoHelmet` (Violation)
- 🔵 **Blue:** `BikeWithRider` (Motorcycle)

---

## 🎞️ How to Use Video Detection

### Option A: Via Streamlit Dashboard
1. Run `streamlit run app.py`.
2. Navigate to **"🎞️ Video Detection"** from the left sidebar.
3. Click **"Select Video File"** and upload an `.mp4`, `.avi`, or `.mov` clip (e.g. `test_clip.mp4`).
4. Adjust the confidence threshold and click **"🚀 Process Video File"**.
5. Watch the live progress bar and preview frame updates.
6. Review the summary metrics (total frames, confirmed violations, helmets detected). All confirmed infractions are automatically stored in `database/violations.db`.

### Option B: Via Command Line
```bash
python webcam.py --source test_clip.mp4 --conf 0.40
```

---

## 📋 How to View Violation History & Analytics

1. Run `streamlit run app.py`.
2. Navigate to **"📋 Violation History"**:
   - Use the **Search bar** to query by timestamp, class, or path.
   - Filter by class (`NoHelmet`, `Helmet`, `BikeWithRider`) or source (`webcam`, `uploaded_video`).
   - Click **"📥 Export Filtered Records (CSV)"** to download the records for reporting.
   - Use the **Evidence Snapshot Inspector** dropdown to inspect the high-resolution photo snapshot captured for any specific incident ID.
3. Navigate to **"📈 Analytics & Trends"**:
   - Inspect the **Compliance Snapshot** (Helmet vs. No-Helmet comparison).
   - View the **Detection Distribution Donut** chart.
   - Analyze the **Daily Incident Trend** to identify peak violation periods.
   - Review the **Class-Wise Statistics Table** for total counts and mean confidence.

---

## ✅ Verification & Test Results

The full application was tested across all 12 functional modules via `run_full_system_test.py`:

```bash
python run_full_system_test.py
```

**Test Output:**
```text
====================================================================
REAL-TIME MOTORCYCLE HELMET DETECTION SYSTEM: FULL INTEGRATION TEST
====================================================================
[TEST 1] Verifying Custom Trained YOLOv8 Model & Class Mappings...
  - Model file confirmed: models/best.pt (6.5 MB)
  - data.yaml class mappings verified: {0: 'BikeWithRider', 1: 'NoHelmet', 2: 'Helmet'}
  - YOLOv8 runtime classes verified: {0: 'BikeWithRider', 1: 'NoHelmet', 2: 'Helmet'}
  -> PASSED: Model and class mappings are 100% consistent across all components.

[TEST 2] Verifying Preprocessed Dataset & Manifests...
  - Dataset splits verified: Train=2074, Val=478, Test=240
  -> PASSED: Dataset preprocessing split (70:20:10) preserved with zero data leakage.

[TEST 3] Verifying Frame Detection & ByteTrack Object Tracking...
  - Frame 1 inference completed: latency=3740.1ms
  - Frame 2 tracking completed: latency=40.8ms
  -> PASSED: Real-time inference and ByteTrack tracker operational.

[TEST 4] Verifying Spatial Association & Rate-Limited Sound Alerts...
  - Audio alarm and 5-second cooldown verified.
  - Spatial association correctly validated rider head relationship.
  - Rider track memory successfully prevents repeated alerts for same rider.
  -> PASSED: Violation detection, spatial association, and sound alerts verified.

[TEST 5] Verifying SQLite Database Operations & Incident Logging...
  - SQLite CRUD operations, indexing, filtering, metrics, and CSV export verified.
  -> PASSED: Database persistence layer is robust and verified.

[TEST 6] Verifying Plotly Visualizations Engine...
  - All 4 Plotly interactive charts generated cleanly.
  -> PASSED: Analytics engine fully operational.

[TEST 7] Verifying Video Processing Pipeline on Sample Video...
  - Sample video loaded: 60 frames @ 15.0 FPS
  - Successfully processed 5 frames with ByteTrack tracking.
  -> PASSED: Video detection pipeline verified.

====================================================================
ALL 7 TEST SUITES (12 INTEGRATION MODULES) PASSED IN 7.91s!
====================================================================
```

*For comprehensive QA matrices, see [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md).*  
*For architectural documentation, see [PROJECT_REPORT.md](PROJECT_REPORT.md).*

---

## 🛡️ License & Attribution

- **Dataset:** EdgeVision Dataset (Licensed under Creative Commons Attribution 4.0 International, CC BY 4.0).
- **Authors:** Kandarp Gajjar, Nancy Patel, Ridham Patel, Krutika Patel, Chrisha Dabhi, Himani Trivedi, Mahesh Goyani, Chintan Bhatt (2026), Mendeley Data, V1.
