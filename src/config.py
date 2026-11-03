"""
Central Configuration Module for Motorcycle Helmet Detection System.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths (Directly pointing to uploaded EdgeVision dataset)
DATASET_ROOT = PROJECT_ROOT / "dataset" / "EdgeVision Dataset" / "EdgeVision-Dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_YOLO_DIR = DATASET_ROOT / "labels" / "yolo"
LABELS_VOC_DIR = DATASET_ROOT / "labels" / "voc_xml"
LABELS_COCO_DIR = DATASET_ROOT / "labels" / "coco"

# Processed Dataset Paths (Sanitized, augmented, and split)
PROCESSED_DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATASET_ROOT / "images"
PROCESSED_LABELS_DIR = PROCESSED_DATASET_ROOT / "labels"

# Generated Split & Data Config Paths
DATA_YAML_PATH = PROJECT_ROOT / "dataset" / "data.yaml"

# Output & Artifact Paths
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "violations.db"
ALERTS_DIR = PROJECT_ROOT / "alerts"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model Default Weights
DEFAULT_PRETRAINED_MODEL = "yolov8n.pt"  # Nano for fast real-time inference
BEST_MODEL_PATH = MODELS_DIR / "best.pt"

# Class Definitions
CLASS_NAMES = {
    0: "BikeWithRider",
    1: "NoHelmet",
    2: "Helmet"
}
CLASS_MAPPING = CLASS_NAMES

# Visualization Colors (BGR for OpenCV: Blue, Red, Green)
CLASS_COLORS = {
    0: (255, 0, 0),      # Blue for BikeWithRider
    1: (0, 0, 255),      # Red for NoHelmet (Violation)
    2: (0, 255, 0)       # Green for Helmet (Compliant)
}

# Detection & Tracking Parameters
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Alert Settings
ALERT_COOLDOWN_SECONDS = 3.0  # Prevent flooding violations for same vehicle
SAVE_ALERT_CROPS = True

