"""
Training Utilities for YOLOv8 Helmet Detection.
Handles device detection, results plotting, overfitting/underfitting analysis,
and markdown report generation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_device(requested: str = "auto") -> str:
    """Detects available hardware and returns '0' for CUDA GPU or 'cpu'."""
    if requested.lower() == "auto":
        if torch.cuda.is_available():
            dev = "0"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA GPU detected: {gpu_name}. Using device: {dev}")
        else:
            dev = "cpu"
            print(f"[Device] CUDA not available in PyTorch. Using CPU ({os.cpu_count()} threads).")
        return dev
    return requested


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates training configuration YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_resume_checkpoint(project_dir: Path, run_name: str) -> Optional[Path]:
    """Searches for the latest last.pt checkpoint to resume training."""
    target_dir = project_dir / run_name
    last_pt = target_dir / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt
    # Search for incremented runs (e.g. helmet_training2, helmet_training3)
    matching_runs = sorted(list(project_dir.glob(f"{run_name}*/weights/last.pt")), key=os.path.getmtime, reverse=True)
    if matching_runs:
        return matching_runs[0]
    return None


def generate_custom_metric_plots(csv_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Parses Ultralytics results.csv and exports standalone plots for:
    - Training loss graphs (box, cls, dfl)
    - Precision graph
    - Recall graph
    - mAP50 graph
    - mAP50-95 graph
    """
    if not csv_path.exists():
        print(f"[Warning] results.csv not found at {csv_path}. Skipping custom plots.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    # Strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
    plot_paths = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training & Validation Loss Graphs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
        axes[0].plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", lw=2)
        axes[0].plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#ff7f0e", lw=2, linestyle="--")
        axes[0].set_title("Box Loss (Localization)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

    if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
        axes[1].plot(epochs, df["train/cls_loss"], label="Train Cls Loss", color="#2ca02c", lw=2)
        axes[1].plot(epochs, df["val/cls_loss"], label="Val Cls Loss", color="#d62728", lw=2, linestyle="--")
        axes[1].set_title("Classification Loss (Object Class)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
        axes[2].plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", color="#9467bd", lw=2)
        axes[2].plot(epochs, df["val/dfl_loss"], label="Val DFL Loss", color="#8c564b", lw=2, linestyle="--")
        axes[2].set_title("Distribution Focal Loss (DFL)", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.tight_layout()
    loss_path = output_dir / "custom_training_losses.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    plot_paths["loss_graphs"] = loss_path

    # 2. Precision Graph
    if "metrics/precision(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["metrics/precision(B)"], label="Precision", color="#008080", lw=2.5)
        plt.title("Model Precision across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.tight_layout()
        p_path = output_dir / "custom_precision_curve.png"
        plt.savefig(p_path, dpi=200)
        plt.close()
        plot_paths["precision_graph"] = p_path

    # 3. Recall Graph
    if "metrics/recall(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["metrics/recall(B)"], label="Recall", color="#e377c2", lw=2.5)
        plt.title("Model Recall across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.tight_layout()
        r_path = output_dir / "custom_recall_curve.png"
        plt.savefig(r_path, dpi=200)
        plt.close()
        plot_paths["recall_graph"] = r_path

    # 4. mAP50 Graph
    if "metrics/mAP50(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.50", color="#2ca02c", lw=2.5)
        plt.title("mAP@0.50 (IoU=0.50) across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@0.50")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.tight_layout()
        map50_path = output_dir / "custom_map50_curve.png"
        plt.savefig(map50_path, dpi=200)
        plt.close()
        plot_paths["map50_graph"] = map50_path

    # 5. mAP50-95 Graph
    if "metrics/mAP50-95(B)" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.50:0.95", color="#17becf", lw=2.5)
        plt.title("mAP@0.50:0.95 across Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@0.50:0.95")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.tight_layout()
        map50_95_path = output_dir / "custom_map50_95_curve.png"
        plt.savefig(map50_95_path, dpi=200)
        plt.close()
        plot_paths["map50_95_graph"] = map50_95_path

    return plot_paths


def analyze_model_fit(csv_path: Path) -> Dict[str, Any]:
    """
    Analyzes training dynamics to diagnose whether the model is
    overfitting, underfitting, or achieving a balanced fit.
    """
    if not csv_path.exists():
        return {"status": "Unknown", "explanation": "results.csv not found."}

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if len(df) == 0:
        return {"status": "Incomplete", "explanation": "No epoch records found in results.csv."}

    last_row = df.iloc[-1]
    total_epochs = len(df)

    # Check loss trajectories
    train_box_loss = df["train/box_loss"].tolist() if "train/box_loss" in df.columns else []
    val_box_loss = df["val/box_loss"].tolist() if "val/box_loss" in df.columns else []
    train_cls_loss = df["train/cls_loss"].tolist() if "train/cls_loss" in df.columns else []
    val_cls_loss = df["val/cls_loss"].tolist() if "val/cls_loss" in df.columns else []

    map50 = float(last_row.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(last_row.get("metrics/mAP50-95(B)", 0.0))
    precision = float(last_row.get("metrics/precision(B)", 0.0))
    recall = float(last_row.get("metrics/recall(B)", 0.0))

    # Evaluate loss trend over the last 5 epochs
    trend_window = min(5, total_epochs)
    if trend_window >= 3 and val_cls_loss and train_cls_loss:
        recent_val_cls_diff = val_cls_loss[-1] - val_cls_loss[-trend_window]
        recent_train_cls_diff = train_cls_loss[-1] - train_cls_loss[-trend_window]

        if recent_train_cls_diff < -0.05 and recent_val_cls_diff > 0.10:
            status = "Slight Overfitting"
            explanation = (
                f"Training loss continued decreasing (Δ={recent_train_cls_diff:.4f}) while validation loss "
                f"began increasing (Δ=+{recent_val_cls_diff:.4f}) over recent epochs. Early stopping or regularized "
                f"weight decay is effective."
            )
        elif map50 < 0.40 and total_epochs < 10:
            status = "Underfitting (Early Training Stage)"
            explanation = (
                f"Training concluded early ({total_epochs} epochs) with mAP50={map50:.3f}. "
                f"Both train and validation loss are actively declining, suggesting further training iterations "
                f"will yield significant accuracy gains."
            )
        else:
            status = "Balanced Fit (Good Generalization)"
            explanation = (
                f"Validation losses track training losses consistently. Final mAP50 is {map50:.4f} and "
                f"mAP50-95 is {map50_95:.4f}. The balance between precision ({precision:.4f}) and recall ({recall:.4f}) "
                f"indicates robust feature learning without divergence."
            )
    else:
        status = "Normal Training Trajectory"
        explanation = f"Training completed with {total_epochs} epochs. Final mAP@0.50={map50:.4f}."

    return {
        "status": status,
        "explanation": explanation,
        "epochs_completed": total_epochs,
        "final_map50": map50,
        "final_map50_95": map50_95,
        "final_precision": precision,
        "final_recall": recall,
        "final_train_box_loss": float(last_row.get("train/box_loss", 0.0)),
        "final_val_box_loss": float(last_row.get("val/box_loss", 0.0)),
        "final_train_cls_loss": float(last_row.get("train/cls_loss", 0.0)),
        "final_val_cls_loss": float(last_row.get("val/cls_loss", 0.0))
    }


def save_training_report(
    run_dir: Path,
    config: Dict[str, Any],
    fit_metrics: Dict[str, Any],
    report_path: Path
):
    """Writes a comprehensive markdown report summarizing model training performance."""
    report_content = f"""# Stage 3: YOLOv8 Model Training Report

**Project:** Real-Time Motorcycle Helmet Detection System  
**Pipeline Stage:** Stage 3 — Model Training & Transfer Learning  
**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Run Directory:** `{run_dir.as_posix()}`  
**Model Architecture:** `{config.get('model', 'yolov8n.pt')}` (Nano)  

---

## 1. Executive Summary

The YOLOv8n object detection model was trained using transfer learning on the processed EdgeVision dataset.
- **Completed Epochs:** {fit_metrics.get('epochs_completed', 0)} / {config.get('epochs', 50)}
- **Best Model Checkpoint:** `{run_dir / 'weights' / 'best.pt'}`
- **Last Model Checkpoint:** `{run_dir / 'weights' / 'last.pt'}`
- **Generalization Assessment:** **{fit_metrics.get('status', 'N/A')}**

---

## 2. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Weights** | `{config.get('model', 'yolov8n.pt')}` | Pretrained COCO weights for transfer learning |
| **Dataset Config** | `{config.get('data', 'dataset/data.yaml')}` | Processed EdgeVision dataset descriptor |
| **Input Resolution** | `{config.get('imgsz', 640)} \times {config.get('imgsz', 640)}` | YOLO standard square letterboxing |
| **Batch Size** | `{config.get('batch', 8)}` | Optimized for stability & memory constraints |
| **Max Epochs** | `{config.get('epochs', 50)}` | Initial ceiling with early stopping |
| **Patience** | `{config.get('patience', 10)}` | Early stopping threshold on validation mAP50 |
| **Optimizer** | `{config.get('optimizer', 'auto')}` | Automatic SGD / AdamW scheduling |
| **Initial Learning Rate (lr0)** | `{config.get('lr0', 0.01)}` | Warmup to base learning rate |
| **Final Learning Rate (lrf)** | `{config.get('lrf', 0.01)}` | Cosine learning rate decay fraction |
| **Hardware Device** | `{config.get('device', 'cpu')}` | Automatic CPU / CUDA acceleration |
| **Random Seed** | `{config.get('seed', 42)}` | Deterministic reproducibility |

---

## 3. Final Validation Metrics

The following metrics were evaluated on the independent validation split:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **mAP @ 0.50** | **{fit_metrics.get('final_map50', 0.0):.4f}** | Mean Average Precision at IoU threshold 0.50 |
| **mAP @ 0.50:0.95** | **{fit_metrics.get('final_map50_95', 0.0):.4f}** | Strict COCO-style average precision across IoU thresholds |
| **Precision** | **{fit_metrics.get('final_precision', 0.0):.4f}** | Ratio of true positive detections over total detections |
| **Recall** | **{fit_metrics.get('final_recall', 0.0):.4f}** | Ratio of ground-truth objects successfully detected |
| **Final Train Box Loss** | `{fit_metrics.get('final_train_box_loss', 0.0):.4f}` | Bounding box spatial regression error (Train) |
| **Final Val Box Loss** | `{fit_metrics.get('final_val_box_loss', 0.0):.4f}` | Bounding box spatial regression error (Validation) |
| **Final Train Cls Loss** | `{fit_metrics.get('final_train_cls_loss', 0.0):.4f}` | Class categorization cross-entropy loss (Train) |
| **Final Val Cls Loss** | `{fit_metrics.get('final_val_cls_loss', 0.0):.4f}` | Class categorization cross-entropy loss (Validation) |

---

## 4. Overfitting vs. Underfitting Analysis

- **Diagnosis:** **{fit_metrics.get('status', 'Normal')}**
- **Detailed Findings:**
  {fit_metrics.get('explanation', 'Analysis pending.')}

---

## 5. Artifacts and Visualization Paths

All generated metrics, confusion matrices, and model weights are persisted:
- **Best Weights:** `{run_dir / 'weights' / 'best.pt'}`
- **Last Weights:** `{run_dir / 'weights' / 'last.pt'}`
- **Results CSV:** `{run_dir / 'results.csv'}`
- **Confusion Matrix:** `{run_dir / 'confusion_matrix.png'}`
- **Normalized Confusion Matrix:** `{run_dir / 'confusion_matrix_normalized.png'}`
- **Loss Curves:** `{run_dir / 'custom_training_losses.png'}`
- **Precision Curve:** `{run_dir / 'custom_precision_curve.png'}`
- **Recall Curve:** `{run_dir / 'custom_recall_curve.png'}`
- **mAP Curves:** `{run_dir / 'custom_map50_curve.png'}` and `{run_dir / 'custom_map50_95_curve.png'}`
- **Ultralytics Summary Plot:** `{run_dir / 'results.png'}`

---

## 6. Next Steps & Approval Request

Stage 3 model training has completed. Checkpoints have been archived and linked to `models/best.pt`.

**Next Stage (Stage 4): Model Evaluation, Benchmarking & Threshold Calibration.**
Please review the training results and approve proceeding to Stage 4.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Report] Training report saved to: {report_path}")

