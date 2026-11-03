"""
Production Training Pipeline for Real-Time Helmet Detection with YOLOv8.
Trains YOLOv8n on the processed EdgeVision dataset with transfer learning,
checkpointing, early stopping, and metric analysis.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RUNS_DIR
from training_utils import (
    get_device,
    load_config,
    find_resume_checkpoint,
    generate_custom_metric_plots,
    analyze_model_fit,
    save_training_report
)

CONFIG_PATH = PROJECT_ROOT / "training_config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Helmet Detection Model")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', or 'auto')")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint (last.pt)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to resume from")
    return parser.parse_args()


def run_training(args):
    print("=" * 68)
    print("STAGE 3: YOLOV8 MODEL TRAINING PIPELINE")
    print("=" * 68)

    # 1. Load Configuration
    config_file = Path(args.config)
    cfg = load_config(config_file)

    # Apply CLI overrides if provided
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 50)
    batch = args.batch if args.batch is not None else cfg.get("batch", 8)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640)
    patience = cfg.get("patience", 10)
    seed = cfg.get("seed", 42)
    workers = cfg.get("workers", 2)
    data_yaml = PROJECT_ROOT / cfg.get("data", "dataset/data.yaml")
    model_name = cfg.get("model", "yolov8n.pt")

    # 2. Device Selection (CUDA GPU or CPU automatically)
    req_device = args.device if args.device is not None else cfg.get("device", "auto")
    device = get_device(req_device)

    # 3. Verify Dataset Configuration
    if not data_yaml.exists():
        print(f"[Error] data.yaml not found at {data_yaml}!")
        print("Please run 'python prepare_dataset.py' first.")
        sys.exit(1)

    # Print summary
    print(f"\n[Configuration Overview]")
    print(f"  Base Model:       {model_name}")
    print(f"  Dataset Config:   {data_yaml}")
    print(f"  Epochs (Max):     {epochs}")
    print(f"  Batch Size:       {batch}")
    print(f"  Resolution:       {imgsz}x{imgsz}")
    print(f"  Early Stopping:   Patience {patience}")
    print(f"  Device:           {device}")
    print(f"  Random Seed:      {seed}")
    print(f"  Workers:          {workers}")

    # 4. Initialize or Resume Model
    from ultralytics import YOLO

    project_dir = PROJECT_ROOT / cfg.get("project", "runs/detect")
    run_name = cfg.get("name", "helmet_training")
    project_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt = None
    if args.resume or args.checkpoint:
        if args.checkpoint:
            resume_ckpt = Path(args.checkpoint)
        else:
            resume_ckpt = find_resume_checkpoint(project_dir, run_name)

        if resume_ckpt and resume_ckpt.exists():
            print(f"\n[Resume] Found previous checkpoint: {resume_ckpt}")
            print("         Resuming training from checkpoint...")
            model = YOLO(str(resume_ckpt))
        else:
            print(f"[Warning] No previous checkpoint found to resume. Initializing fresh model {model_name}...")
            model = YOLO(model_name)
    else:
        print(f"\n[Init] Initializing model with pretrained weights: {model_name}")
        model = YOLO(model_name)

    # 5. Execute Training
    print("\n" + "-" * 68)
    print("STARTING TRAINING (Transfer Learning on EdgeVision Dataset)...")
    print("-" * 68)

    try:
        train_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "patience": patience,
            "device": device,
            "seed": seed,
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": False if not args.resume else True,
            "save": True,
            "val": True,
            "workers": workers,
            "plots": True,
            "verbose": True
        }

        if args.resume and resume_ckpt:
            train_args["resume"] = True

        results = model.train(**train_args)

    except Exception as e:
        print(f"\n[Training Error] Encountered exception during training: {e}")
        # Check if batch size needs adjustment
        if "out of memory" in str(e).lower() and batch > 2:
            fallback_batch = max(2, batch // 2)
            print(f"\n[Recovery] Retrying with smaller batch size: {fallback_batch}...")
            train_args["batch"] = fallback_batch
            results = model.train(**train_args)
        else:
            raise e

    # 6. Locate Output Run Directory
    actual_run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else (project_dir / run_name)
    best_pt = actual_run_dir / "weights" / "best.pt"
    last_pt = actual_run_dir / "weights" / "last.pt"
    csv_path = actual_run_dir / "results.csv"

    # Copy best.pt to models/best.pt for application use
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        target_model = MODELS_DIR / "best.pt"
        shutil.copy2(best_pt, target_model)
        print(f"\n[Checkpoint] Copied best weights to: {target_model}")

    # 7. Generate Standalone Graphs
    print("\n[Plots] Generating metric curves and loss graphs...")
    custom_plots = generate_custom_metric_plots(csv_path, actual_run_dir)
    for plot_name, p_path in custom_plots.items():
        print(f"  - Generated {plot_name}: {p_path.name}")

    # 8. Overfitting / Underfitting Diagnosis
    print("\n[Analysis] Analyzing model generalization and fit dynamics...")
    fit_metrics = analyze_model_fit(csv_path)

    # 9. Save Reports
    run_report = actual_run_dir / "TRAINING_REPORT.md"
    root_report = PROJECT_ROOT / "TRAINING_REPORT.md"
    save_training_report(actual_run_dir, cfg, fit_metrics, run_report)
    shutil.copy2(run_report, root_report)

    # 10. Display Final Results in Console
    print("\n" + "=" * 68)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 68)
    print(f"1. Best Model Checkpoint:  {best_pt}")
    print(f"2. Final Model Checkpoint: {last_pt}")
    print(f"3. Completed Epochs:       {fit_metrics.get('epochs_completed', 0)}")
    print(f"4. Key Validation Metrics:")
    print(f"     - Precision:          {fit_metrics.get('final_precision', 0.0):.4f}")
    print(f"     - Recall:             {fit_metrics.get('final_recall', 0.0):.4f}")
    print(f"     - mAP @ 0.50:         {fit_metrics.get('final_map50', 0.0):.4f}")
    print(f"     - mAP @ 0.50:0.95:    {fit_metrics.get('final_map50_95', 0.0):.4f}")
    print(f"     - Final Val Box Loss: {fit_metrics.get('final_val_box_loss', 0.0):.4f}")
    print(f"     - Final Val Cls Loss: {fit_metrics.get('final_val_cls_loss', 0.0):.4f}")
    print(f"5. Generalization Status:  {fit_metrics.get('status')}")
    print(f"   Explanation:            {fit_metrics.get('explanation')}")
    print(f"6. Full Report Generated:  {root_report}")
    print("=" * 68)


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
