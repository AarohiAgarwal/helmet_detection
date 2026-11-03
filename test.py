"""
YOLOv8 Helmet Detection Test & Benchmarking Script.
Evaluates model performance on the test split, calculates Precision, Recall, mAP50,
and mAP50-95, and exports class-wise performance metrics.
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_YAML_PATH, PROCESSED_DATASET_ROOT
from evaluate_model import (
    locate_trained_checkpoint,
    run_test_evaluation,
    analyze_errors_and_samples,
    write_evaluation_report,
    EVAL_REPORT_MD_PATH,
    EVAL_SAMPLES_DIR
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test and benchmark YOLOv8 Helmet Detection Model")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights (default: auto-locate best.pt)")
    parser.add_argument("--data", type=str, default=str(DATA_YAML_PATH), help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, 0, auto)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("YOLOv8 HELMET DETECTION TEST EVALUATION")
    print("=" * 65)

    # Automatically locate trained checkpoint without selecting unrelated models
    checkpoint = locate_trained_checkpoint(args.weights)
    if not checkpoint:
        print("\n[Notice] No trained helmet detection checkpoint ('best.pt') found.")
        print("Safety Rule Active: Unrelated models will not be silently selected,")
        print("and performance values will not be fabricated.")
        print("\nPlease run model training first to generate 'best.pt':")
        print("    python train.py")
        sys.exit(1)

    print(f"Using Checkpoint: {checkpoint}")
    print(f"Dataset YAML:     {args.data}")
    print(f"Evaluating Split: {args.split}")

    # Run evaluation
    results = run_test_evaluation(
        model_path=checkpoint,
        data_yaml=Path(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device
    )

    # Analyze errors on test images
    test_img_dir = PROCESSED_DATASET_ROOT / "images" / args.split
    test_lbl_dir = PROCESSED_DATASET_ROOT / "labels" / args.split
    analyze_errors_and_samples(
        model_path=checkpoint,
        test_img_dir=test_img_dir,
        test_lbl_dir=test_lbl_dir,
        output_samples_dir=EVAL_SAMPLES_DIR
    )

    # Write report
    write_evaluation_report(results, EVAL_REPORT_MD_PATH)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"mAP@50:      {results['map50']:.4f}")
    print(f"mAP@50-95:   {results['map50_95']:.4f}")
    print(f"Report:      {EVAL_REPORT_MD_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
