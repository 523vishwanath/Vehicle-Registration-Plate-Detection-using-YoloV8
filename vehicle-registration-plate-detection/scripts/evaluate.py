"""
evaluate.py
-----------
Runs COCO-style detection evaluation (mAP@50, mAP@50-95, Precision, Recall)
using the YOLOv8 validation pipeline and prints a readable summary.

Usage:
    python scripts/evaluate.py \
        --model runs/detect/yolov8m/weights/best.pt \
        --data vehicleRegistration.yaml
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="COCO detection evaluation with YOLOv8")
    p.add_argument("--model",   required=True,                   help="Path to best.pt")
    p.add_argument("--data",    default="vehicleRegistration.yaml")
    p.add_argument("--split",   default="val",                   help="Dataset split to evaluate")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="")
    p.add_argument("--plots",   action="store_true",             help="Show training loss / metric plots")
    return p.parse_args()


def print_metrics(metrics):
    """Pretty-print the COCO metrics returned by model.val()."""
    print("\n" + "=" * 50)
    print("  COCO Detection Evaluation Results")
    print("=" * 50)
    try:
        box = metrics.box
        print(f"  mAP@50         : {box.map50:.4f}")
        print(f"  mAP@50-95      : {box.map:.4f}")
        print(f"  Precision      : {box.mp:.4f}")
        print(f"  Recall         : {box.mr:.4f}")
    except AttributeError:
        print(f"  Raw metrics    : {metrics}")
    print("=" * 50 + "\n")


def show_training_plots(run_dir: str):
    results_img = os.path.join(run_dir, "results.png")
    if os.path.exists(results_img):
        img = plt.imread(results_img)
        plt.figure(figsize=(14, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Training Loss & Metrics", fontsize=13)
        plt.tight_layout()
        plt.show()
    else:
        print(f"[INFO] results.png not found at {results_img}")

    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        print("\nLast 5 training epochs:")
        print(df.tail(5).to_string(index=False))


def main():
    args = parse_args()

    print("=" * 60)
    print("Vehicle Registration Plate Detection – Evaluation")
    print("=" * 60)

    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        verbose=True,
        device=args.device if args.device else None,
    )

    print_metrics(metrics)

    if args.plots:
        # Derive training run dir from model path
        run_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.model))  # weights/../
        )
        show_training_plots(run_dir)


if __name__ == "__main__":
    main()
