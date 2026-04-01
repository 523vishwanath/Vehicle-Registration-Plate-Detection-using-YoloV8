"""
inference.py
------------
Runs YOLOv8 inference on images OR a video and saves the results.

Usage – images:
    python scripts/inference.py \
        --model runs/detect/yolov8m/weights/best.pt \
        --source vehicleDataset/images/val \
        --conf 0.25

Usage – video:
    python scripts/inference.py \
        --model runs/detect/yolov8m/weights/best.pt \
        --source path/to/video.mp4 \
        --conf 0.15 \
        --name yolov8m_predict_video
"""

import os
import glob
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference – images or video")
    p.add_argument("--model",  required=True,                   help="Path to best.pt")
    p.add_argument("--source", required=True,                   help="Image folder or video file")
    p.add_argument("--name",   default="yolov8m_predict",       help="Run name under runs/detect/")
    p.add_argument("--imgsz",  type=int, default=640,           help="Inference image size")
    p.add_argument("--conf",   type=float, default=0.25,        help="Confidence threshold")
    p.add_argument("--device", default="",                      help="Device: '' = auto, 'cpu', '0'")
    p.add_argument("--show_results", action="store_true",       help="Display sample result images")
    return p.parse_args()


def display_results(result_dir: str, max_images: int = 3):
    """Display up to *max_images* sample predictions as a matplotlib grid."""
    image_paths = glob.glob(os.path.join(result_dir, "*.jpg"))[:max_images]
    if not image_paths:
        print(f"No result images found in: {result_dir}")
        return

    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, img_path in zip(axes, image_paths):
        image = plt.imread(img_path)
        ax.imshow(image)
        ax.set_title(os.path.basename(img_path), fontsize=9)
        ax.axis("off")

    plt.suptitle("YOLOv8 Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    print("=" * 60)
    print("Vehicle Registration Plate Detection – Inference")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Source : {args.source}")
    print(f"  Conf   : {args.conf}")
    print()

    model = YOLO(args.model)

    model.predict(
        source=args.source,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        exist_ok=True,
        verbose=True,
        device=args.device if args.device else None,
    )

    result_dir = os.path.join("runs", "detect", args.name)
    print(f"\nInference complete ✅")
    print(f"Results saved to: {result_dir}")

    if args.show_results:
        display_results(result_dir)


if __name__ == "__main__":
    main()
