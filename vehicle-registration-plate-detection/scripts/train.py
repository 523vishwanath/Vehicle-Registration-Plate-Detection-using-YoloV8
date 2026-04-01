"""
train.py
--------
Fine-tunes YOLOv8m on the Vehicle Registration Plate dataset.

Usage:
    python scripts/train.py [--epochs 30] [--batch 8] [--imgsz 640]
"""

import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8m for licence-plate detection")
    p.add_argument("--model",   default="yolov8m.pt",              help="Base model weights")
    p.add_argument("--data",    default="vehicleRegistration.yaml", help="Dataset YAML")
    p.add_argument("--epochs",  type=int, default=30,               help="Number of training epochs")
    p.add_argument("--batch",   type=int, default=8,                help="Batch size")
    p.add_argument("--imgsz",   type=int, default=640,              help="Input image size")
    p.add_argument("--name",    default="yolov8m",                  help="Run name inside runs/detect/")
    p.add_argument("--device",  default="",                         help="Device: '' = auto, '0' = GPU 0, 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Vehicle Registration Plate Detection – Training")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Data   : {args.data}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch}")
    print(f"  Imgsz  : {args.imgsz}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        exist_ok=True,
        verbose=True,
        device=args.device if args.device else None,
    )

    print("\nTraining complete ✅")
    print(f"Best weights saved to: runs/detect/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
