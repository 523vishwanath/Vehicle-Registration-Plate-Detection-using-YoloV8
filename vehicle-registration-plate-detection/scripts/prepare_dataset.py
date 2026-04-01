"""
prepare_dataset.py
------------------
Builds the YOLO folder structure expected by vehicleRegistration.yaml.

Steps performed:
  1. Converts Open Images bounding-box labels → YOLO format (train + val).
  2. Copies images and converted labels into vehicleDataset/.

Expected input layout (produced by unzipping the downloaded archive):
    Dataset/
    ├── train/
    │   └── Vehicle registration plate/
    │       ├── *.jpg
    │       └── Label/
    │           └── *.txt
    └── validation/
        └── Vehicle registration plate/
            ├── *.jpg
            └── Label/
                └── *.txt

Output layout:
    vehicleDataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage:
    python scripts/prepare_dataset.py --dataset_root /path/to/Dataset
"""

import os
import shutil
import argparse
import cv2

CLASS_MAP = {"Vehicle registration plate": 0}
SPLITS = {
    "train": "train",
    "validation": "val",
}


def convert_and_copy(images_root, labels_root, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for filename in os.listdir(images_root):
        if not filename.lower().endswith(".jpg"):
            continue

        src_img = os.path.join(images_root, filename)
        label_path = os.path.join(labels_root, os.path.splitext(filename)[0] + ".txt")

        # Read image dimensions
        image = cv2.imread(src_img)
        if image is None:
            continue
        h, w, _ = image.shape

        # Convert labels
        yolo_lines = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    xmin, ymin, xmax, ymax = map(float, parts[-4:])
                    class_name = " ".join(parts[:-4])
                    class_id = CLASS_MAP.get(class_name, 0)
                    x_c = (xmin + xmax) / 2 / w
                    y_c = (ymin + ymax) / 2 / h
                    bw  = (xmax - xmin) / w
                    bh  = (ymax - ymin) / h
                    yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # Copy image
        shutil.copy(src_img, os.path.join(dst_img_dir, filename))

        # Write YOLO label
        out_lbl = os.path.join(dst_lbl_dir, os.path.splitext(filename)[0] + ".txt")
        with open(out_lbl, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    print(f"  Done → {dst_img_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="Dataset", help="Path to unzipped Dataset folder")
    parser.add_argument("--output_root",  default="vehicleDataset", help="Output YOLO dataset folder")
    args = parser.parse_args()

    for src_split, dst_split in SPLITS.items():
        print(f"\nProcessing split: {src_split} → {dst_split}")
        images_root = os.path.join(args.dataset_root, src_split, "Vehicle registration plate")
        labels_root = os.path.join(images_root, "Label")
        dst_img_dir = os.path.join(args.output_root, "images", dst_split)
        dst_lbl_dir = os.path.join(args.output_root, "labels", dst_split)
        convert_and_copy(images_root, labels_root, dst_img_dir, dst_lbl_dir)

    print("\nDataset preparation complete ✅")
    print(f"YOLO dataset ready at: {args.output_root}")


if __name__ == "__main__":
    main()
