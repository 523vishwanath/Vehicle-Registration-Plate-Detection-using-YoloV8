"""
convert_labels.py
-----------------
Converts Open Images-style bounding box labels (xmin ymin xmax ymax)
into YOLO normalised format (x_center y_center width height).

Usage:
    python scripts/convert_labels.py \
        --images_root Dataset/train/Vehicle\ registration\ plate \
        --labels_root Dataset/train/Vehicle\ registration\ plate/Label \
        --output_dir Dataset/train/labels_yolo

    # For validation:
    python scripts/convert_labels.py \
        --images_root Dataset/validation/Vehicle\ registration\ plate \
        --labels_root Dataset/validation/Vehicle\ registration\ plate/Label \
        --output_dir Dataset/validation/labels_yolo
"""

import os
import cv2
import argparse

CLASS_MAP = {
    "Vehicle registration plate": 0
}


def convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_split(images_root: str, labels_root: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    converted = skipped = 0

    for filename in os.listdir(images_root):
        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(images_root, filename)
        label_path = os.path.join(
            labels_root,
            os.path.splitext(filename)[0] + ".txt"
        )

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [WARN] Could not read image: {image_path}")
            skipped += 1
            continue

        h, w, _ = image.shape
        yolo_lines = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    xmin = float(parts[-4])
                    ymin = float(parts[-3])
                    xmax = float(parts[-2])
                    ymax = float(parts[-1])

                    class_name = " ".join(parts[:-4])
                    class_id   = CLASS_MAP.get(class_name, 0)

                    x_c, y_c, bw, bh = convert_box_to_yolo(xmin, ymin, xmax, ymax, w, h)
                    yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        converted += 1

    print(f"  Converted: {converted}  |  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert Open Images labels to YOLO format")
    parser.add_argument("--images_root", required=True, help="Path to folder with .jpg images")
    parser.add_argument("--labels_root", required=True, help="Path to folder with .txt label files")
    parser.add_argument("--output_dir",  required=True, help="Destination folder for YOLO labels")
    args = parser.parse_args()

    print(f"Converting labels from: {args.labels_root}")
    convert_split(args.images_root, args.labels_root, args.output_dir)
    print("Conversion Complete ✅")


if __name__ == "__main__":
    main()
