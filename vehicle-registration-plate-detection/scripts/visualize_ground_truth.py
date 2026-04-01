"""
visualize_ground_truth.py
--------------------------
Plots ground-truth bounding boxes (YOLO format) on random images and
saves a summary grid to disk.

Usage:
    python scripts/visualize_ground_truth.py \
        --images_dir vehicleDataset/images/val \
        --labels_dir vehicleDataset/labels/val \
        --num_samples 4 \
        --output assets/ground_truth_preview.jpg
"""

import os
import glob
import random
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Class config ────────────────────────────────────────────────────────────
CLASSES = ["RegPlate"]
COLORS  = [(20, 246, 100)]          # BGR; one per class


# ── Helpers ──────────────────────────────────────────────────────────────────
def yolo2bbox(xc, yc, w, h, img_w, img_h):
    """Convert normalised YOLO box → pixel (xmin, ymin, xmax, ymax)."""
    xmin = int((xc - w / 2) * img_w)
    ymin = int((yc - h / 2) * img_h)
    xmax = int((xc + w / 2) * img_w)
    ymax = int((yc + h / 2) * img_h)
    return xmin, ymin, xmax, ymax


def draw_boxes(image, label_path):
    """Draw all ground-truth boxes on *image* (BGR NumPy array)."""
    h, w = image.shape[:2]
    thickness = max(2, int(w / 275))

    if not os.path.exists(label_path):
        return image

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
            xmin, ymin, xmax, ymax = yolo2bbox(xc, yc, bw, bh, w, h)

            color = COLORS[cls_id % len(COLORS)][::-1]   # BGR → RGB for cv2
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            cv2.putText(
                image, CLASSES[cls_id],
                (xmin, max(0, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
            )
    return image


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualise ground-truth bounding boxes")
    parser.add_argument("--images_dir",  default="vehicleDataset/images/val")
    parser.add_argument("--labels_dir",  default="vehicleDataset/labels/val")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output",      default="assets/ground_truth_preview.jpg")
    args = parser.parse_args()

    all_images = glob.glob(os.path.join(args.images_dir, "*.jpg")) + \
                 glob.glob(os.path.join(args.images_dir, "*.JPG"))

    if not all_images:
        print(f"No images found in: {args.images_dir}")
        return

    samples = random.sample(all_images, min(args.num_samples, len(all_images)))
    cols = 2
    rows = (len(samples) + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 7 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, img_path in zip(axes, samples):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.labels_dir, stem + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = draw_boxes(image, label_path)

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(stem, fontsize=9)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.suptitle("Ground-Truth Bounding Boxes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved preview → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
