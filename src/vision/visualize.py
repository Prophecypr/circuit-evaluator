"""Visualize detected components on circuit diagram with bounding boxes."""

import cv2
import numpy as np
from pathlib import Path

# Colors per class (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),      # resistor: red
    1: (0, 255, 255),    # led: yellow
    2: (255, 0, 0),      # voltage_source: blue
    3: (0, 255, 0),      # capacitor: green
    5: (128, 128, 128),  # ground: gray
}

CLASS_NAMES = ["resistor", "led", "voltage_source", "capacitor",
               "transistor", "ground", "diode", "inductor", "wire"]


def visualize_labels(image_path: str, label_path: str, output_path: str):
    """Draw bounding boxes from YOLO-format label file on the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    h, w = img.shape[:2]
    labels = Path(label_path).read_text().strip()

    if not labels:
        print(f"No labels in: {label_path}")
        cv2.imwrite(str(output_path), img)
        return

    for line in labels.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert normalized to pixel
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls_{cid}"
        cv2.putText(img, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(output_path), img)
    print(f"Visualized: {output_path}")


def visualize_dataset(data_dir: str = "data/images", max_per_type: int = 2):
    """Create visualization images for the first few samples."""
    import random
    for folder in ["train", "val"]:
        out_dir = Path(data_dir) / f"{folder}_viz"
        out_dir.mkdir(exist_ok=True)

        images = sorted(Path(data_dir).glob(f"{folder}/*.png"))
        if not images:
            continue

        for img_path in images[:max_per_type]:
            label_path = Path(str(img_path).replace(".png", ".txt"))
            if not label_path.exists():
                continue
            out_path = out_dir / img_path.name
            visualize_labels(str(img_path), str(label_path), str(out_path))


def create_comparison_image(preview_dir: str = "data/images"):
    """Create a side-by-side comparison of good vs bad circuits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = [
        ("preview_led_good.png", "preview_led_bad.png",
         "正确: 5V→220Ω→LED→GND", "错误: 5V→LED→GND (无电阻!)"),
    ]

    for good, bad, good_label, bad_label in pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for ax, fname, title in [(ax1, good, good_label),
                                  (ax2, bad, bad_label)]:
            img = cv2.imread(str(Path(preview_dir) / fname))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=12)
                ax.axis('off')

        out_path = str(Path(preview_dir) / "comparison.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"Comparison saved: {out_path}")


if __name__ == "__main__":
    # Visualize some labeled images
    visualize_dataset("data/images", max_per_type=3)

    # Create comparison
    create_comparison_image()
