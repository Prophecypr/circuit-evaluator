"""Convert CGHD PASCAL VOC XML annotations to YOLO format for training.

CGHD: 60 classes, 2837 annotated images, 29 drafters
Output: YOLO-format labels + dataset.yaml
"""

import os, xml.etree.ElementTree as ET, json, random
from pathlib import Path
import cv2

CGHD_ROOT = Path("E:/circuit_image/cghd-zenodo-13")
OUT_DIR = Path("E:/circuit_data/cghd_yolo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CGHD class → contiguous YOLO class_id
CGHD_NAMES = [
    "text", "junction", "crossover", "terminal", "gnd", "vss",
    "voltage.dc", "voltage.ac", "voltage.battery",
    "resistor", "resistor.adjustable", "resistor.photo",
    "capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable",
    "inductor", "inductor.ferrite", "inductor.coupled", "transformer",
    "diode", "diode.light_emitting", "diode.thyrector", "diode.zener",
    "diac", "triac", "thyristor", "varistor",
    "transistor.bjt", "transistor.fet", "transistor.photo",
    "operational_amplifier", "operational_amplifier.schmitt_trigger", "optocoupler",
    "integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator",
    "xor", "and", "or", "not", "nand", "nor",
    "probe", "probe.current", "probe.voltage",
    "switch", "relay", "socket", "fuse",
    "speaker", "motor", "lamp", "microphone", "antenna", "crystal",
    "mechanical",
]
CLASS_MAP = {name: i for i, name in enumerate(CGHD_NAMES)}
NUM_CLASSES = len(CGHD_NAMES)  # 57 classes
CLASS_NAMES = CGHD_NAMES


def convert_annotation(xml_path, img_w, img_h):
    """Convert one PASCAL VOC XML to YOLO format labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        cid = CLASS_MAP[name]
        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        # YOLO format: cx cy w h (normalized)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return labels


def main():
    # Collect all image-annotation pairs
    pairs = []
    for drafter in sorted(CGHD_ROOT.iterdir()):
        if not drafter.is_dir() or not drafter.name.startswith("drafter_"):
            continue
        ann_dir = drafter / "annotations"
        img_dir = drafter / "images"
        if not ann_dir.is_dir() or not img_dir.is_dir():
            continue
        for ann_file in ann_dir.iterdir():
            if not ann_file.suffix.lower() == ".xml":
                continue
            # Find matching image
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                img_file = img_dir / (base + ext)
                if img_file.exists():
                    pairs.append((img_file, ann_file))
                    break

    print(f"Found {len(pairs)} annotated image-annotation pairs")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.85)
    train = pairs[:split]
    val = pairs[split:]

    total_labels = 0
    for subset, pairs_sub in [("train", train), ("val", val)]:
        img_out = OUT_DIR / "images" / subset
        lbl_out = OUT_DIR / "labels" / subset
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, ann_path in pairs_sub:
            # Copy image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            out_name = f"{img_path.parent.name}_{img_path.stem}"

            # Convert to jpg for consistency
            cv2.imwrite(str(img_out / f"{out_name}.jpg"), img)

            # Convert labels
            labels = convert_annotation(str(ann_path), w, h)
            with open(lbl_out / f"{out_name}.txt", "w") as f:
                f.write("\n".join(labels))
            total_labels += len(labels)

    # Write YOLO dataset config
    yaml = f"""path: {OUT_DIR.as_posix()}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml += f"  {i}: {name}\n"

    config_path = OUT_DIR / "cghd_dataset.yaml"
    config_path.write_text(yaml)

    print(f"\nTrain: {len(train)} images, Val: {len(val)} images")
    print(f"Total labels: {total_labels}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Config: {config_path}")
    print(f"\nNext: yolo train data={config_path} model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    main()
