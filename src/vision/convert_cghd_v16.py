"""Convert CGHD v16 PASCAL VOC XML annotations to YOLO format.

CGHD v16: 61 classes, 3269 annotations, 32 drafters (0-31 + -1)
Output: YOLO-format labels + dataset.yaml
"""

import os, xml.etree.ElementTree as ET, random
from pathlib import Path
import cv2

CGHD_ROOT = Path("E:/circuit_image/cghd-zenodo-16")
OUT_DIR = Path("E:/circuit_data/cghd_yolo_v16")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CGHD v16 class list (from classes.json, skipping __background__ at index 0)
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
    "magnetic", "optical", "block", "explanatory", "unknown",
]
CLASS_MAP = {name: i for i, name in enumerate(CGHD_NAMES)}  # YOLO IDs: 0-60
NUM_CLASSES = len(CGHD_NAMES)


def convert_annotation(xml_path, img_w, img_h):
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
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return labels


def main():
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
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                img_file = img_dir / (base + ext)
                if img_file.exists():
                    pairs.append((img_file, ann_file))
                    break

    print(f"Found {len(pairs)} annotated image-annotation pairs")

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
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            out_name = f"{img_path.parent.name}_{img_path.stem}"
            cv2.imwrite(str(img_out / f"{out_name}.jpg"), img)

            labels = convert_annotation(str(ann_path), w, h)
            with open(lbl_out / f"{out_name}.txt", "w") as f:
                f.write("\n".join(labels))
            total_labels += len(labels)

    # Write YOLO dataset config
    yaml = f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnc: {NUM_CLASSES}\nnames:\n"
    for i, name in enumerate(CGHD_NAMES):
        yaml += f"  {i}: {name}\n"

    config_path = OUT_DIR / "cghd_dataset.yaml"
    config_path.write_text(yaml)

    print(f"\nTrain: {len(train)} images, Val: {len(val)} images")
    print(f"Total labels: {total_labels}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Config: {config_path}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
