"""Extract text crops from CGHD dataset for CRNN OCR training.

CGHD XML annotations contain <object name="text"> with <text> labels
(ground-truth text content). This script:
1. Parses all CGHD XML files across all drafters
2. Filters text labels to the CRNN character set
3. Crops text regions from original images
4. Generates train_labels.txt / val_labels.txt in data/cghd_text/

Usage:
    python -m src.vision.extract_cghd_text
"""

import os, sys, random
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CGHD_BASE = r"E:\circuit_image\cghd-zenodo-16"
OUTPUT_DIR = Path("data/cghd_text")
VAL_SPLIT = 0.10  # 10% for validation

# CRNN character set (must match train_ocr.py CHARS)
CHARS = "0123456789.kKmMΩμunpFV AHz-+/"
VALID_CHARS = set(CHARS)


def is_valid_label(text: str) -> bool:
    """Check if text only contains characters the CRNN can recognize."""
    if not text or len(text) < 1:
        return False
    for c in text:
        if c not in VALID_CHARS:
            return False
    return True


def extract_cghd_texts():
    """Extract all valid text crops from CGHD dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all drafters
    drafters = sorted(
        d for d in os.listdir(CGHD_BASE)
        if d.startswith("drafter_") and os.path.isdir(os.path.join(CGHD_BASE, d))
    )
    print(f"Found {len(drafters)} drafters")

    samples = []  # [(crop_path, text_label), ...]
    skipped_chars = 0
    skipped_empty = 0
    total_texts = 0
    crop_idx = 0

    for dr in drafters:
        ann_dir = os.path.join(CGHD_BASE, dr, "annotations")
        img_dir = os.path.join(CGHD_BASE, dr, "images")
        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            continue

        for xml_file in sorted(os.listdir(ann_dir)):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(ann_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                print(f"  WARNING: parse error in {xml_path}")
                continue

            root = tree.getroot()
            img_filename = root.findtext("filename", xml_file.replace(".xml", ".jpg"))
            img_path = os.path.join(img_dir, img_filename)
            if not os.path.isfile(img_path):
                img_filename = img_filename.replace(".jpg", ".png")
                img_path = os.path.join(img_dir, img_filename)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is None or name_el.text != "text":
                    continue

                text_el = obj.find("text")
                if text_el is None or not text_el.text:
                    skipped_empty += 1
                    total_texts += 1
                    continue

                text = text_el.text.strip()
                total_texts += 1

                if not is_valid_label(text):
                    skipped_chars += 1
                    continue

                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                try:
                    x1 = int(float(bbox.findtext("xmin", "0")))
                    y1 = int(float(bbox.findtext("ymin", "0")))
                    x2 = int(float(bbox.findtext("xmax", "0")))
                    y2 = int(float(bbox.findtext("ymax", "0")))
                except (ValueError, TypeError):
                    continue

                # Clamp to image bounds
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 2, min(x2, img_w))
                y2 = max(y1 + 2, min(y2, img_h))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Convert to grayscale and save
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # Apply light adaptive threshold to clean up hand-drawn text
                # (invert: white bg, black text -> keep as-is for CRNN)
                crop_path = OUTPUT_DIR / f"cghd_{crop_idx:06d}.png"
                cv2.imwrite(str(crop_path), crop_gray)
                samples.append((str(crop_path), text))
                crop_idx += 1

    print(f"\nTotal CGHD text objects: {total_texts}")
    print(f"  Valid crops saved: {len(samples)}")
    print(f"  Skipped (empty text): {skipped_empty}")
    print(f"  Skipped (invalid chars): {skipped_chars}")

    if not samples:
        print("ERROR: No valid samples extracted!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * (1 - VAL_SPLIT))
    train = samples[:split]
    val = samples[split:]

    # Write label files
    for name, subset in [("train_labels.txt", train), ("val_labels.txt", val)]:
        label_path = OUTPUT_DIR / name
        with open(label_path, "w", encoding="utf-8") as f:
            for path, text in subset:
                f.write(f"{path}\t{text}\n")
        print(f"\n{name}: {len(subset)} samples -> {label_path}")

    # Show character coverage
    all_text = "".join(t for _, t in samples)
    covered = sorted(set(all_text))
    missing = [c for c in CHARS if c not in covered]
    print(f"\nCharacter coverage: {len(covered)}/{len(CHARS)}")
    print(f"  Covered: {''.join(covered)}")
    if missing:
        print(f"  Missing from training data: {''.join(missing)}")
        print(f"  (CRNN will still have these as output classes via CTC blank)")

    # Show sample texts
    print(f"\nSample labels: {[t for _, t in samples[:15]]}")

    print(f"\nDone! Update train_ocr.py DATA_DIR = Path('{OUTPUT_DIR}')")
    print(f"Then run: python -m src.vision.train_ocr --train")


if __name__ == "__main__":
    extract_cghd_texts()
