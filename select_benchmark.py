"""Select 70 images from CGHD for benchmark annotation.
Stratified by circuit complexity (component count).
"""
import os, random, xml.etree.ElementTree as ET
from collections import defaultdict
import shutil
from pathlib import Path

CGHD_BASE = r"E:\circuit_image\cghd-zenodo-16"
OUTPUT = Path("benchmark")
OUTPUT.mkdir(exist_ok=True)

# Collect all images with component counts
images = []
drafters = [d for d in os.listdir(CGHD_BASE) if d.startswith("drafter_")]

for dr in drafters:
    ann_dir = os.path.join(CGHD_BASE, dr, "annotations")
    img_dir = os.path.join(CGHD_BASE, dr, "images")
    if not os.path.isdir(ann_dir):
        continue
    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()
        comp_count = 0
        term_count = 0
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "terminal":
                term_count += 1
            if name not in ("junction", "crossover", "text",
                           "probe.current", "probe.voltage"):
                comp_count += 1
        img_name = root.findtext("filename", xml_file.replace(".xml", ".jpg"))
        img_path = os.path.join(img_dir, img_name)
        if os.path.isfile(img_path):
            images.append({
                "path": img_path,
                "xml": os.path.join(ann_dir, xml_file),
                "drafter": dr,
                "count": comp_count,
                "name": img_name,
                "terminals": term_count,
            })

print(f"Total valid images: {len(images)}")
term_images = [img for img in images if img["terminals"] > 0]
non_term = [img for img in images if img["terminals"] == 0]
print(f"With terminals: {len(term_images)}, Without: {len(non_term)}")

random.seed(42)
selected = []
used_drafters = defaultdict(int)

# Phase 1: pick 10 terminal images across drafters
term_by_drafter = defaultdict(list)
for img in term_images:
    term_by_drafter[img["drafter"]].append(img)
# Pick 1 from each of 10 different drafters with most terminals
sorted_drafters = sorted(term_by_drafter.keys(), key=lambda d: len(term_by_drafter[d]), reverse=True)
for dr in sorted_drafters[:10]:
    pool = term_by_drafter[dr]
    # Pick one with median component count
    pool.sort(key=lambda x: x["count"])
    pick = pool[len(pool)//2]
    selected.append(pick)
    used_drafters[pick["drafter"]] += 1
print(f"  Terminal images: {len(selected)}")

# Phase 2: pick 70 non-terminal images stratified by component count
bins = [
    ("0-4", 0, 4, 12),
    ("5-9", 5, 9, 22),
    ("10-19", 10, 19, 22),
    ("20+", 20, 999, 14),
]

for label, lo, hi, target in bins:
    pool = [img for img in images if lo <= img["count"] <= hi and img not in selected]
    # Prefer drafters not yet used
    pool.sort(key=lambda x: used_drafters[x["drafter"]])
    picked = pool[:target]
    for img in picked:
        used_drafters[img["drafter"]] += 1
    selected.extend(picked)
    print(f"  {label} components: {len(picked)} selected (target {target})")

print(f"\nTotal selected: {len(selected)}")
print(f"Drafters covered: {len(used_drafters)}")

# Copy images to benchmark folder and save manifest
with open(OUTPUT / "manifest.txt", "w") as f:
    for i, img in enumerate(selected):
        dst = OUTPUT / f"img_{i:03d}.jpg"
        shutil.copy(img["path"], dst)
        f.write(f"{dst.name}\t{img['drafter']}\t{img['count']}\t{img['name']}\n")

print(f"\nImages copied to {OUTPUT}/")
print(f"Manifest: {OUTPUT}/manifest.txt")
