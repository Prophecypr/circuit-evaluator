"""Generate detection data JSON for each benchmark image.
Run once to pre-compute YOLO detections so the annotation tool loads instantly.
"""
import os, json
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vision.unified_pipeline import _load_models

BENCHMARK = Path("benchmark")
OUTPUT = BENCHMARK / "detections"
OUTPUT.mkdir(exist_ok=True)

CGH_SKIP = {"junction", "crossover", "text", "probe.current", "probe.voltage", "explanatory"}
CGH_CONF_THRESH = 0.40
JUNCTION_CONF = 0.10
CGH_NAME_MAP = {
    "resistor": "Resistor", "resistor.adjustable": "Potentiometer",
    "capacitor.unpolarized": "Capacitor", "capacitor.polarized": "Polarized-Capacitor",
    "capacitor.adjustable": "Variable-Capacitor",
    "inductor": "Inductor", "inductor.ferrite": "Inductor", "diode": "Diode", "diode.light_emitting": "LED",
    "diode.zener": "Zener-Diode", "diode.thyrector": "Diac",
    "voltage.dc": "V-DC", "voltage.ac": "V-AC", "current.dc": "I-DC",
    "current.ac": "I-AC", "voltage.battery": "Battery",
    "gnd": "GND", "vss": "GND",
    "switch": "Switch", "speaker": "Speaker",
    "transistor.bjt": "BJT-PNP", "transistor.fet": "FET",
    "transistor.photo": "PhotoTransistor",
    "integrated_circuit": "IC", "integrated_circuit.ne555": "NE555",
    "integrated_circuit.voltage_regulator": "Voltage-Regulator",
    "op_amp": "Op-Amp", "schmitt_trigger": "Schmitt-Trigger",
    "optocoupler": "Optocoupler",
    "explanatory": "Explanatory", "terminal": "Terminal",
    "operational_amplifier": "Op-Amp", "op_amp": "Op-Amp",
    "schmitt_trigger": "Schmitt-Trigger", "lamp": "Lamp",
    "relay": "Relay", "thyristor": "Thyristor",
    "motor": "Motor",
}
PORT_POSITIONS = {
    "Resistor": [(0, 0.5), (1, 0.5)], "Potentiometer": [(0, 0.5), (1, 0.5), (0.5, 0.0)],
    "Capacitor": [(0, 0.5), (1, 0.5)], "Polarized-Capacitor": [(0, 0.5), (1, 0.5)],
    "Inductor": [(0, 0.5), (1, 0.5)], "Diode": [(0, 0.5), (1, 0.5)],
    "LED": [(0, 0.5), (1, 0.5)], "Zener-Diode": [(0, 0.5), (1, 0.5)],
    "V-DC": [(0.5, 0.0), (0.5, 1.0)], "V-AC": [(0.5, 0.0), (0.5, 1.0)],
    "I-DC": [(0.5, 0.0), (0.5, 1.0)], "I-AC": [(0.5, 0.0), (0.5, 1.0)],
    "Battery": [(0.5, 0.0), (0.5, 1.0)],
    "GND": [(0.5, 0.0)], "Switch": [(0, 0.5), (1, 0.5)],
    "Speaker": [(0, 0.5), (1, 0.5)], "BJT-PNP": [(0.3, 0.0), (0.7, 0.0), (0.5, 1.0)],
    "Diac": [(0, 0.5), (1, 0.5)], "Variable-Capacitor": [(0, 0.5), (1, 0.5)],
    "Terminal": [(0.5, 0.5)], "Explanatory": [],
    "Op-Amp": [(0, 0.15), (0, 0.85), (1, 0.5), (0.5, 0.0), (0.5, 1.0)],
    "Lamp": [(0, 0.5), (1, 0.5)],
    "PhotoTransistor": [(0.5, 0.0), (0.5, 1.0)],
    "FET": [(0, 0.5), (0.5, 1.0), (0.5, 0.0)],
    "Relay": [(0, 0.5), (1, 0.5), (0.5, 0.0), (0.5, 1.0)],
    "Thyristor": [(0, 0.5), (1, 0.5)],
    "Motor": [(0, 0.5), (1, 0.5)],
}
PORT_LABELS = {
    "Resistor": ["1", "2"], "Potentiometer": ["1", "2", "W"],
    "Capacitor": ["1", "2"], "Polarized-Capacitor": ["+", "-"],
    "Inductor": ["1", "2"], "Diode": ["+", "-"],
    "LED": ["+", "-"], "Zener-Diode": ["+", "-"],
    "V-DC": ["-", "+"], "V-AC": ["-", "+"],
    "I-DC": ["-", "+"], "I-AC": ["-", "+"],
    "Battery": ["-", "+"], "GND": ["GND"],
    "Switch": ["1", "2"], "Speaker": ["1", "2"],
    "BJT-PNP": ["B", "C", "E"], "Diac": ["1", "2"],
    "Variable-Capacitor": ["1", "2"], "Terminal": ["T"], "Explanatory": [],
    "Op-Amp": ["-", "+", "OUT", "V+", "V-"], "Lamp": ["1", "2"],
    "PhotoTransistor": ["C", "E"],
    "FET": ["G", "S", "D"],
    "Relay": ["C1", "C2", "NO", "COM"],
    "Thyristor": ["A", "K"],
    "Motor": ["1", "2"],
}

print("Loading models...")
cgh_model = _load_models()
import cv2, numpy as np

def detect_orientation(img_path, x1, y1, x2, y2, plist, raw_name=""):
    """Same logic as unified_pipeline._detect_orientation"""
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0: return False
    is_default_h = abs(plist[0][0] - plist[1][0]) > abs(plist[0][1] - plist[1][1])
    is_cap = "capacitor" in raw_name.lower() if raw_name else False
    is_led = "light_emitting" in raw_name.lower() if raw_name else False

    if (is_cap or is_led) and is_default_h:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            cx1, cy1 = max(0,x1), max(0,y1); cx2, cy2 = min(w,x2), min(h,y2)
            if cx2 > cx1+5 and cy2 > cy1+5:
                crop = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
                ve = float(np.sum(np.abs(gx))); he = float(np.sum(np.abs(gy)))
                if he+ve>0:
                    thr = 1.5 if is_led else 1.3
                    return he > ve * thr
        ratio = bh/max(bw,1)
        if is_led: return ratio < 0.8
        return ratio > 2.5
    elif is_cap:
        img = cv2.imread(img_path)
        if img is not None:
            h,w=img.shape[:2]; cx1,cy1=max(0,x1),max(0,y1); cx2,cy2=min(w,x2),min(h,y2)
            if cx2>cx1+5 and cy2>cy1+5:
                crop=cv2.cvtColor(img[cy1:cy2,cx1:cx2],cv2.COLOR_BGR2GRAY)
                ve=float(np.sum(np.abs(cv2.Sobel(crop,cv2.CV_64F,1,0,ksize=3))))
                he=float(np.sum(np.abs(cv2.Sobel(crop,cv2.CV_64F,0,1,ksize=3))))
                if he+ve>0: return ve > he * 1.3
        return bh/max(bw,1) < 0.4
    else:
        ratio = bh/max(bw,1)
        if is_default_h: return ratio > 1.3
        return ratio < 0.77

manifest = BENCHMARK / "manifest.txt"
with open(manifest) as f:
    images = [line.strip().split("\t")[0] for line in f if line.strip()]

for i, img_name in enumerate(images):
    img_path = str(BENCHMARK / img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"  SKIP {img_name}: cannot read")
        continue
    h, w = img.shape[:2]

    results = cgh_model(img_path)[0]
    components = []
    for box in (results.boxes or []):
        name = cgh_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if name in CGH_SKIP or conf < CGH_CONF_THRESH:
            continue
        hcd_name = CGH_NAME_MAP.get(name, name)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        ports = []
        plist = PORT_POSITIONS.get(hcd_name, [(0.5, 0.5)])
        no_rotate = {"GND", "gnd", "vss"}
        need_rotate = False
        if len(plist) == 2 and name not in no_rotate and hcd_name not in no_rotate:
            need_rotate = detect_orientation(img_path, x1, y1, x2, y2, plist, name)
        for rx, ry in plist:
            if need_rotate:
                sx, sy = 1 - ry, rx
            else:
                sx, sy = rx, ry
            px = int(x1 + sx * (x2 - x1))
            py = int(y1 + sy * (y2 - y1))
            ports.append([px, py])
        labels = PORT_LABELS.get(hcd_name, ["?"] * len(ports))
        # Swap LED labels if rotated
        if need_rotate and "light_emitting" in name:
            labels = list(reversed(labels))
        components.append({
            "idx": len(components),
            "name": hcd_name, "raw_name": name,
            "designator": hcd_name[:1] + str(len(components) + 1),
            "xyxy": [x1, y1, x2, y2],
            "ports": ports,
            "labels": labels,
            "conf": conf,
        })

    # NMS dedup: remove overlapping same-family detections
    NMS_FAMILIES = [
        {"diode", "diode.light_emitting", "diode.zener", "diode.thyrector"},
        {"capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable"},
        {"integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator"},
    ]
    specific_order = {"diode.light_emitting": 3, "diode.zener": 3, "diode.thyrector": 2, "diode": 1,
                      "capacitor.polarized": 3, "capacitor.adjustable": 2, "capacitor.unpolarized": 1,
                      "integrated_circuit.ne555": 3, "integrated_circuit.voltage_regulator": 3, "integrated_circuit": 1}
    nms_removed = set()
    for family in NMS_FAMILIES:
        for i in range(len(components)):
            if i in nms_removed: continue
            ci = components[i]
            if ci.get("raw_name", ci["name"]) not in family: continue
            for j in range(i + 1, len(components)):
                if j in nms_removed: continue
                cj = components[j]
                if cj.get("raw_name", cj["name"]) not in family: continue
                xa1, ya1, xa2, ya2 = ci["xyxy"]
                xb1, yb1, xb2, yb2 = cj["xyxy"]
                ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
                ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
                if ix2 <= ix1 or iy2 <= iy1: continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (xa2 - xa1) * (ya2 - ya1)
                area_b = (xb2 - xb1) * (yb2 - yb1)
                iou = inter / (area_a + area_b - inter)
                if iou > 0.35:
                    ri = ci.get("raw_name", ci["name"])
                    rj = cj.get("raw_name", cj["name"])
                    si = specific_order.get(ri, 0)
                    sj = specific_order.get(rj, 0)
                    if si >= sj: nms_removed.add(j)
                    else: nms_removed.add(i); break
    # Cross-family dedup: remove highly overlapping detections, keep more specific class
    for i in range(len(components)):
        if i in nms_removed: continue
        for j in range(i+1, len(components)):
            if j in nms_removed: continue
            xa1, ya1, xa2, ya2 = components[i]["xyxy"]
            xb1, yb1, xb2, yb2 = components[j]["xyxy"]
            ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
            ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
            if ix2 <= ix1 or iy2 <= iy1: continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = (xa2 - xa1) * (ya2 - ya1)
            area_b = (xb2 - xb1) * (yb2 - yb1)
            iou = inter / (area_a + area_b - inter) if (area_a+area_b-inter) > 0 else 0
            if iou > 0.7:
                conf_i = components[i].get("conf", 0)
                conf_j = components[j].get("conf", 0)
                # Keep higher confidence detection
                keep_i = conf_i >= conf_j
                nms_removed.add(j if keep_i else i)
                if not keep_i: break
                if not keep_i: break

    if nms_removed:
        components = [c for idx, c in enumerate(components) if idx not in nms_removed]
        for i, c in enumerate(components): c["idx"] = i
        print(f"    NMS: removed {len(nms_removed)} overlapping")

    data = {
        "image": img_name,
        "width": w, "height": h,
        "components": components,
    }
    json_path = OUTPUT / f"{Path(img_name).stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  [{i+1}/{len(images)}] {img_name}: {len(components)} components")

print(f"\nDone. Detection data in {OUTPUT}/")
