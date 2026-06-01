"""图像 → YOLO元件检测 + OCR数值读取 → 文字描述 → LLM评价

用法:
    python -m src.vision.image_eval test1.jpg
    python -m src.vision.image_eval "E:/circuit_image/.../circuit_100.jpg"
"""

import sys, os, cv2, json, math, re
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from src.llm import ask

import torch
from src.vision.train_ocr import load_trained_model, predict as crnn_predict

# Try to load CRNN model; fall back to Tesseract if not available
CRNN_MODEL = None
CRNN_CHARS = ""
CRNN_ITOC = {}
CRNN_IMG_H = 32
try:
    CRNN_MODEL, CRNN_CHARS, _, CRNN_ITOC, CRNN_IMG_H = load_trained_model()
    print(f"OCR: CRNN loaded ({len(CRNN_CHARS)} chars)")
except Exception as e:
    print(f"OCR: CRNN not available ({e}), using Tesseract fallback")

# Semantic value filters per component type
VALUE_PATTERNS = {
    "Resistor": (r"^[\d.]+[kKM]?\s*[Ω]?$", "Ω"),
    "Capacitor": (r"^[\d.]+[μunpUNP]?\s*F?$", "F"),
    "Inductor": (r"^[\d.]+[μumM]?\s*H?$", "H"),
    "V-DC": (r"^[\d.]+[μumM]?\s*V?$", "V"),
    "V-AC": (r"^[\d.]+[μumM]?\s*V?$", "V"),
    "I-DC": (r"^[\d.]+[μumM]?\s*A?$", "A"),
    "I-AC": (r"^[\d.]+[μumM]?\s*A?$", "A"),
}

# Anti-patterns: values that should NEVER match certain component types
ANTI_PATTERNS = {
    "Diode":       [r"[AVΩ]$", r"[Fμ]$", r"^[\d.]+\s*[FHz]$"],
    "Zener Diode": [r"[AVΩ]$", r"[Fμ]$", r"^[\d.]+\s*[FHz]$"],
    "GND":         [r".*"],
    "Wire Crossover": [r".*"],
    "MOSFET-N":    [r"[ΩFVHzμA]$"],
    "MOSFET-P":    [r"[ΩFVHzμA]$"],
    "BJT-NPN":     [r"[ΩFVHzμA]$"],
    "BJT-PNP":     [r"[ΩFVHzμA]$"],
    "Op-Amp":      [r"[ΩFVHzμA]$"],
    # Power semiconductors: reject all passive/current values
    "Thyristor":   [r"[ΩFVHzμA]$"],
    "Triac":       [r"[ΩFVHzμA]$"],
    "Diac":        [r"[ΩFVHzμA]$"],
    "Varistor":    [r"[ΩFVHzμA]$"],
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "runs", "detect")

import numpy as np

def _detect_orientation(img_path, x1, y1, x2, y2, plist):
    """Sobel edge analysis on component crop to determine if ports need rotation."""
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return False
    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return False
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
    v_edges = np.abs(sobel_x).sum()  # vertical edges (|| capacitor plates)
    h_edges = np.abs(sobel_y).sum()  # horizontal edges (-- plates)

    is_default_h = abs(plist[0][0] - plist[1][0]) > abs(plist[0][1] - plist[1][1])
    if is_default_h:
        return h_edges > v_edges * 1.3
    else:
        return v_edges > h_edges * 1.3
COMP_MODEL = None
TEXT_MODEL = None
CGH_MODEL = None       # CGHD 56-class model (optional)
USE_CGH = False


def _load_models():
    global COMP_MODEL, TEXT_MODEL, CGH_MODEL, USE_CGH
    if COMP_MODEL is None:
        # Try CGHD model first (56 classes, better granularity)
        cghd_path = os.path.join(MODELS_DIR, "cghd_56cls", "weights", "best.pt")
        if os.path.isfile(cghd_path):
            CGH_MODEL = YOLO(cghd_path)
            COMP_MODEL = CGH_MODEL  # use CGHD as primary
            USE_CGH = True
            print("YOLO: CGHD 56-class model loaded")
        else:
            COMP_MODEL = YOLO(os.path.join(MODELS_DIR, "circuit_real", "weights", "best.pt"))
            print("YOLO: HCD 17-class model loaded (CGHD not found)")

        TEXT_MODEL = YOLO(os.path.join(MODELS_DIR, "circuit_text", "weights", "best.pt"))
    return COMP_MODEL, TEXT_MODEL


# CGHD → HCD name mapping (for value matching compatibility)
# CGHD → HCD name mapping (for value matching compatibility)
CGH_NAME_MAP = {
    "resistor": "Resistor", "capacitor.unpolarized": "Capacitor",
    "capacitor.polarized": "Capacitor", "capacitor.adjustable": "Capacitor",
    "inductor": "Inductor", "inductor.ferrite": "Inductor",
    "diode": "Diode", "diode.light_emitting": "Diode", "diode.zener": "Zener Diode",
    "diode.thyrector": "Diode",
    "voltage.dc": "V-DC", "voltage.ac": "V-AC", "voltage.battery": "V-DC",
    "gnd": "GND", "vss": "GND",
    "transistor.bjt": "BJT-PNP", "transistor.fet": "MOSFET-P",
    "operational_amplifier": "Op-Amp",
    # Power semiconductors: treat as semiconductors (no passive values)
    "thyristor": "Thyristor", "triac": "Triac", "diac": "Diac",
    "varistor": "Varistor",
}
# CGHD display names (shorter)
CGH_DISPLAY = {
    "resistor": "R", "capacitor.unpolarized": "C", "capacitor.polarized": "Cp",
    "inductor": "L", "diode": "D", "diode.light_emitting": "LED",
    "diode.zener": "ZD", "voltage.dc": "Vdc", "voltage.ac": "Vac",
    "gnd": "GND", "transistor.bjt": "BJT", "transistor.fet": "FET",
    "operational_amplifier": "OpAmp",
    "thyristor": "SCR", "triac": "TRIAC", "diac": "DIAC", "varistor": "VDR",
}
# CGHD classes to skip (not electrical components)
CGH_SKIP = {"text", "junction", "crossover", "terminal"}
CGH_CONF_THRESH = 0.45  # minimum confidence for CGHD detections

# Port positions from CGHD classes_ports.json (ground truth, 0-1 normalized within bbox)
# Default orientation is HORIZONTAL (left-right ports). Rotated if bbox is taller than wide.
PORT_POSITIONS = {
    # 2-pin horizontal default
    "Resistor":       [(0,0.5), (1,0.5)],
    "Capacitor":      [(0,0.5), (1,0.5)],
    "Inductor":       [(0,0.5), (1,0.5)],
    "Diode":          [(0,0.5), (1,0.5)],
    "Zener Diode":    [(0,0.5), (1,0.5)],
    "Thyristor":      [(0,0.5), (1,0.5)],
    "Triac":          [(0,0.5), (1,0.5)],
    "Diac":           [(0,0.5), (1,0.5)],
    "Varistor":       [(0,0.5), (1,0.5)],
    # Vertical default (top-bottom)
    "V-DC":           [(0.5,1.0), (0.5,0.0)],
    "V-AC":           [(0.5,1.0), (0.5,0.0)],
    "I-DC":           [(0.5,1.0), (0.5,0.0)],
    "I-AC":           [(0.5,1.0), (0.5,0.0)],
    # 1-pin
    "GND":            [(0.5,0.0)],
    # 3-pin (from classes_ports.json)
    "BJT-NPN":        [(0,0.5), (0.7,1.0), (0.7,0.0)],
    "BJT-PNP":        [(0,0.5), (0.7,1.0), (0.7,0.0)],
    "MOSFET-N":       [(0,0.5), (0.5,1.0), (0.5,0.0)],
    "MOSFET-P":       [(0,0.5), (0.5,1.0), (0.5,0.0)],
    "Op-Amp":         [(0,0.5), (0,0.3), (1,0.5)],
    "Wire Crossover": [],
}
# CGHD raw names → port key
CGH_TO_PORT_KEY = {
    "resistor": "Resistor", "resistor.adjustable": "Resistor",
    "capacitor.unpolarized": "Capacitor", "capacitor.polarized": "Capacitor",
    "inductor": "Inductor", "inductor.ferrite": "Inductor",
    "diode": "Diode", "diode.light_emitting": "Diode",
    "diode.zener": "Zener Diode", "diode.thyrector": "Diode",
    "voltage.dc": "V-DC", "voltage.ac": "V-AC",
    "gnd": "GND", "vss": "GND",
    "transistor.bjt": "BJT-PNP", "transistor.fet": "MOSFET-P",
    "operational_amplifier": "Op-Amp",
    "thyristor": "Thyristor", "triac": "Triac", "diac": "Diac", "varistor": "Varistor",
}


def detect_components(img_path: str) -> list[dict]:
    """YOLO component detection. Uses CGHD 56-class if available, else HCD 17-class."""
    comp_model, _ = _load_models()
    results = comp_model(img_path)[0]
    components = []
    for box in (results.boxes or []):
        name = comp_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        if USE_CGH:
            # CGHD model: filter noise + remap names
            if name in CGH_SKIP or conf < CGH_CONF_THRESH:
                continue
            # Map to HCD-compatible name for value matching
            hcd_name = CGH_NAME_MAP.get(name, name)
            display = CGH_DISPLAY.get(name, name)
        else:
            hcd_name = name
            display = name
            if name == "Wire Crossover":
                continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bw, bh = x2-x1, y2-y1

        # Port positions: Sobel edge analysis determines orientation
        ports = []
        if USE_CGH:
            port_key = CGH_TO_PORT_KEY.get(name, name)
        else:
            port_key = hcd_name
        if port_key in PORT_POSITIONS and len(PORT_POSITIONS[port_key]) >= 1:
            plist = PORT_POSITIONS[port_key]
            if len(plist) == 2:
                need_rotate = _detect_orientation(img_path, x1, y1, x2, y2, plist)
            else:
                need_rotate = False
            for rx, ry in plist:
                if need_rotate:
                    sx, sy = 1-ry, rx
                else:
                    sx, sy = rx, ry
                px = int(x1 + sx*bw)
                py = int(y1 + sy*bh)
                ports.append((px, py))

        components.append(dict(
            name=hcd_name, display=display, raw_name=name,
            xyxy=(x1,y1,x2,y2),
            cx=(x1+x2)//2, cy=(y1+y2)//2,
            conf=conf, value="", ports=ports,
        ))
    return components


def read_text_values(img_path: str) -> list[dict]:
    """YOLO text region detection + CRNN/Tesseract OCR. Returns list of {text, cx, cy}."""
    _, text_model = _load_models()
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # First pass: detect components to filter false text inside component bboxes
    comp_model, _ = _load_models()
    r_comp = comp_model(img_path)[0]
    comp_bboxes = []
    for box in (r_comp.boxes or []):
        name = comp_model.names[int(box.cls[0])]
        # Skip text/junction bboxes from CGHD — they're not component interiors
        if USE_CGH and name in ("text", "junction", "crossover", "terminal"):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        comp_bboxes.append((x1, y1, x2, y2))

    def inside_any_component(tx1, ty1, tx2, ty2):
        """Check if text region overlaps significantly with component bbox."""
        area_t = (tx2-tx1)*(ty2-ty1)
        for cx1, cy1, cx2, cy2 in comp_bboxes:
            ox1 = max(tx1, cx1); oy1 = max(ty1, cy1)
            ox2 = min(tx2, cx2); oy2 = min(ty2, cy2)
            if ox2 > ox1 and oy2 > oy1:
                overlap = (ox2-ox1)*(oy2-oy1)
                if overlap > area_t * 0.5:  # >50% overlap → inside component
                    return True
        return False

    results = text_model(img_path)[0]
    values = []
    for box in (results.boxes or []):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Skip text regions inside component bboxes (false positives from symbols)
        if inside_any_component(x1, y1, x2, y2):
            continue

        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Use CRNN if available
        if CRNN_MODEL is not None:
            raw = crnn_predict(CRNN_MODEL, crop, CRNN_CHARS, CRNN_ITOC, CRNN_IMG_H)
            raw = _clean_ocr(raw)
        else:
            # Tesseract fallback
            import pytesseract
            TESSERACT_PATH = r"E:/Tesseract-OCR/tesseract.exe"
            if os.path.isfile(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            os.environ.setdefault("TESSDATA_PREFIX", r"E:/Tesseract-OCR/tessdata")

            h, w = crop.shape[:2]
            if h < 25:
                crop = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for polarity in [thresh, 255 - thresh]:
                raw = pytesseract.image_to_string(polarity, config="--psm 7").strip()
                raw = _clean_ocr(raw)
                if raw:
                    break

        if raw:
            values.append(dict(
                text=raw, cx=(x1+x2)//2, cy=(y1+y2)//2,
                xyxy=(x1,y1,x2,y2), conf=float(box.conf[0])
            ))
    return values


def _clean_ocr(text: str) -> str:
    """Clean OCR output + normalize circuit notation like '3V3' -> '3.3V'."""
    import re
    text = text.strip().replace(" ", "").replace("\n", "")
    if len(text) < 1:
        return ""
    # Filter lines that are mostly non-circuit characters
    valid = sum(1 for c in text if c.isalnum() or c in ".kKmMΩμunpFVAHz-+")
    if valid < len(text) * 0.5:
        return ""

    # Normalize circuit notation: "3V3" → "3.3V", "1k2" → "1.2kΩ", "4R7" → "4.7Ω"
    # Also handle CRNN errors: "33V" might be "3.3V" (lost middle char)
    def _expand_short(val):
        m = re.match(r"^(\d+)([Rr])(\d+)(.*)$", val)   # 4R7 → 4.7Ω
        if m: return f"{m.group(1)}.{m.group(3)}Ω"
        m = re.match(r"^(\d+)([kKmM])(\d+)([Ω]?)$", val)  # 1k2 → 1.2kΩ
        if m: return f"{m.group(1)}.{m.group(3)}{m.group(2)}{m.group(4) or 'Ω'}"
        m = re.match(r"^(\d+)([Vv])(\d+)$", val)       # 3V3 → 3.3V
        if m: return f"{m.group(1)}.{m.group(3)}V"
        m = re.match(r"^(\d+)([Aa])(\d+)$", val)       # 1A5 → 1.5A
        if m: return f"{m.group(1)}.{m.group(3)}A"
        # Catch CRNN errors: "33V" near V-source might be "3.3V", "10k" might be "1.0k"
        m = re.match(r"^(\d{2})[Vv]$", val)              # 33V → maybe 3.3V
        if m and int(m.group(1)) <= 50:  # likely voltage, not 33V but 3.3V
            return f"{int(m.group(1))/10:.1f}V"
        return val
    text = _expand_short(text)

    return text


def match_values(components: list[dict], text_values: list[dict]) -> list[dict]:
    """Match OCR text values to nearest compatible component. Infers missing units."""
    import re
    # Default units per component type
    default_units = {"Resistor": "Ω", "Capacitor": "F", "Inductor": "H",
                     "V-DC": "V", "V-AC": "V", "I-DC": "A", "I-AC": "A"}

    for c in components:
        best_text = ""
        best_score = 99999
        # GND should never have a value (it's a reference node, not a component with a value)
        if c["name"] == "GND":
            c["value"] = ""
            continue
        for tv in text_values:
            dist = abs(tv["cx"] - c["cx"]) + abs(tv["cy"] - c["cy"])
            if dist > 250:
                continue
            v = tv["text"]

            # Check anti-patterns first
            anti = ANTI_PATTERNS.get(c["name"], [])
            if any(re.search(ap, v) for ap in anti):
                continue  # value has wrong unit for this component type

            pattern, default_unit = VALUE_PATTERNS.get(c["name"], (None, ""))
            # If value doesn't end with a known unit, try appending default unit
            if not re.search(r"[ΩFHVAmHμ]$", v):
                v_test = v + default_unit
            else:
                v_test = v
            if pattern and not re.match(pattern, v_test):
                continue
            if dist < best_score:
                # Store with inferred unit if needed
                if not re.search(r"[ΩFHVAmH]$", v):
                    best_text = v + default_unit
                else:
                    best_text = v
                best_score = dist
        c["value"] = best_text
    return components


def build_description(components: list[dict], text_values: list[dict] = None) -> str:
    """Convert detected components to natural language description.

    Also infers power sources from unmatched voltage/current OCR text values.
    """
    lines = []
    lines.append("以下是YOLO目标检测从电路图中自动识别的元器件：")
    lines.append("")

    type_counts = {}
    for c in components:
        if c["name"] in ("Wire Crossover",):
            continue
        type_counts[c["name"]] = type_counts.get(c["name"], 0) + 1

    for comp_type in sorted(type_counts.keys()):
        items = [c for c in components if c["name"] == comp_type]
        for i, c in enumerate(items):
            label = c.get("display", c["name"])
            if c["value"]:
                lines.append(f"- {label}{i+1}：数值={c['value']}")
            else:
                lines.append(f"- {label}{i+1}：数值未识别")

    # Infer power sources from unmatched text values
    if text_values:
        has_v_source = any(c["name"].startswith("V-") for c in components)
        has_i_source = any(c["name"].startswith("I-") for c in components)
        import re
        for tv in text_values:
            txt = tv["text"]
            if re.search(r"[\d.]+[Vv]$", txt) and not has_v_source:
                lines.insert(1, f"- 推测电压源：数值={txt}（检测到电压标注但元件漏检）")
                has_v_source = True
            if re.search(r"[\d.]+[Aa]$", txt) and not has_i_source:
                lines.insert(1, f"- 推测电流源：数值={txt}（检测到电流标注但元件漏检）")
                has_i_source = True

    # Remove Wire Crossovers from count
    real_count = sum(1 for c in components if c["name"] != "Wire Crossover")
    lines.insert(1, f"（共检测到{real_count}个元器件）")

    lines.append("")
    lines.append("请作为电路评审专家，评价以上元器件组成的电路图：")
    lines.append("1. 是否存在元器件数值标注错误（如电容标了电流单位）？")
    lines.append("2. 根据元器件组合，推测这可能是什么类型的电路？")
    lines.append("3. 整体质量如何？有无明显设计缺陷？")

    # Special hint for LED circuits
    has_diode = any(c["name"] == "Diode" for c in components)
    has_resistor = any(c["name"] == "Resistor" for c in components)
    if has_diode and has_resistor:
        lines.append("4. （注意：如果二极管实际是LED，请分析限流电阻阻值是否合理、是否需要调整）")

    return "\n".join(lines)


def evaluate_image(img_path: str) -> dict:
    """Full pipeline: image → detect → OCR → LLM evaluation.

    Saves result to `<imagename>_result.txt` alongside the image.
    """
    img_name = os.path.basename(img_path)
    print(f"\n{'='*60}")
    print(f"  图像评价: {img_name}")
    print(f"{'='*60}")

    # Step 1: Detect components
    components = detect_components(img_path)
    print(f"\n[1] 元件检测: {len(components)} 个")
    for c in components:
        print(f"    [{c['conf']:.0%}] {c['name']:15s} @ ({c['cx']},{c['cy']})")

    # Step 2: OCR text values
    text_values = read_text_values(img_path)
    print(f"\n[2] 文字OCR: {len(text_values)} 个")
    for tv in text_values:
        print(f"    [{tv['conf']:.0%}] \"{tv['text']}\" @ ({tv['cx']},{tv['cy']})")

    # Step 3: Match values to components
    components = match_values(components, text_values)
    print(f"\n[3] 数值匹配:")
    for c in components:
        val_str = f"= {c['value']}" if c['value'] else "(未识别)"
        print(f"    {c['name']:15s} {val_str}")

    # Step 4: LLM evaluation
    desc = build_description(components, text_values)
    print(f"\n[4] LLM评价中...")
    response = ask(desc)
    print(f"\n{'='*60}")
    print(f"  LLM 评价结果")
    print(f"{'='*60}")
    print(response)

    # Save text result
    out_txt = Path(img_path).parent / (Path(img_path).stem + "_result.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"图像评价结果: {img_name}\n{'='*60}\n\n")
        f.write(f"[元件检测] 共 {len(components)} 个\n\n")
        for c in components:
            v = f"= {c['value']}" if c['value'] else "(未识别)"
            f.write(f"  [{c['conf']:.0%}] {c['name']:15s} {v}\n")
        f.write(f"\n[OCR文字] 共 {len(text_values)} 个\n\n")
        for tv in text_values:
            f.write(f"  [{tv['conf']:.0%}] \"{tv['text']}\"\n")
        f.write(f"\n[LLM评价]\n{'='*60}\n{response}")

    # Save annotated image
    out_img = Path(img_path).parent / (Path(img_path).stem + "_annotated.jpg")
    _draw_annotated(img_path, components, text_values, str(out_img))

    print(f"\n结果已保存: {out_txt}")
    print(f"标注图已保存: {out_img}")

    return dict(
        image=img_path, components=components, text_values=text_values,
        description=desc, evaluation=response, result_file=str(out_txt),
        annotated_image=str(out_img),
    )


def _draw_annotated(img_path: str, components: list[dict], text_values: list[dict], out_path: str):
    """Draw component bboxes, OCR values, and port points on the circuit image."""
    import cv2, numpy as np
    img = cv2.imread(img_path)
    if img is None:
        return

    # Component colors
    colors = {
        "Resistor": (0, 200, 0), "Capacitor": (200, 200, 0),
        "Inductor": (0, 200, 200), "Diode": (200, 0, 100),
        "V-DC": (200, 0, 0), "V-AC": (150, 0, 0),
        "I-DC": (0, 100, 0), "I-AC": (0, 150, 100),
        "GND": (0, 0, 200), "MOSFET-N": (100, 100, 0),
        "MOSFET-P": (100, 0, 100), "BJT-NPN": (150, 150, 0),
        "BJT-PNP": (150, 0, 150), "Op-Amp": (0, 100, 100),
        "Zener Diode": (200, 0, 150),
    }

    # Draw components
    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        color = colors.get(c["name"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = c.get("display", c["name"])
        if c["value"]:
            label += "=" + c["value"]
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
        # Draw ports as red dots
        for px, py in c.get("ports", []):
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1)

    # Draw OCR text regions (dashed box)
    for tv in text_values:
        x1, y1, x2, y2 = tv.get("xyxy", (0, 0, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 1)
        cv2.putText(img, f'OCR:"{tv["text"]}"', (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1, cv2.LINE_AA)

    # Legend
    y = 20
    for name, color in sorted(colors.items()):
        cv2.putText(img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)
        y += 15

    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test2.jpg"
    evaluate_image(img)
