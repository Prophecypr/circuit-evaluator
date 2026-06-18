"""Unified circuit evaluation pipeline: YOLO detection + OCR + junction wiring + LLM eval.

用法:
    python -m src.vision.unified_pipeline circuit_5.jpg
    python -m src.vision.unified_pipeline circuit_5.jpg circuit_6.jpg circuit_8.jpg
"""

import sys, os, cv2, json, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from src.llm import ask
import torch
from src.vision.train_ocr import load_trained_model, predict as crnn_predict
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "runs", "detect")

CGH_SKIP = {"crossover", "probe.current", "probe.voltage"}
CGH_CONF_THRESH = 0.40
JUNCTION_CONF = 0.10
PORT_JUNCTION_RADIUS = 300
PORT_JUNCTION_FALLBACK = 250  # wider radius for second-pass matching
JJ_ALIGN_PX = 15  # horizontal/vertical alignment tolerance (hand-drawn variance)
JJ_PROXIMITY = 12  # connect junctions within this range even without alignment
JJ_MAX_ALIGNED_DIST = 1000  # max distance for aligned junction pairs (crosses+would_short prevent false merges)
GRID_SNAP_TOL = 5
UF_MERGE_TOL = 8

# Ablation experiment config (all True = full algorithm)
DEFAULT_CONFIG = {
    "use_skeleton": True, "use_degree_constraint": False,
    "use_nn_filter": True, "use_skel_jj": True,
    "use_sobel": True, "use_close_port": True,
    "use_force_connect": True, "use_los": True,
    "use_ccl": False,
    "skip_llm": False,
}

# CGHD → HCD name mapping
CGH_NAME_MAP = {
    "resistor": "Resistor", "capacitor.unpolarized": "Capacitor",
    "capacitor.polarized": "Polarized-Capacitor", "capacitor.adjustable": "Capacitor",
    "inductor": "Inductor", "inductor.ferrite": "Inductor",
    "diode": "Diode", "diode.light_emitting": "LED", "diode.zener": "Zener Diode",
    "diode.thyrector": "Diode",
    "voltage.dc": "V-DC", "voltage.ac": "V-AC", "voltage.battery": "Battery",
    "gnd": "GND", "vss": "GND",
    "transistor.bjt": "BJT-PNP", "transistor.fet": "MOSFET-P",
    "operational_amplifier": "Op-Amp",
    "thyristor": "Thyristor", "triac": "Triac", "diac": "Diac", "varistor": "Varistor",
    "lamp": "Lamp",
    # P2: remaining CGHD classes
    "resistor.adjustable": "Potentiometer", "resistor.photo": "Photo-Resistor",
    "inductor.coupled": "Transformer", "transformer": "Transformer",
    "operational_amplifier.schmitt_trigger": "Schmitt-Trigger",
    "optocoupler": "Optocoupler",
    "integrated_circuit": "IC", "integrated_circuit.ne555": "NE555",
    "integrated_circuit.voltage_regulator": "Voltage-Regulator",
    "transistor.photo": "Photo-Transistor",
    "xor": "XOR", "and": "AND", "or": "OR", "not": "NOT",
    "nand": "NAND", "nor": "NOR",
    "switch": "Switch", "relay": "Relay", "socket": "Socket",
    "fuse": "Fuse", "speaker": "Speaker", "motor": "Motor",
    "microphone": "Microphone", "antenna": "Antenna",
    "crystal": "Crystal", "mechanical": "Mechanical",
    "magnetic": "Magnetic", "optical": "Optical",
    "block": "Block", "explanatory": "Explanatory", "unknown": "Unknown",
}

CGH_DISPLAY = {
    "resistor": "R", "capacitor.unpolarized": "C", "capacitor.polarized": "Cp",
    "inductor": "L", "diode": "D", "diode.light_emitting": "LED",
    "diode.zener": "ZD", "voltage.dc": "Vdc", "voltage.ac": "Vac",
    "gnd": "GND", "transistor.bjt": "BJT", "transistor.fet": "FET",
    "operational_amplifier": "OpAmp",
    "thyristor": "SCR", "triac": "TRIAC", "diac": "DIAC", "varistor": "VDR",
    "lamp": "Lamp",
    # P2
    "resistor.adjustable": "RP", "resistor.photo": "Rph",
    "inductor.coupled": "TX", "transformer": "TX",
    "operational_amplifier.schmitt_trigger": "Sch",
    "optocoupler": "OC",
    "integrated_circuit": "IC", "integrated_circuit.ne555": "555",
    "integrated_circuit.voltage_regulator": "VR",
    "transistor.photo": "Qph",
    "xor": "XOR", "and": "AND", "or": "OR", "not": "NOT",
    "nand": "NAND", "nor": "NOR",
    "switch": "SW", "relay": "K", "socket": "J",
    "fuse": "F", "speaker": "SP", "motor": "M",
    "microphone": "MIC", "antenna": "ANT",
    "crystal": "XTAL", "mechanical": "MECH",
    "magnetic": "MAG", "optical": "OPT",
    "block": "BLK", "explanatory": "EXP", "unknown": "UNK",
}

CGH_TO_PORT_KEY = {
    "resistor": "Resistor", "resistor.adjustable": "Potentiometer",
    "resistor.photo": "Photo-Resistor",
    "capacitor.unpolarized": "Capacitor", "capacitor.polarized": "Polarized-Capacitor",
    "capacitor.adjustable": "Capacitor",
    "inductor": "Inductor", "inductor.ferrite": "Inductor",
    "inductor.coupled": "Transformer", "transformer": "Transformer",
    "diode": "Diode", "diode.light_emitting": "LED",
    "diode.zener": "Zener Diode", "diode.thyrector": "Diode",
    "voltage.dc": "V-DC", "voltage.ac": "V-AC", "voltage.battery": "Battery",
    "gnd": "GND", "vss": "GND",
    "transistor.bjt": "BJT-PNP", "transistor.fet": "MOSFET-P",
    "transistor.photo": "Photo-Transistor",
    "operational_amplifier": "Op-Amp",
    "operational_amplifier.schmitt_trigger": "Schmitt-Trigger",
    "optocoupler": "Optocoupler",
    "integrated_circuit": "IC", "integrated_circuit.ne555": "NE555",
    "integrated_circuit.voltage_regulator": "Voltage-Regulator",
    "thyristor": "Thyristor", "triac": "Triac", "diac": "Diac", "varistor": "Varistor",
    "xor": "XOR", "and": "AND", "or": "OR", "not": "NOT",
    "nand": "NAND", "nor": "NOR",
    "switch": "Switch", "relay": "Relay", "socket": "Socket",
    "fuse": "Fuse", "speaker": "Speaker", "motor": "Motor",
    "microphone": "Microphone", "antenna": "Antenna",
    "crystal": "Crystal", "mechanical": "Mechanical",
    "magnetic": "Magnetic", "optical": "Optical",
    "block": "Block", "explanatory": "Explanatory", "unknown": "Unknown",
    "lamp": "Lamp",
}

# Port labels for polarity/terminal identification
PORT_LABELS = {
    "Resistor":       ["1", "2"],
    "Capacitor":      ["1", "2"],
    "Polarized-Capacitor": ["+", "-"],
    "Inductor":       ["1", "2"],
    "Diode":          ["A", "K"],
    "LED":            ["+", "-"],
    "Zener Diode":    ["A", "K"],
    "Thyristor":      ["A", "K"],
    "Triac":          ["T1", "T2"],
    "Diac":           ["T1", "T2"],
    "Varistor":       ["1", "2"],
    "V-DC":           ["+", "-"],
    "V-AC":           ["~", "~"],
    "I-DC":           ["-", "+"],
    "I-AC":           ["~", "~"],
    "GND":            ["GND"],
    "BJT-NPN":        ["B", "E", "C"],
    "BJT-PNP":        ["B", "E", "C"],
    "MOSFET-N":       ["G", "S", "D"],
    "MOSFET-P":       ["G", "S", "D"],
    "Op-Amp":         ["-", "+", "OUT"],
    "Lamp":           ["1", "2"],
    # P2
    "Potentiometer":   ["1", "2", "W"],
    "Photo-Resistor":  ["1", "2"],
    "Transformer":     ["P1", "P2", "S1", "S2"],
    "Schmitt-Trigger": ["-", "+", "OUT"],
    "Optocoupler":     ["A", "K", "C", "E"],
    "IC":              ["-", "+", "OUT"],
    "NE555":           ["GND","TRIG","OUT","RST","CTRL","THR","DIS","VCC"],
    "Voltage-Regulator": ["IN", "OUT", "GND"],
    "Photo-Transistor": ["C", "E"],
    "XOR":             ["A", "B", "Y"],
    "AND":             ["A", "B", "Y"],
    "OR":              ["A", "B", "Y"],
    "NOT":             ["A", "Y"],
    "NAND":            ["A", "B", "Y"],
    "NOR":             ["A", "B", "Y"],
    "Switch":          ["1", "2"],
    "Relay":           ["COIL1", "COIL2", "COM"],
    "Socket":          ["1", "2"],
    "Fuse":            ["1", "2"],
    "Speaker":         ["1", "2"],
    "Motor":           ["+", "-"],
    "Microphone":      ["+", "-"],
    "Antenna":         ["ANT"],
    "Crystal":         ["1", "2"],
    "Mechanical":      ["1", "2"],
    "Magnetic":        ["1", "2"],
    "Optical":         ["1", "2"],
    "Block":           ["1", "2"],
    "Explanatory":     [],
    "Unknown":         [],
}

PORT_POSITIONS = {
    "Resistor":       [(0,0.5), (1,0.5)],
    "Capacitor":      [(0,0.5), (1,0.5)],
    "Polarized-Capacitor": [(0,0.5), (1,0.5)],
    "Inductor":       [(0,0.5), (1,0.5)],
    "Diode":          [(0,0.5), (1,0.5)],
    "LED":            [(0,0.5), (1,0.5)],
    "Zener Diode":    [(0,0.5), (1,0.5)],
    "Thyristor":      [(0,0.5), (1,0.5)],
    "Triac":          [(0,0.5), (1,0.5)],
    "Diac":           [(0,0.5), (1,0.5)],
    "Varistor":       [(0,0.5), (1,0.5)],
    "V-DC":           [(0.5,0.0), (0.5,1.0)],
    "V-AC":           [(0.5,0.0), (0.5,1.0)],
    "I-DC":           [(0.5,0.0), (0.5,1.0)],
    "I-AC":           [(0.5,0.0), (0.5,1.0)],
    "GND":            [(0.5,0.0)],
    "BJT-NPN":        [(0.0,0.5), (0.7,0.0), (0.7,1.0)],
    "BJT-PNP":        [(0.0,0.5), (0.7,0.0), (0.7,1.0)],
    "MOSFET-N":       [(0,0.5), (0.5,1.0), (0.5,0.0)],
    "MOSFET-P":       [(0,0.5), (0.5,1.0), (0.5,0.0)],
    "Op-Amp":         [(0,0.5), (0,0.3), (1,0.5)],
    "Lamp":           [(0,0.5), (1,0.5)],
    # P2: new component types
    "Potentiometer":   [(0,0.5), (1,0.5), (0.5,1.0)],
    "Photo-Resistor":  [(0,0.5), (1,0.5)],
    "Transformer":     [(0,0.5), (1,0.5), (0,0.2), (1,0.2)],
    "Schmitt-Trigger": [(0,0.5), (0,0.3), (1,0.5)],
    "Optocoupler":     [(0,0.5), (1,0.5), (0,0.8), (1,0.8)],
    "IC":              [(0,0.3), (0,0.7), (1,0.5)],
    "NE555":           [(0,0.2), (0,0.4), (0,0.6), (0,0.8), (1,0.5), (0.5,1.0), (0.5,0.0), (1,0.3)],
    "Voltage-Regulator": [(0,0.5), (1,0.5), (0.5,0.0)],
    "Photo-Transistor": [(0.5,1.0), (0.5,0.0)],
    "XOR":             [(0,0.5), (1,0.5), (0.5,0.0)],
    "AND":             [(0,0.5), (1,0.5), (0.5,0.0)],
    "OR":              [(0,0.5), (1,0.5), (0.5,0.0)],
    "NOT":             [(0,0.5), (1,0.5), (0.5,0.0)],
    "NAND":            [(0,0.5), (1,0.5), (0.5,0.0)],
    "NOR":             [(0,0.5), (1,0.5), (0.5,0.0)],
    "Switch":          [(0,0.5), (1,0.5)],
    "Relay":           [(0,0.3), (0,0.7), (1,0.5)],
    "Socket":          [(0,0.5), (1,0.5)],
    "Fuse":            [(0,0.5), (1,0.5)],
    "Speaker":         [(0,0.5), (1,0.5)],
    "Motor":           [(0,0.5), (1,0.5)],
    "Microphone":      [(0,0.5), (1,0.5)],
    "Antenna":         [(0.5,0.0)],
    "Crystal":         [(0,0.5), (1,0.5)],
    "Mechanical":      [(0,0.5), (1,0.5)],
    "Magnetic":        [(0,0.5), (1,0.5)],
    "Optical":         [(0,0.5), (1,0.5)],
    "Block":           [(0,0.5), (1,0.5)],
    "Explanatory":     [],
    "Unknown":         [],
    "Wire Crossover": [],
}

VALUE_PATTERNS = {
    "Resistor": (r"^[\d.]+[kKM]?\s*[Ω]?$", "Ω"),
    "Capacitor": (r"^[\d.]+[μunpUNPmM]?\s*F?$", "F"),
    "Polarized-Capacitor": (r"^[\d.]+[μunpUNPmM]?\s*F?$", "F"),
    "Inductor": (r"^[\d.]+[μumM]?\s*H?$", "H"),
    "V-DC": (r"^[\d.]+[μumM]?\s*V?$", "V"),
    "V-AC": (r"^[\d.]+[μumM]?\s*V?$", "V"),
    "I-DC": (r"^[\d.]+[μumM]?\s*A?$", "A"),
    "I-AC": (r"^[\d.]+[μumM]?\s*A?$", "A"),
}

ANTI_PATTERNS = {
    "Diode":       [r"[AVΩ]$", r"[Fμ]$", r"^[\d.]+\s*[FHz]$", r"[Hh]$"],
    "LED":         [r"[AVΩ]$", r"[Fμ]$", r"^[\d.]+\s*[FHz]$", r"[Hh]$"],
    "Zener Diode": [r"[AVΩ]$", r"[Fμ]$", r"^[\d.]+\s*[FHz]$", r"[Hh]$"],
    "GND":         [r".*"],
    "Wire Crossover": [r".*"],
    "MOSFET-N":    [r"[ΩFVHzμAH]$"],
    "MOSFET-P":    [r"[ΩFVHzμAH]$"],
    "BJT-NPN":     [r"[ΩFVHzμAH]$"],
    "BJT-PNP":     [r"[ΩFVHzμAH]$"],
    "Op-Amp":      [r"[ΩFVHzμAH]$"],
    "Thyristor":   [r"[ΩFVHzμAH]$"],
    "Triac":       [r"[ΩFVHzμAH]$"],
    "Diac":        [r"[ΩFVHzμAH]$"],
    "Varistor":    [r"[ΩFVHzμAH]$"],
    "lamp":        [r"[AVΩH].*$"],
}

# Same-family groups for NMS dedup (overlapping detections of related classes)
NMS_FAMILIES = [
    {"diode", "diode.light_emitting", "diode.zener", "diode.thyrector"},
    {"voltage.dc", "voltage.ac", "voltage.battery"},
    {"gnd", "vss"},
    {"capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable"},
    {"integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator"},
]
NMS_IOU_THRESH = 0.35  # IoU above which same-family overlapping detections are merged

DESIG = {"Resistor": "R", "Capacitor": "C", "Polarized-Capacitor": "C", "Inductor": "L",
         "Diode": "D", "Zener Diode": "ZD", "LED": "LED",
         "V-DC": "V", "V-AC": "V", "I-DC": "I", "I-AC": "I",
         "GND": "GND", "BJT-NPN": "Q", "BJT-PNP": "Q",
         "MOSFET-N": "Q", "MOSFET-P": "Q", "Op-Amp": "U",
         "Thyristor": "SCR", "Triac": "TRIAC", "Diac": "DIAC", "Varistor": "VDR",
         "Lamp": "Lamp",
         "Potentiometer": "RP", "Photo-Resistor": "Rph", "Transformer": "TX",
         "Schmitt-Trigger": "U", "Optocoupler": "OC", "IC": "IC",
         "NE555": "U", "Voltage-Regulator": "VR", "Photo-Transistor": "Qph",
         "XOR": "U", "AND": "U", "OR": "U", "NOT": "U", "NAND": "U", "NOR": "U",
         "Switch": "SW", "Relay": "K", "Socket": "J",
         "Fuse": "F", "Speaker": "SP", "Motor": "M",
         "Microphone": "MIC", "Antenna": "ANT",
         "Crystal": "XTAL", "Mechanical": "MECH",
         "Magnetic": "MAG", "Optical": "OPT",
         "Block": "BLK", "Explanatory": "EXP", "Unknown": "UNK"}

NM_CH = {"Resistor": "电阻", "Capacitor": "电容", "Polarized-Capacitor": "极性电容", "Inductor": "电感",
         "Diode": "二极管", "LED": "发光二极管", "Zener Diode": "稳压管", "GND": "地线",
         "V-DC": "直流电压源", "V-AC": "交流电压源",
         "I-DC": "直流电流源", "I-AC": "交流电流源",
         "BJT-NPN": "NPN三极管", "BJT-PNP": "PNP三极管",
         "MOSFET-N": "N沟道MOSFET", "MOSFET-P": "P沟道MOSFET",
         "Op-Amp": "运算放大器", "Thyristor": "晶闸管",
         "Triac": "双向晶闸管", "Diac": "双向触发二极管", "Varistor": "压敏电阻",
         "Lamp": "灯泡",
         "Potentiometer": "电位器", "Photo-Resistor": "光敏电阻",
         "Transformer": "变压器", "Schmitt-Trigger": "施密特触发器",
         "Optocoupler": "光耦", "IC": "集成电路", "NE555": "NE555定时器",
         "Voltage-Regulator": "稳压器", "Photo-Transistor": "光电三极管",
         "XOR": "异或门", "AND": "与门", "OR": "或门", "NOT": "非门",
         "NAND": "与非门", "NOR": "或非门",
         "Switch": "开关", "Relay": "继电器", "Socket": "插座",
         "Fuse": "保险丝", "Speaker": "扬声器", "Motor": "电机",
         "Microphone": "麦克风", "Antenna": "天线",
         "Crystal": "晶振", "Mechanical": "机械元件",
         "Magnetic": "磁性元件", "Optical": "光学元件",
         "Block": "模块", "Explanatory": "说明文字", "Unknown": "未知"}

# ---------------------------------------------------------------------------
# OCR Correction Memory (persistent learning from corrections)
# ---------------------------------------------------------------------------
CORRECTION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "ocr_corrections.json")

def _load_corrections():
    """Load OCR correction memory from JSON file."""
    if os.path.isfile(CORRECTION_FILE):
        with open(CORRECTION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_corrections(corrections):
    """Save OCR correction memory to JSON file."""
    with open(CORRECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

def _apply_corrections(text, corrections):
    """Apply known OCR corrections. corrections = {ocr_pattern: corrected_value}."""
    for pattern, replacement in corrections.items():
        if pattern in text:
            return text.replace(pattern, replacement)
    return text

def add_correction(ocr_text, correct_value):
    """Add a correction to memory. Called when user identifies an OCR error."""
    corrections = _load_corrections()
    # Only add if OCR text is clearly wrong for its context
    corrections[ocr_text] = correct_value
    _save_corrections(corrections)
    print(f"  Correction saved: '{ocr_text}' -> '{correct_value}'")

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_CGH_MODEL = None
_OCR_MODEL = None
_OCR_CHARS = ""
_OCR_ITOC = {}
_OCR_IMG_H = 32

def _load_models():
    global _CGH_MODEL, _OCR_MODEL, _OCR_CHARS, _OCR_ITOC, _OCR_IMG_H
    if _CGH_MODEL is None:
        cghd_path = os.path.join(MODELS_DIR, "cghd_61cls", "weights", "best.pt")
        if not os.path.isfile(cghd_path):
            raise FileNotFoundError(f"CGHD model not found: {cghd_path}")
        _CGH_MODEL = YOLO(cghd_path)
        print(f"YOLO: CGHD 61-class loaded ({len(_CGH_MODEL.names)} classes)")

    if _OCR_MODEL is None:
        _OCR_MODEL, _OCR_CHARS, _, _OCR_ITOC, _OCR_IMG_H = load_trained_model("runs/ocr_crnn_machine/best.pt")
        print(f"OCR: CRNN loaded ({len(_OCR_CHARS)} chars)")
    return _CGH_MODEL

# ---------------------------------------------------------------------------
# Sobel orientation detection
# ---------------------------------------------------------------------------
def _detect_ic_ports(img_path, x1, y1, x2, y2, max_ports=4):
    """Detect IC pin attachment points by scanning bbox perimeter for wire entry.
    Returns list of (rx, ry) normalized port positions, limited to max_ports by edge strength.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c + 5 or y2c <= y1c + 5:
        return []
    crop = gray[y1c:y2c, x1c:x2c]
    ch, cw = crop.shape

    # Edge detection on crop
    edges = cv2.Canny(crop, 40, 120)
    border_px = 6  # scan this many pixels from border

    candidates = []  # (strength, rx, ry)
    # Scan left edge
    left_region = edges[:, :border_px]
    row_sums = left_region.sum(axis=1)
    if row_sums.max() > 0:
        for r in range(1, ch - 1):
            if row_sums[r] > row_sums[r-1] and row_sums[r] >= row_sums[r+1] and row_sums[r] > 10:
                # Check not clustered with already-added
                too_close = any(abs(r/ch - ry) * ch < 8 for _, _, ry in candidates if abs(_ - 0) < 0.01)
                if not too_close:
                    candidates.append((float(row_sums[r]), 0.0, r / ch))

    # Scan right edge
    right_region = edges[:, cw - border_px:]
    row_sums = right_region.sum(axis=1)
    if row_sums.max() > 0:
        for r in range(1, ch - 1):
            if row_sums[r] > row_sums[r-1] and row_sums[r] >= row_sums[r+1] and row_sums[r] > 10:
                too_close = any(abs(r/ch - ry) * ch < 8 for _, rx, ry in candidates if abs(rx - 1.0) < 0.01)
                if not too_close:
                    candidates.append((float(row_sums[r]), 1.0, r / ch))

    # Scan top edge
    top_region = edges[:border_px, :]
    col_sums = top_region.sum(axis=0)
    if col_sums.max() > 0:
        for c in range(1, cw - 1):
            if col_sums[c] > col_sums[c-1] and col_sums[c] >= col_sums[c+1] and col_sums[c] > 10:
                too_close = any(abs(c/cw - rx) * cw < 8 for _, rx, ry in candidates if abs(ry - 0.0) < 0.01)
                if not too_close:
                    candidates.append((float(col_sums[c]), c / cw, 0.0))

    # Scan bottom edge
    bottom_region = edges[ch - border_px:, :]
    col_sums = bottom_region.sum(axis=0)
    if col_sums.max() > 0:
        for c in range(1, cw - 1):
            if col_sums[c] > col_sums[c-1] and col_sums[c] >= col_sums[c+1] and col_sums[c] > 10:
                too_close = any(abs(c/cw - rx) * cw < 8 for _, rx, ry in candidates if abs(ry - 1.0) < 0.01)
                if not too_close:
                    candidates.append((float(col_sums[c]), c / cw, 1.0))

    # Sort by edge strength (strongest first), take top max_ports
    candidates.sort(key=lambda x: x[0], reverse=True)
    ports = [(rx, ry) for _, rx, ry in candidates[:max_ports]]

    # Fall back if too few detected
    if len(ports) < 2:
        return [(0, 0.3), (0, 0.7), (1, 0.5)]
    return ports


def _detect_orientation(img_path, x1, y1, x2, y2, plist, raw_name="", use_sobel=True):
    """Determine if a 2-pin component needs port rotation.

    LEDs: use aspect ratio only (Sobel unreliable — internal triangle/arrows mislead).
    Capacitors: use Sobel if available, otherwise aspect ratio.
    Others: aspect ratio.
    """
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    is_default_h = abs(plist[0][0] - plist[1][0]) > abs(plist[0][1] - plist[1][1])
    is_cap = "capacitor" in raw_name.lower() if raw_name else False
    is_led = "light_emitting" in raw_name.lower() if raw_name else False
    ratio = bh / max(bw, 1)

    if is_led:
        # LED default is horizontal (ports left-right). Tall-narrow LED -> vertical ports.
        return ratio > 1.4

    if is_cap and use_sobel:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 > cx1 + 10 and cy2 > cy1 + 10:
                crop = cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
                ve = float(np.sum(np.abs(cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3))))
                he = float(np.sum(np.abs(cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3))))
                if he + ve > 0:
                    # Horizontal edges stronger → capacitor plates are horizontal lines
                    # → symbol is drawn vertically → ports top/bottom → NEED ROTATION
                    return he > ve * 1.3
        return ratio > 1.8

    if is_default_h:
        return ratio > 1.3
    else:
        return ratio < 0.77
# ---------------------------------------------------------------------------
# Grid snap + Union-Find merging
# ---------------------------------------------------------------------------
def snap(points, tol=GRID_SNAP_TOL):
    """Cluster nearby points by coordinate and snap each dimension to cluster median."""
    if len(points) <= 1:
        return [(int(x), int(y)) for x, y in points]
    for dim in (0, 1):
        vals = sorted(set(p[dim] for p in points))
        clusters = []
        cur = [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                clusters.append(cur)
                cur = [v]
        clusters.append(cur)
        median_map = {}
        for cl in clusters:
            m = int(np.median(cl))
            for v in cl:
                median_map[v] = m
        points = [(median_map.get(p[0], p[0]), median_map.get(p[1], p[1])) for p in points]
    return [(int(x), int(y)) for x, y in points]

def uf_merge(points, tol=UF_MERGE_TOL):
    """Union-Find: merge points within tol distance. Returns dict {original_idx: representative_point}."""
    n = len(points)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) <= tol:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    result = {}
    for root, indices in groups.items():
        rx = int(np.mean([points[i][0] for i in indices]))
        ry = int(np.mean([points[i][1] for i in indices]))
        for i in indices:
            result[i] = (rx, ry)
    return result

def _snap_ports_to_grid(components, t_h=5, t_v=5, t_same=8):
    """Snap port coordinates to common grid lines (per 预处理规则 steps 1-2).

    Step 1: Cluster y coords within t_h, snap to cluster median → same row.
    Step 2: Cluster x coords within t_v, snap to cluster median → same column.
    Step 3: Merge ports within t_same (both dims) → midpoint.

    Modifies component port coordinates in-place.
    """
    if not components:
        return
    # Collect all port→component references
    port_refs = []  # [(ci, pi, x, y)]
    for ci, c in enumerate(components):
        for pi, (px, py) in enumerate(c["ports"]):
            port_refs.append([ci, pi, px, py])

    if len(port_refs) <= 1:
        return

    # Step 1: Horizontal alignment (cluster y, snap to median)
    ys = sorted(set(p[3] for p in port_refs))
    y_clusters = []
    cur = [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= t_h:
            cur.append(y)
        else:
            y_clusters.append(cur)
            cur = [y]
    y_clusters.append(cur)
    y_map = {}
    for cl in y_clusters:
        med = int(np.median(cl))
        for y in cl:
            y_map[y] = med
    for p in port_refs:
        p[3] = y_map.get(p[3], p[3])

    # Step 2: Vertical alignment (cluster x, snap to median)
    xs = sorted(set(p[2] for p in port_refs))
    x_clusters = []
    cur = [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= t_v:
            cur.append(x)
        else:
            x_clusters.append(cur)
            cur = [x]
    x_clusters.append(cur)
    x_map = {}
    for cl in x_clusters:
        med = int(np.median(cl))
        for x in cl:
            x_map[x] = med
    for p in port_refs:
        p[2] = x_map.get(p[2], p[2])

    # Step 3: Merge nearby ports (within t_same in both dims)
    n = len(port_refs)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    for i in range(n):
        for j in range(i + 1, n):
            if abs(port_refs[i][2] - port_refs[j][2]) <= t_same and \
               abs(port_refs[i][3] - port_refs[j][3]) <= t_same:
                union(i, j)
    # Group and average
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    for root, indices in groups.items():
        if len(indices) > 1:
            mx = int(np.mean([port_refs[i][2] for i in indices]))
            my = int(np.mean([port_refs[i][3] for i in indices]))
            for i in indices:
                port_refs[i][2] = mx
                port_refs[i][3] = my

    # Write back to components
    for ci, pi, px, py in port_refs:
        components[ci]["ports"][pi] = (px, py)

def _on_same_wire(jx1, jy1, jx2, jy2, wire_bboxes, margin=5):
    """Check if two junction points lie within the same detected wire segment."""
    if not wire_bboxes:
        return False
    for x1, y1, x2, y2 in wire_bboxes:
        if (x1 - margin <= jx1 <= x2 + margin and y1 - margin <= jy1 <= y2 + margin and
            x1 - margin <= jx2 <= x2 + margin and y1 - margin <= jy2 <= y2 + margin):
            return True
    return False

def _snap_points_to_grid(points, t_h=5, t_v=5):
    """Snap a list of (x, y) points to common grid lines. Returns snapped list."""
    if len(points) <= 1:
        return [(int(x), int(y)) for x, y in points]
    ys = sorted(set(p[1] for p in points))
    y_clusters, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= t_h: cur.append(y)
        else: y_clusters.append(cur); cur = [y]
    y_clusters.append(cur)
    y_map = {}
    for cl in y_clusters:
        med = int(np.median(cl))
        for y in cl: y_map[y] = med
    xs = sorted(set(p[0] for p in points))
    x_clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= t_v: cur.append(x)
        else: x_clusters.append(cur); cur = [x]
    x_clusters.append(cur)
    x_map = {}
    for cl in x_clusters:
        med = int(np.median(cl))
        for x in cl: x_map[x] = med
    return [(x_map.get(x, x), y_map.get(y, y)) for x, y in points]

# ---------------------------------------------------------------------------
# Skeleton-based wire validation
# ---------------------------------------------------------------------------
def _count_skeleton_branches(skeleton, x, y, radius=15):
    """Count distinct wire branches emanating from a junction point.

    Extracts a circular patch, finds skeleton pixels touching the border,
    and groups them via Union-Find (8-connected). Each group = one wire branch.
    """
    h, w = skeleton.shape[:2]
    x1, y1 = max(0, x-radius), max(0, y-radius)
    x2, y2 = min(w, x+radius+1), min(h, y+radius+1)
    if x2 <= x1+4 or y2 <= y1+4:
        return 8
    patch = skeleton[y1:y2, x1:x2]
    ph, pw = patch.shape
    cx, cy = radius, radius  # center in patch coords

    # Find skeleton pixels touching the circular border
    border_pixels = []
    for py in range(ph):
        for px in range(pw):
            if patch[py, px] == 0:
                continue
            dx, dy = px - cx, py - cy
            if dx*dx + dy*dy >= (radius-2)*(radius-2):  # near border
                border_pixels.append((px, py))

    if len(border_pixels) <= 1:
        return max(1, len(border_pixels))

    # Union-Find grouping (8-connected)
    n = len(border_pixels)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i+1, n):
            if abs(border_pixels[i][0]-border_pixels[j][0]) <= 1 and                abs(border_pixels[i][1]-border_pixels[j][1]) <= 1:
                union(i, j)

    groups = set(find(i) for i in range(n))
    return max(1, len(groups))


def _verify_skeleton_path(skeleton, x1, y1, x2, y2, margin=5, min_ratio=0.35):
    """Check if the skeleton supports a wire between two ALIGNED points.

    For horizontal/vertical alignments, samples the straight line between the
    two points and checks what fraction of pixels fall on the skeleton (within
    ±margin perpendicular). Returns True if coverage >= min_ratio.
    For non-aligned points, returns False (must be axis-aligned).
    """
    h, w = skeleton.shape[:2]
    if abs(y1 - y2) < 10 and abs(x1 - x2) > 10:  # horizontal
        y_mid = (y1 + y2) // 2
        x_start, x_end = min(x1, x2), max(x1, x2)
        length = x_end - x_start
        if length < 10:
            return True
        covered = 0
        for x in range(x_start, x_end + 1, max(1, length // 80)):
            found = False
            for dy in range(-margin, margin + 1):
                yy = y_mid + dy
                if 0 <= x < w and 0 <= yy < h and skeleton[yy, x] > 0:
                    found = True
                    break
            if found:
                covered += 1
        return covered / max(1, (length // max(1, length // 80)) + 1) >= min_ratio
    elif abs(x1 - x2) < 10 and abs(y1 - y2) > 10:  # vertical
        x_mid = (x1 + x2) // 2
        y_start, y_end = min(y1, y2), max(y1, y2)
        length = y_end - y_start
        if length < 10:
            return True
        covered = 0
        for y in range(y_start, y_end + 1, max(1, length // 80)):
            found = False
            for dx in range(-margin, margin + 1):
                xx = x_mid + dx
                if 0 <= xx < w and 0 <= y < h and skeleton[y, xx] > 0:
                    found = True
                    break
            if found:
                covered += 1
        return covered / max(1, (length // max(1, length // 80)) + 1) >= min_ratio
    return False  # not aligned


def _verify_skeleton_any(skeleton, x1, y1, x2, y2, margin=6, min_ratio=0.30):
    """Check skeleton support for ANY path (aligned or diagonal).

    Samples points along the straight line between (x1,y1) and (x2,y2),
    checking for skeleton pixels within margin. Works for non-axis-aligned
    junction pairs that the regular _verify_skeleton_path can't handle.
    """
    h, w = skeleton.shape[:2]
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 10:
        return True
    steps = max(20, int(dist / 3))
    covered = 0
    for i in range(steps + 1):
        t = i / steps
        sx = int(x1 + (x2 - x1) * t)
        sy = int(y1 + (y2 - y1) * t)
        found = False
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                px, py = sx + dx, sy + dy
                if 0 <= px < w and 0 <= py < h and skeleton[py, px] > 0:
                    found = True
                    break
            if found:
                break
        if found:
            covered += 1
    return covered / (steps + 1) >= min_ratio


# ---------------------------------------------------------------------------
# Skeleton-based wire tracing
# ---------------------------------------------------------------------------
def _extract_skeleton(gray_img, max_dim=800):
    """Extract single-pixel skeleton from grayscale circuit image.
    Steps: resize → adaptive threshold → morph close → Zhang-Suen thinning.
    Returns binary skeleton image (255=wire, 0=background) at ORIGINAL resolution.
    """
    orig_h, orig_w = gray_img.shape[:2]
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        gray_img = cv2.resize(gray_img, (int(orig_w*scale), int(orig_h*scale)))
    # Adaptive threshold for varying illumination / pen pressure
    bin_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 4)
    # Bridge small gaps (≤3px)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Zhang-Suen thinning
    skel = (closed // 255).astype(np.uint8)
    prev = np.zeros_like(skel)
    for _ in range(80):  # safety limit
        if np.array_equal(skel, prev):
            break
        prev = skel.copy()
        for step in (0, 1):
            markers = np.zeros_like(skel)
            for y in range(1, skel.shape[0] - 1):
                for x in range(1, skel.shape[1] - 1):
                    if skel[y, x] != 1:
                        continue
                    p2 = skel[y-1, x];   p3 = skel[y-1, x+1]
                    p4 = skel[y,   x+1]; p5 = skel[y+1, x+1]
                    p6 = skel[y+1, x];   p7 = skel[y+1, x-1]
                    p8 = skel[y,   x-1]; p9 = skel[y-1, x-1]
                    neigh = [p2, p3, p4, p5, p6, p7, p8, p9]
                    A = sum(1 for i in range(8)
                            if neigh[i] == 0 and neigh[(i+1) % 8] == 1)
                    B = sum(neigh)
                    if step == 0:
                        if 2 <= B <= 6 and A == 1 and p2*p4*p6 == 0 and p4*p6*p8 == 0:
                            markers[y, x] = 1
                    else:
                        if 2 <= B <= 6 and A == 1 and p2*p4*p8 == 0 and p2*p6*p8 == 0:
                            markers[y, x] = 1
            skel[markers == 1] = 0
    skeleton = skel * 255
    # Resize back to original dimensions if downscaled
    if scale < 1.0:
        skeleton = cv2.resize(skeleton, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return skeleton

def _snap_ports_to_skeleton(components, skeleton, search_radius=15):
    """改进1: Snap each port to the nearest skeleton endpoint.

    For each component port, search the skeleton image in a radius.
    If a skeleton endpoint (exactly 1 skeleton neighbor) is found nearby,
    replace the port coordinate with the endpoint position.
    If an internal skeleton point (2+ neighbors) is found, trace outward
    along the skeleton to the nearest endpoint.

    This fixes the ~10-20px offset between hardcoded PORT_POSITIONS and
    actual wire attachment points in hand-drawn circuits.
    """
    if skeleton is None:
        return 0
    h, w = skeleton.shape

    def _count_neighbors(sk, y, x):
        n = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] > 0:
                    n += 1
        return n

    def _trace_to_endpoint(sk, y, x, max_steps=30):
        """BFS along skeleton outward from (y,x) to nearest endpoint."""
        visited = {(y, x)}
        q = [(y, x)]
        for _ in range(max_steps):
            if not q:
                break
            cy, cx = q.pop(0)
            nn = _count_neighbors(sk, cy, cx)
            if nn == 1 and (cy != y or cx != x):
                return (cy, cx)  # found endpoint
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] > 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        q.append((ny, nx))
        return None

    snapped = 0
    for c in components:
        for pi, (px, py) in enumerate(c["ports"]):
            # Search for nearest skeleton pixel within radius
            best_pt = None
            best_dist = search_radius ** 2
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    sy, sx = py + dy, px + dx
                    if 0 <= sy < h and 0 <= sx < w and skeleton[sy, sx] > 0:
                        d2 = dy * dy + dx * dx
                        if d2 < best_dist:
                            best_dist = d2
                            best_pt = (sy, sx)
            if best_pt is None:
                continue
            sy, sx = best_pt
            nn = _count_neighbors(skeleton, sy, sx)
            if nn == 1:
                # Directly use the endpoint
                c["ports"][pi] = (sx, sy)
                snapped += 1
            elif nn >= 2:
                # Internal skeleton point: trace to nearest endpoint
                ep = _trace_to_endpoint(skeleton, sy, sx)
                if ep is not None:
                    c["ports"][pi] = (ep[1], ep[0])  # (cx, cy) = (x, y)
                    snapped += 1
    return snapped

def _trace_wire_connections(skeleton, components, junctions, img_h, img_w,
                             search_radius=10, gap_bridge=8):
    """BFS along skeleton from each port to discover wire-verified connections.

    Returns list of (comp_idx, port_idx, target_x, target_y) following the same
    format as p2j_connections, ready to merge into existing logic.
    """
    # Build fast lookup: skeleton pixel set
    skel_set = set()
    ys, xs = np.where(skeleton > 0)
    skel_coords = list(zip(xs, ys))
    for xy in skel_coords:
        skel_set.add(xy)

    # Build junction lookup (rounded to int)
    junc_set = set((int(jx), int(jy)) for jx, jy in junctions)

    # Build port map: (px, py) → (comp_idx, port_idx)
    port_map = {}
    for ci, c in enumerate(components):
        for pi, (px, py) in enumerate(c["ports"]):
            key = (int(px), int(py))
            # If multiple ports share same position, store all
            if key not in port_map:
                port_map[key] = []
            port_map[key].append((ci, pi))

    connections = []  # [(comp_idx, port_idx, tx, ty)]
    visited_global = set()  # avoid re-tracing already walked paths

    for ci, c in enumerate(components):
        for pi, (px, py) in enumerate(c["ports"]):
            px_i, py_i = int(px), int(py)

            # Find nearest skeleton pixel to port
            best_start, best_d = None, search_radius
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    sx, sy = px_i + dx, py_i + dy
                    if 0 <= sx < img_w and 0 <= sy < img_h and (sx, sy) in skel_set:
                        d = abs(dx) + abs(dy)
                        if d < best_d:
                            best_d = d
                            best_start = (sx, sy)
            if best_start is None:
                continue  # port not on skeleton → fall back to P2J

            # BFS from best_start
            queue = [best_start]
            port_visited = {best_start}
            found_target = False

            while queue and not found_target:
                cx, cy = queue.pop(0)  # BFS (FIFO)

                # Check if reached a junction
                if (cx, cy) in junc_set and (cx != px_i or cy != py_i):
                    connections.append((ci, pi, cx, cy))
                    found_target = True
                    break

                # Check if reached another port
                if (cx, cy) in port_map:
                    for oc, op in port_map[(cx, cy)]:
                        if oc != ci:
                            # Route to other component's port
                            connections.append((ci, pi, cx, cy))
                            found_target = True
                            break

                if found_target:
                    break

                # Explore 8 neighbors
                for nx, ny in [(cx-1,cy-1),(cx,cy-1),(cx+1,cy-1),
                               (cx-1,cy),           (cx+1,cy),
                               (cx-1,cy+1),(cx,cy+1),(cx+1,cy+1)]:
                    if 0 <= nx < img_w and 0 <= ny < img_h:
                        if (nx, ny) not in skel_set:
                            continue
                        if (nx, ny) in port_visited:
                            continue
                        port_visited.add((nx, ny))
                        queue.append((nx, ny))

            # Gap bridge: if BFS dead-ended, search nearby skeleton within gap_bridge
            if not found_target and port_visited:
                # Find all dead-end pixels (1 neighbor in visited set)
                for vx, vy in list(port_visited):
                    # Look for unvisited skeleton within gap_bridge
                    for dx in range(-gap_bridge, gap_bridge + 1):
                        for dy in range(-gap_bridge, gap_bridge + 1):
                            gx, gy = vx + dx, vy + dy
                            if 0 <= gx < img_w and 0 <= gy < img_h:
                                if (gx, gy) in skel_set and (gx, gy) not in port_visited:
                                    # Bridge gap: continue BFS from this point
                                    port_visited.add((gx, gy))
                                    queue.append((gx, gy))

            # Add visited pixels to global set
            visited_global |= port_visited

    return connections

# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def _clean_ocr(text):
    text = text.strip().replace(" ", "").replace("\n", "")
    if len(text) < 1:
        return ""
    valid = sum(1 for c in text if c.isalnum() or c in ".kKmMΩμunpFV AHz-+")
    if valid < len(text) * 0.5:
        return ""
    def _expand_short(val):
        # 3V3 → 3.3V, 4R7 → 4.7Ω, etc.
        m = re.match(r"^(\d+)([Rr])(\d+)(.*)$", val)
        if m: return f"{m.group(1)}.{m.group(3)}Ω"
        m = re.match(r"^(\d+)([kKmM])(\d+)([Ω]?)$", val)
        if m: return f"{m.group(1)}.{m.group(3)}{m.group(2)}{m.group(4) or 'Ω'}"
        m = re.match(r"^(\d+)([Vv])(\d+)$", val)
        if m: return f"{m.group(1)}.{m.group(3)}V"
        m = re.match(r"^(\d+)([Aa])(\d+)$", val)
        if m: return f"{m.group(1)}.{m.group(3)}A"
        m = re.match(r"^(\d{2})[Vv]$", val)
        if m and int(m.group(1)) <= 50:
            return f"{int(m.group(1))/10:.1f}V"
        return val
    val = _expand_short(text)
    # Fix double dots: "0..1" → "0.1"
    val = val.replace("..", ".")
    # Fix bare 2-digit numbers likely meant as X.Y (3V3 format lost)
    # "33" alone near a voltage source → likely "3.3V"
    return val


def _ocr_variants(text):
    """Generate alternate interpretations of OCR text for common CRNN errors.
    Handwritten: k↔H, Ω↔V, μF↔V, missing decimal point.
    Returns list of (value, unit_hint) tuples.
    """
    variants = [(text, "")]
    # If ends with "H" or "z" after digits, could be "k" (handwritten k looks like H or z)
    m = re.match(r"^([\d.]+)[Hz]$", text)
    if m:
        variants.append((f"{m.group(1)}kΩ", "Ω"))
        variants.append((f"{m.group(1)}MΩ", "Ω"))
        variants.append((f"{m.group(1)}Ω", "Ω"))
    # If ends with "V" alone after digits, could be "Ω" (resistor) or "μF" (capacitor)
    m = re.match(r"^([\d.]+)V$", text)
    if m:
        variants.append((f"{m.group(1)}Ω", "Ω"))
        variants.append((f"{m.group(1)}μF", "F"))
        variants.append((f"{m.group(1)}F", "F"))
    # Handle missing decimal: "22k" could be "2.2k"
    m = re.match(r"^(\d{2})([kKmMμ])", text)
    if m:
        n = m.group(1)
        variants.append((f"{int(n)/10:.1f}{m.group(2)}", ""))
    # "k" without Ω → add Ω
    if re.search(r"[kKmM]$", text) and not text.endswith("Ω"):
        variants.append((text + "Ω", "Ω"))
    return variants


def _try_match(value_text, comp_type):
    """Check if value_text matches a component's pattern. Returns (True/False, corrected_value)."""
    if comp_type == "GND":
        return False, ""
    pattern, def_unit = VALUE_PATTERNS.get(comp_type, (None, ""))
    if pattern is None:
        return True, value_text
    anti = ANTI_PATTERNS.get(comp_type, [])
    if any(re.search(ap, value_text) for ap in anti):
        return False, ""
    # Try matching with/without unit suffix
    if not re.search(r"[ΩFHVAmHμ]$", value_text):
        v_test = value_text + def_unit
    else:
        v_test = value_text
    # Context-aware: bare numbers like "33" near V-DC → try "3.3V"
    if comp_type in ("V-DC", "V-AC") and re.match(r"^\d{2}$", value_text):
        n = int(value_text)
        if 10 <= n <= 50:
            alt = f"{n/10:.1f}V"
            if re.match(pattern, alt):
                return True, alt
    if re.match(pattern, v_test):
        if not re.search(r"[ΩFHVAmHμ]$", value_text):
            return True, value_text + def_unit
        return True, value_text
    return False, ""

# ---------------------------------------------------------------------------
# Bbox crossing check
# ---------------------------------------------------------------------------
def crosses(line_p1, line_p2, components, own_idx):
    """Check if line segment crosses any component bbox (excluding own_idx)."""
    lx1, ly1 = line_p1; lx2, ly2 = line_p2
    for idx, c in enumerate(components):
        if idx == own_idx:
            continue
        x1, y1, x2, y2 = c["xyxy"]
        # Expand margin to prevent routes passing through component edges
        x1, y1 = x1 - 8, y1 - 8
        x2, y2 = x2 + 8, y2 + 8
        # Check if segment intersects rectangle
        if _line_rect_intersect(lx1, ly1, lx2, ly2, x1, y1, x2, y2):
            return True
    return False

def _line_rect_intersect(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
    """Cohen-Sutherland line clipping: returns True if line intersects rectangle."""
    # Quick rejection: both endpoints on same side
    if max(lx1, lx2) < rx1 or min(lx1, lx2) > rx2:
        return False
    if max(ly1, ly2) < ry1 or min(ly1, ly2) > ry2:
        return False
    # Check if either endpoint inside
    if rx1 <= lx1 <= rx2 and ry1 <= ly1 <= ry2:
        return True
    if rx1 <= lx2 <= rx2 and ry1 <= ly2 <= ry2:
        return True
    # Check intersection with each rect edge
    edges = [((rx1, ry1), (rx2, ry1)), ((rx2, ry1), (rx2, ry2)),
             ((rx2, ry2), (rx1, ry2)), ((rx1, ry2), (rx1, ry1))]
    for (ex1, ey1), (ex2, ey2) in edges:
        if _segments_intersect(lx1, ly1, lx2, ly2, ex1, ey1, ex2, ey2):
            return True
    return False

def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = ccw(x1, y1, x2, y2, x3, y3)
    d2 = ccw(x1, y1, x2, y2, x4, y4)
    d3 = ccw(x3, y3, x4, y4, x1, y1)
    d4 = ccw(x3, y3, x4, y4, x2, y2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2):
        return True
    if d2 == 0 and min(x1, x2) <= x4 <= max(x1, x2) and min(y1, y2) <= y4 <= max(y1, y2):
        return True
    if d3 == 0 and min(x3, x4) <= x1 <= max(x3, x4) and min(y3, y4) <= y1 <= max(y3, y4):
        return True
    if d4 == 0 and min(x3, x4) <= x2 <= max(x3, x4) and min(y3, y4) <= y2 <= max(y3, y4):
        return True
    return False

# ---------------------------------------------------------------------------
# Manhattan routing
# ---------------------------------------------------------------------------
def route(p1, p2, components, own_idx, margin=20):
    """Manhattan route from p1 to p2 avoiding component bboxes. Returns list of points."""
    x1, y1 = p1; x2, y2 = p2
    # Try horizontal-first
    mid = (x2, y1)
    if not crosses(p1, mid, components, own_idx) and not crosses(mid, p2, components, own_idx):
        return [p1, mid, p2]
    # Try vertical-first
    mid = (x1, y2)
    if not crosses(p1, mid, components, own_idx) and not crosses(mid, p2, components, own_idx):
        return [p1, mid, p2]
    # Offset horizontal-first
    for dx in [margin, -margin, margin*2, -margin*2, margin*3, -margin*3]:
        mid = (x2 + dx, y1)
        if not crosses(p1, mid, components, own_idx) and not crosses(mid, p2, components, own_idx):
            return [p1, mid, p2]
    # Offset vertical-first
    for dy in [margin, -margin, margin*2, -margin*2, margin*3, -margin*3]:
        mid = (x1, y2 + dy)
        if not crosses(p1, mid, components, own_idx) and not crosses(mid, p2, components, own_idx):
            return [p1, mid, p2]
    # Midpoint offset routes
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    for dx, dy in [(margin,0),(-margin,0),(0,margin),(0,-margin),
                   (margin*2,0),(-margin*2,0),(0,margin*2),(0,-margin*2)]:
        mid = (mx + dx, my + dy)
        if not crosses(p1, mid, components, own_idx) and not crosses(mid, p2, components, own_idx):
            return [p1, mid, p2]
    return []  # no route found

# ---------------------------------------------------------------------------
# Same-position detection (model duplicates)
# ---------------------------------------------------------------------------
def same_position(c1, c2, tol=10):
    xa1, ya1, xa2, ya2 = c1["xyxy"]
    xb1, yb1, xb2, yb2 = c2["xyxy"]
    return abs(xa1 - xb1) < tol and abs(ya1 - yb1) < tol and abs(xa2 - xb2) < tol and abs(ya2 - yb2) < tol

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_image(img_path, config=None):
    if config is None:
        config = dict(DEFAULT_CONFIG)
    else:
        cfg = dict(DEFAULT_CONFIG); cfg.update(config); config = cfg

    img_name = os.path.basename(img_path)
    print(f"\n{'='*60}")
    print(f"  {img_name}")
    print(f"{'='*60}")

    cgh_model = _load_models()
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Cannot read {img_path}")
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image-size normalization: scale distance thresholds by diagonal vs reference
    DIAG_REF = 2000.0
    img_diag = math.hypot(h, w)
    im_scale = img_diag / DIAG_REF
    pjr = int(PORT_JUNCTION_RADIUS * im_scale)
    pjf = int(PORT_JUNCTION_FALLBACK * im_scale)
    jj_align = max(8, int(JJ_ALIGN_PX * im_scale))
    jj_max = int(JJ_MAX_ALIGNED_DIST * im_scale)
    p2p_max = int(100 * im_scale)
    p2p_line = int(200 * im_scale)
    agg_p2p = int(60 * im_scale)
    skel_search = max(10, int(20 * im_scale))
    skel_gap = max(8, int(12 * im_scale))
    grid_snap = max(5, int(GRID_SNAP_TOL * im_scale))
    uf_merge_t = max(5, int(UF_MERGE_TOL * im_scale))

    # ---- Step 1: Detect components + junctions + text from CGHD ----
    results = cgh_model(img_path)[0]
    components = []
    junctions_raw = []
    text_bboxes = []  # CGHD's own text detections for OCR

    for box in (results.boxes or []):
        name = cgh_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        if name in ("junction", "terminal") and conf >= JUNCTION_CONF:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            junctions_raw.append(((x1 + x2) // 2, (y1 + y2) // 2))
            continue

        # Collect CGHD text detections for OCR (not added to components)
        if name == "text" and conf >= CGH_CONF_THRESH:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            text_bboxes.append((x1, y1, x2, y2, conf))
            continue

        if name in CGH_SKIP or conf < CGH_CONF_THRESH:
            continue

        hcd_name = CGH_NAME_MAP.get(name, name)
        display = CGH_DISPLAY.get(name, name)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bw, bh = x2 - x1, y2 - y1

        # Ports: use Sobel edge detection for IC/multi-pin types
        ports = []
        port_key = CGH_TO_PORT_KEY.get(name, name)
        IC_PORT_TYPES = {"IC", "NE555", "Voltage-Regulator", "Op-Amp",
                         "Schmitt-Trigger", "Optocoupler"}
        IC_MAX_PORTS = {"IC": 5, "NE555": 8, "Voltage-Regulator": 4,
                        "Op-Amp": 5, "Schmitt-Trigger": 5, "Optocoupler": 4}
        need_rotate = False
        if port_key in IC_PORT_TYPES:
            # Detect actual pin attachment points from image edges
            max_p = IC_MAX_PORTS.get(port_key, 4)
            detected = _detect_ic_ports(img_path, x1, y1, x2, y2, max_ports=max_p)
            if detected:
                for rx, ry in detected:
                    px = int(x1 + rx * bw)
                    py = int(y1 + ry * bh)
                    ports.append((px, py))
        if not ports and port_key in PORT_POSITIONS and len(PORT_POSITIONS[port_key]) >= 1:
            plist = PORT_POSITIONS[port_key]
            # Types that should NEVER rotate (GND always points down)
            no_rotate_types = {"GND", "gnd", "vss"}
            if len(plist) == 2 and name not in no_rotate_types and port_key not in no_rotate_types:
                need_rotate = _detect_orientation(img_path, x1, y1, x2, y2, plist, name, use_sobel=config["use_sobel"])
            else:
                need_rotate = False
            for rx, ry in plist:
                if need_rotate:
                    sx, sy = 1 - ry, rx
                else:
                    sx, sy = rx, ry
                px = int(x1 + sx * bw)
                py = int(y1 + sy * bh)
                ports.append((px, py))

        # LED label swap: when LED rotates, anode (+) and cathode (-) swap positions
        label_swap = (need_rotate and "light_emitting" in name)

        components.append(dict(
            idx=len(components), name=hcd_name, display=display, raw_name=name,
            xyxy=(x1, y1, x2, y2), cx=(x1 + x2) // 2, cy=(y1 + y2) // 2,
            conf=conf, value="", ports=ports, designator="",
            label_swap=label_swap,
        ))

    print(f"  Components: {len(components)}")

    # ---- NMS dedup: remove overlapping same-family detections ----
    # For families like {diode, diode.light_emitting, diode.zener}, keep the higher-conf one
    nms_removed = set()
    for family in NMS_FAMILIES:
        for i in range(len(components)):
            if i in nms_removed:
                continue
            if components[i]["raw_name"] not in family:
                continue
            for j in range(i + 1, len(components)):
                if j in nms_removed:
                    continue
                if components[j]["raw_name"] not in family:
                    continue
                # Compute IoU
                xa1, ya1, xa2, ya2 = components[i]["xyxy"]
                xb1, yb1, xb2, yb2 = components[j]["xyxy"]
                ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
                ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (xa2 - xa1) * (ya2 - ya1)
                area_b = (xb2 - xb1) * (yb2 - yb1)
                iou = inter / (area_a + area_b - inter)
                if iou > NMS_IOU_THRESH:
                    # Keep higher conf, prefer more specific class (LED over Diode)
                    ci, cj = components[i], components[j]
                    specific_order = {"diode.light_emitting": 3, "diode.zener": 3,
                                     "diode.thyrector": 2, "diode": 1,
                                     "capacitor.polarized": 3, "capacitor.adjustable": 2,
                                     "capacitor.unpolarized": 1,
                                     "integrated_circuit.ne555": 3,
                                     "integrated_circuit.voltage_regulator": 3,
                                     "integrated_circuit": 1}
                    score_i = specific_order.get(ci["raw_name"], 0)
                    score_j = specific_order.get(cj["raw_name"], 0)
                    if abs(ci["conf"] - cj["conf"]) < 0.05 or ci["conf"] > 0.35 and cj["conf"] > 0.35:
                        # Close conf OR both reasonable: prefer more specific class
                        keep_i = score_i >= score_j
                    else:
                        keep_i = ci["conf"] >= cj["conf"]
                    if keep_i:
                        nms_removed.add(j)
                    else:
                        nms_removed.add(i)
                        break  # i removed, stop comparing j

    if nms_removed:
        components = [c for idx, c in enumerate(components) if idx not in nms_removed]
        # Re-index
        for i, c in enumerate(components):
            c["idx"] = i
        print(f"  After NMS dedup: {len(components)} (removed {len(nms_removed)} overlapping)")

    # ---- P0: GND context power type correction ----
    has_gnd = any(c["name"] == "GND" for c in components)
    has_resistive_load = any(c["name"] in ("Resistor", "LED", "Diode", "Capacitor",
                                           "Inductor", "Op-Amp", "IC")
                           for c in components)
    for c in components:
        if c["name"] == "V-AC" and has_gnd and has_resistive_load:
            # GND + resistive load → almost certainly DC, reclassify unconditionally
            c["name"] = "V-DC"
            c["raw_name"] = "voltage.dc"
            print(f"    Auto-correct: {c['designator']} V-AC→V-DC (GND+负载)")
        elif c["name"] == "V-DC" and not has_gnd and c["conf"] < 0.50:
            c["name"] = "V-AC"
            c["raw_name"] = "voltage.ac"
            print(f"    Auto-correct: {c['designator']} V-DC→V-AC (no GND)")

    # ---- Step 2: OCR text values (using CGHD text detections) ----
    def inside_gnd_only(tx1, ty1, tx2, ty2):
        for c in components:
            if c["name"] == "GND":
                cx1, cy1, cx2, cy2 = c["xyxy"]
                ox1, oy1 = max(tx1, cx1), max(ty1, cy1)
                ox2, oy2 = min(tx2, cx2), min(ty2, cy2)
                area_t = (tx2 - tx1) * (ty2 - ty1)
                if ox2 > ox1 and oy2 > oy1:
                    if (ox2 - ox1) * (oy2 - oy1) > area_t * 0.5:
                        return True
        return False

    text_values = []
    for x1, y1, x2, y2, conf in text_bboxes:
        if inside_gnd_only(x1, y1, x2, y2):
            continue
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        raw = crnn_predict(_OCR_MODEL, crop, _OCR_CHARS, _OCR_ITOC, _OCR_IMG_H)
        raw = _clean_ocr(raw)
        corrections = _load_corrections()
        raw = _apply_corrections(raw, corrections)
        if raw:
            text_values.append(dict(
                text=raw, cx=(x1 + x2) // 2, cy=(y1 + y2) // 2,
                xyxy=(x1, y1, x2, y2), conf=conf
            ))

    print(f"  OCR texts: {len(text_values)}")

    # ---- Step 3: Match values (proximity-first with OCR variant correction) ----
    # For each text value, try all component types. Pick best (comp, variant) by:
    # 1) Exact pattern match preferred, 2) Closer distance preferred.
    # If multiple components want the same text, closest one wins.
    tv_assignments = {}  # tv_idx -> (comp_idx, matched_value, distance, is_exact)
    for ti, tv in enumerate(text_values):
        variants = _ocr_variants(tv["text"])
        best_for_tv = None  # (ci, value, dist, is_exact)
        for ci, c in enumerate(components):
            if c["name"] == "GND":
                continue
            dist = abs(tv["cx"] - c["cx"]) + abs(tv["cy"] - c["cy"])
            if dist > 250:
                continue
            for v_text, v_hint in variants:
                ok, corrected = _try_match(v_text, c["name"])
                if ok:
                    # Exact (non-variant) match gets bonus
                    is_exact = (v_text == tv["text"])
                    score = dist - (40 if is_exact else 0)  # prefer exact match
                    if best_for_tv is None or score < best_for_tv[2] - (20 if best_for_tv[3] else 0):
                        best_for_tv = (ci, corrected or v_text, score, is_exact)
        if best_for_tv:
            tv_assignments[ti] = best_for_tv

    # Assign: each component gets its closest matching text
    for c in components:
        c["value"] = ""
    assigned_tv = set()
    # Assign by distance priority (closest first)
    sorted_assignments = sorted(tv_assignments.items(), key=lambda x: x[1][2])
    for ti, (ci, val, score, is_exact) in sorted_assignments:
        if ti in assigned_tv:
            continue
        if components[ci]["value"]:
            continue  # component already has a value
        components[ci]["value"] = val
        assigned_tv.add(ti)
    # Second pass: fill unassigned components from remaining text (with variants)
    for c in components:
        if c["value"] or c["name"] == "GND":
            continue
        best_text, best_ti, best_dist = "", -1, 99999
        for ti, tv in enumerate(text_values):
            if ti in assigned_tv:
                continue
            dist = abs(tv["cx"] - c["cx"]) + abs(tv["cy"] - c["cy"])
            if dist > 250:
                continue
            # Try original text and variants
            for v_text, v_hint in _ocr_variants(tv["text"]):
                ok, corrected = _try_match(v_text, c["name"])
                if ok and dist < best_dist:
                    best_text = corrected or v_text
                    best_dist = dist
                    best_ti = ti
        if best_text:
            c["value"] = best_text
            assigned_tv.add(best_ti)

    # ---- Step 3b: High-confidence OCR unit correction ----
    # When component conf > 0.6 and value unit doesn't match component type, force-correct
    HC_CONF = 0.60
    UNIT_MAP = {
        "Resistor": ["Ω", "kΩ", "MΩ"],
        "Capacitor": ["F", "μF", "mF", "nF", "pF"],
        "Inductor": ["H", "mH", "μH"],
        "V-DC": ["V", "mV"],
        "V-AC": ["V", "mV"],
        "I-DC": ["A", "mA"],
        "I-AC": ["A", "mA"],
    }
    for c in components:
        if c["conf"] < HC_CONF or not c["value"] or c["name"] == "GND":
            continue
        valid_units = UNIT_MAP.get(c["name"], [])
        if not valid_units:
            continue
        val = c["value"]
        # Check if value ends with a valid unit for this component type
        has_valid_unit = any(val.endswith(u) for u in valid_units)
        if not has_valid_unit:
            # Find the closest OCR text to see what it originally said
            best_tv = None
            best_dist = 99999
            for tv in text_values:
                d = abs(tv["cx"] - c["cx"]) + abs(tv["cy"] - c["cy"])
                if d < best_dist:
                    best_dist = d
                    best_tv = tv
            if best_tv:
                orig = best_tv["text"]
                # Extract numeric part
                num_match = re.match(r"^([\d.]+)", orig)
                if num_match:
                    num = num_match.group(1)
                    # Use first valid unit as default for this component type
                    def_unit = valid_units[0]
                    # Try to determine magnitude from context
                    corrected = num + def_unit
                    c["value"] = corrected

    # ---- Step 3c: Tesseract re-read for IC-type components ----
    IC_TYPES = {"IC", "NE555", "Voltage-Regulator", "Optocoupler", "Op-Amp"}
    for c in components:
        if c["name"] not in IC_TYPES:
            continue
        if c["value"] and re.search(r"[A-Za-z]", c["value"]):
            continue  # already has a chip-like name
        # Collect ALL text regions near this IC (within bbox + margin)
        ic_x1, ic_y1, ic_x2, ic_y2 = c["xyxy"]
        nearby_texts = []
        for tv in text_values:
            tx, ty = tv["cx"], tv["cy"]
            if ic_x1 - 20 <= tx <= ic_x2 + 20 and ic_y1 - 20 <= ty <= ic_y2 + 20:
                nearby_texts.append(tv)
        if not nearby_texts:
            # Fall back to closest
            best_tv, best_dist = None, 99999
            for tv in text_values:
                d = abs(tv["cx"] - c["cx"]) + abs(tv["cy"] - c["cy"])
                if d < best_dist:
                    best_dist = d
                    best_tv = tv
            if best_tv:
                nearby_texts = [best_tv]
        # Try Tesseract on each nearby text, pick best (longest alphanumeric)
        best_tsr = ""
        for tv in nearby_texts:
            x1, y1, x2, y2 = tv.get("xyxy", (0, 0, 0, 0))
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tsr = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-./").strip()
            if tsr and len(tsr) >= 3 and re.search(r"[A-Za-z]", tsr):
                if len(tsr) > len(best_tsr):
                    best_tsr = tsr
        if best_tsr:
            c["value"] = best_tsr
            print(f"    Tesseract: {c['designator']} = {best_tsr}")

    # ---- Step 4: Assign designators ----
    counts = defaultdict(int)
    for c in components:
        if c["name"] == "GND":
            c["designator"] = "GND"
        else:
            prefix = DESIG.get(c["name"], "X")
            counts[prefix] += 1
            c["designator"] = f"{prefix}{counts[prefix]}"

    # Print component summary
    for c in components:
        val_str = f"= {c['value']}" if c['value'] else ""
        print(f"    {c['designator']:6s} {c['name']:15s} {val_str}")

    # ---- Step 4b (CCL): Connected Component Analysis wiring ----
    # When enabled, replaces all junction/P2J/JJ/skeleton steps with a
    # simpler approach: mask components → dilate wires → CCL → assign ports.
    if config["use_ccl"]:
        print("  Wiring: CCL (Connected Component Analysis)")
        # 1. Binarize: adaptive threshold handles uneven lighting in hand-drawn scans
        wire_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 6)
        # White-out component INTERIORS only (inset), keep edges visible for ports
        inset = max(3, int(5 * im_scale))
        for c in components:
            x1, y1, x2, y2 = c["xyxy"]
            mx1 = max(0, x1 + inset); my1 = max(0, y1 + inset)
            mx2 = min(w, x2 - inset); my2 = min(h, y2 - inset)
            if mx2 > mx1 and my2 > my1:
                wire_mask[my1:my2, mx1:mx2] = 0  # black

        # 2. Dilate to bridge hand-drawn gaps
        k = max(2, int(3 * im_scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wire_mask = cv2.dilate(wire_mask, kernel, iterations=1)

        # 3. Connected component analysis
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            wire_mask, connectivity=8)
        print(f"    CCL: {n_labels - 1} connected regions found")

        # 4. Assign each port to its closest connected component label
        port_labels = {}  # (ci, pi) -> label
        for ci, c in enumerate(components):
            for pi, (px, py) in enumerate(c["ports"]):
                px_c = max(0, min(w - 1, px))
                py_c = max(0, min(h - 1, py))
                # Sample labels in a small radial search around port
                r = max(3, int(6 * im_scale))
                found_labels = []
                for dy in range(-r, r + 1, 2):
                    for dx in range(-r, r + 1, 2):
                        sy, sx = py_c + dy, px_c + dx
                        if 0 <= sy < h and 0 <= sx < w:
                            lbl = labels[sy, sx]
                            if lbl > 0:
                                found_labels.append(lbl)
                if found_labels:
                    # Use most common label
                    lbl = max(set(found_labels), key=found_labels.count)
                    port_labels[(ci, pi)] = lbl

        # 5. Build p2j_connections: ports sharing the same label → same net
        #    Use virtual junction points at centroid of each region
        p2j_connections = []
        label_groups = defaultdict(list)
        for (ci, pi), lbl in port_labels.items():
            label_groups[lbl].append((ci, pi))

        for lbl, port_list in label_groups.items():
            if len(port_list) >= 2:
                cx = int(centroids[lbl][0])
                cy = int(centroids[lbl][1])
                for ci, pi in port_list:
                    p2j_connections.append((ci, pi, cx, cy))

        jj_connections = []
        junctions = [(int(centroids[l][0]), int(centroids[l][1]))
                     for l in range(1, n_labels) if l in label_groups
                     and len(label_groups[l]) >= 2]
        routes = []
        print(f"    CCL: {len(p2j_connections)} port connections, {len(junctions)} nets")
        # Skip to Step 9 (connection graph building)
        # We need to set these for the code below
        jp_map = defaultdict(list)
        for ci, pi, jx, jy in p2j_connections:
            jp_map[(jx, jy)].append((ci, pi))

    # ---- Step 5: Junction processing (skipped for CCL) ----
    if not config["use_ccl"]:
        if junctions_raw:
            junctions_snapped = snap(junctions_raw, grid_snap)
            merge_map = uf_merge(junctions_snapped, uf_merge_t)
            junctions = sorted(set(merge_map.values()), key=lambda p: (p[1], p[0]))
        else:
            junctions = []
        # Junction numbering
        jid_map = {j: f"J{i+1}" for i, j in enumerate(junctions)}
        print(f"  Junctions: {len(junctions)} (raw: {len(junctions_raw)})")

    # CI2N wire model removed — insufficient hand-drawn circuit coverage
    wire_bboxes = []

    # ---- Step 5c: Port grid snapping (per 预处理规则) ----
    # Snap port coordinates to common horizontal/vertical lines
    _snap_ports_to_grid(components)

    # ---- Step 5d: Junction grid snapping ----
    # Also snap junctions to common grid lines (t=15 for hand-drawn variance)
    if junctions:
        jxys = [(jx, jy) for jx, jy in junctions]
        jxys_snapped = _snap_points_to_grid(jxys, t_h=8, t_v=8)
        junctions = list(set(jxys_snapped))
        junctions.sort(key=lambda p: (p[1], p[0]))

    # ---- Step 5e: Skeleton-based wire validation ----
    if config["use_skeleton"]:
        try:
            skeleton = _extract_skeleton(gray)
            # Pre-compute junction branch counts from skeleton
            _junc_branch_count = {}
            for jx, jy in junctions:
                _junc_branch_count[(jx, jy)] = _count_skeleton_branches(skeleton, jx, jy)
        except Exception:
            skeleton = None
            _junc_branch_count = {}
    else:
        skeleton = None
        _junc_branch_count = {}

    # ---- 改进1: Snap ports to skeleton endpoints ----
    if config["use_skeleton"] and skeleton is not None:
        n_snapped = _snap_ports_to_skeleton(components, skeleton,
                                            search_radius=max(10, int(15 * im_scale)))
        if n_snapped:
            print(f"  Skeleton port snap: {n_snapped} ports adjusted")

    # ---- Step 6: Port → Junction connections ----
    if not config["use_ccl"]:
        p2j_connections = []  # [(comp_idx, port_idx, jx, jy)]
        for ci, c in enumerate(components):
            for pi, (px, py) in enumerate(c["ports"]):
                best_j, best_d = None, pjr
                for jx, jy in junctions:
                    d = math.hypot(px - jx, py - jy)
                    if d >= pjr:
                        continue
                    # 改进2: skeleton-verified priority lock
                    skel_locked = False
                    if config["use_skeleton"] and skeleton is not None:
                        if _verify_skeleton_path(skeleton, px, py, jx, jy,
                                                 margin=3, min_ratio=0.50):
                            if not crosses((px, py), (jx, jy), components, ci):
                                best_j = (jx, jy)
                                skel_locked = True
                                break  # skeleton verified → lock immediately
                    if skel_locked:
                        break
                    # Standard distance-based matching (fallback when no skeleton lock)
                    if d < best_d:
                        if not crosses((px, py), (jx, jy), components, ci):
                            best_d = d
                            best_j = (jx, jy)
                if best_j:
                    p2j_connections.append((ci, pi, best_j[0], best_j[1]))

        # Cleanup: remove same-component shorts (both ports of 2-port comp -> same junction)
        # For 3+ port components: only remove if ALL ports go to same junction
        p2j_by_comp_junc = defaultdict(list)
        for idx, (ci, pi, jx, jy) in enumerate(p2j_connections):
            p2j_by_comp_junc[(ci, jx, jy)].append((idx, pi, math.hypot(
                components[ci]["ports"][pi][0] - jx,
                components[ci]["ports"][pi][1] - jy)))
        remove_idxs = set()
        for (ci, jx, jy), entries in p2j_by_comp_junc.items():
            n_ports = len(components[ci]["ports"])
            if n_ports == 2 and len(entries) >= 2:
                entries.sort(key=lambda x: x[2])
                for idx, pi, d in entries[1:]:
                    remove_idxs.add(idx)
            elif n_ports >= 3 and len(entries) == n_ports:
                # All ports of multi-port component go to same junction - remove 2 farthest
                entries.sort(key=lambda x: x[2])
                for idx, pi, d in entries[2:]:
                    remove_idxs.add(idx)
        if remove_idxs:
            p2j_connections = [c for i, c in enumerate(p2j_connections) if i not in remove_idxs]
            print(f"  Port->Junction: {len(p2j_connections)} (removed {len(remove_idxs)} same-comp shorts)")
        else:
            print(f"  Port->Junction: {len(p2j_connections)}")

        # ---- Step 6a-skel: Skeleton fallback for truly isolated ports ----
        # Only extends P2J radius — skeleton connection must land on a YOLO junction
        if skeleton is not None:
            connected_now = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
            all_skel = _trace_wire_connections(
                skeleton, components, junctions, h, w,
                search_radius=skel_search, gap_bridge=skel_gap)
            skel_added = 0
            junc_set = set((int(jx), int(jy)) for jx, jy in junctions)
            for s_ci, s_pi, tx, ty in all_skel:
                if (s_ci, s_pi) in connected_now:
                    continue
                # Only accept junction-target connections (not port-to-port)
                if (int(tx), int(ty)) not in junc_set:
                    continue
                px, py = components[s_ci]["ports"][s_pi]
                min_jd = min((math.hypot(px - jx, py - jy) for jx, jy in junctions), default=99999)
                # For extreme far ports (>450px), also accept port-to-port skeleton trace
                if min_jd > pjf and \
                   ((int(tx), int(ty)) in junc_set or min_jd > pjr * 1.5):
                    p2j_connections.append((s_ci, s_pi, tx, ty))
                    skel_added += 1
            if skel_added:
                print(f"  Port→Junction (skeleton fallback): +{skel_added}")

        # ---- Step 6b: Direct port→port proximity fallback ----
        PORT_PORT_MAX = p2p_max
        # Also match ports on same horizontal/vertical line up to 120px
        PORT_LINE_MAX = p2p_line
        for i, ci in enumerate(components):
            for pi, (px, py) in enumerate(ci["ports"]):
                already_connected = any(cx == i and px_idx == pi for cx, px_idx, jx, jy in p2j_connections)
                if already_connected:
                    continue
                best_j, best_port, best_dist = None, None, PORT_PORT_MAX
                for j, cj in enumerate(components):
                    if j == i:
                        continue
                    for pj_idx, (qx, qy) in enumerate(cj["ports"]):
                        # Prioritize ports on same X or Y line
                        aligned = abs(px - qx) < 15 or abs(py - qy) < 15
                        max_d = PORT_LINE_MAX if aligned else PORT_PORT_MAX
                        d = math.hypot(px - qx, py - qy)
                        if d < max_d and d < best_dist:
                            if aligned or not crosses((px, py), (qx, qy), components, i):
                                best_dist = d
                                best_j = (qx, qy)
                                best_port = (j, pj_idx)
                if best_j:
                    mx, my = (px + best_j[0]) // 2, (py + best_j[1]) // 2
                    p2j_connections.append((i, pi, mx, my))
                    p2j_connections.append((best_port[0], best_port[1], mx, my))
                    if (mx, my) not in junctions:
                        junctions.append((mx, my))

        print(f"  Port→Junction (after direct P2P): {len(p2j_connections)}")

        # ---- Step 6c: Second-pass — wider radius for still-isolated ports ----
        connected_ports = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
        # Track which (ci, junction) already has a port to avoid same-comp short in fallback
        comp_junc_used = defaultdict(set)
        for ci, pi, jx, jy in p2j_connections:
            comp_junc_used[(ci, jx, jy)].add(pi)
        second_pass_added = 0
        for ci, c in enumerate(components):
            for pi, (px, py) in enumerate(c["ports"]):
                if (ci, pi) in connected_ports:
                    continue
                best_j, best_d = None, pjf
                for jx, jy in junctions:
                    d = math.hypot(px - jx, py - jy)
                    if d < best_d and not crosses((px, py), (jx, jy), components, ci):
                        # Don't connect both ports of same component to same junction
                        if len(c["ports"]) == 2:
                            other_pi = 1 - pi
                            if other_pi in comp_junc_used.get((ci, jx, jy), set()):
                                continue
                        best_d = d
                        best_j = (jx, jy)
                if best_j:
                    p2j_connections.append((ci, pi, best_j[0], best_j[1]))
                    connected_ports.add((ci, pi))
                    comp_junc_used[(ci, best_j[0], best_j[1])].add(pi)
                    second_pass_added += 1
        if second_pass_added:
            print(f"  Port→Junction (fallback pass): +{second_pass_added}")

        # ---- Step 6d: Aggressive P2P (always active) ----
        connected_ports = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
        AGGRESSIVE_P2P = agg_p2p
        aggressive_added = 0
        for ci, c in enumerate(components):
            for pi, (px, py) in enumerate(c["ports"]):
                if (ci, pi) in connected_ports:
                    continue
                best_j, best_port, best_dist = None, None, AGGRESSIVE_P2P
                for cj_idx, cj in enumerate(components):
                    if cj_idx == ci:
                        continue
                    for pj_idx, (qx, qy) in enumerate(cj["ports"]):
                        aligned = abs(px - qx) < 15 or abs(py - qy) < 15
                        max_d = PORT_LINE_MAX if aligned else AGGRESSIVE_P2P
                        d = math.hypot(px - qx, py - qy)
                        if d < max_d and d < best_dist:
                            if aligned or not crosses((px, py), (qx, qy), components, ci):
                                best_dist = d
                                best_j = (qx, qy)
                                best_port = (cj_idx, pj_idx)
                if best_j:
                    cj_idx = best_port[0]
                    mx, my = (px + best_j[0]) // 2, (py + best_j[1]) // 2
                    p2j_connections.append((ci, pi, mx, my))
                    p2j_connections.append((best_port[0], best_port[1], mx, my))
                    connected_ports.add((ci, pi))
                    connected_ports.add(best_port)
                    if (mx, my) not in junctions:
                        junctions.append((mx, my))
                    aggressive_added += 1
        if aggressive_added:
            print(f"  Port->Junction (aggressive P2P): +{aggressive_added}")
        else:
            print(f"  Aggressive P2P: no connections")

        # ---- Step 7: Junction -> Junction connections ----
        # Build temp junction->ports map for JJ validity check
        jj_port_map = defaultdict(set)
        jj_port_detail = defaultdict(list)
        for ci, pi, jx, jy in p2j_connections:
            jj_port_map[(jx, jy)].add(ci)
            jj_port_detail[(jx, jy)].append((ci, pi))

        from collections import Counter

        # Track connections per junction for degree constraint
        junc_conn_count = defaultdict(int)

        # Collect all aligned pairs with skeleton verification
        jj_raw = []  # (dist, jx1, jy1, jx2, jy2)
        for i in range(len(junctions)):
            for j in range(i + 1, len(junctions)):
                jx1, jy1 = junctions[i]
                jx2, jy2 = junctions[j]
                comps_1 = jj_port_map.get((jx1, jy1), set())
                comps_2 = jj_port_map.get((jx2, jy2), set())
                if comps_1 and comps_2 and comps_1 == comps_2:
                    continue
                would_short = False
                for ci in comps_1 & comps_2:
                    ports_at_j1 = {p[1] for p in jj_port_detail.get((jx1, jy1), []) if p[0] == ci}
                    ports_at_j2 = {p[1] for p in jj_port_detail.get((jx2, jy2), []) if p[0] == ci}
                    if ports_at_j1 and ports_at_j2 and ports_at_j1 != ports_at_j2:
                        would_short = True
                        break
                if would_short:
                    continue
                non_gnd_1 = {ci for ci in comps_1 if components[ci]["name"] != "GND"}
                non_gnd_2 = {ci for ci in comps_2 if components[ci]["name"] != "GND"}
                if not non_gnd_1 and not non_gnd_2:
                    continue
                dist = math.hypot(jx1 - jx2, jy1 - jy2)
                if dist < JJ_PROXIMITY:
                    jj_raw.append((dist, jx1, jy1, jx2, jy2))
                    continue
                aligned_h = abs(jy1 - jy2) < jj_align and abs(jx1 - jx2) > 10
                aligned_v = abs(jx1 - jx2) < jj_align and abs(jy1 - jy2) > 10
                if not (aligned_h or aligned_v) or dist >= jj_max:
                    continue
                skel_ok = False
                if skeleton is not None:
                    skel_ok = _verify_skeleton_path(skeleton, jx1, jy1, jx2, jy2, margin=5, min_ratio=0.15)
                on_wire = _on_same_wire(jx1, jy1, jx2, jy2, wire_bboxes)
                if skel_ok or on_wire or not config["use_skeleton"]:
                    jj_raw.append((dist, jx1, jy1, jx2, jy2))

        jj_connections = []
        junc_conn_count = defaultdict(int)
        if config["use_nn_filter"]:
            # Nearest-neighbor filter: each junction connects only to its CLOSEST
            # aligned neighbor in each of 4 directions (L, R, U, D). This prevents
            # cascading over-merging from all-pairs alignment.
            neighbor_map = defaultdict(list)  # (jx,jy) -> [(dist, nx, ny), ...]
            for dist, jx1, jy1, jx2, jy2 in jj_raw:
                neighbor_map[(jx1, jy1)].append((dist, jx2, jy2))
                neighbor_map[(jx2, jy2)].append((dist, jx1, jy1))

            for (jx, jy), neighbors in neighbor_map.items():
                # Group neighbors by direction
                left = [(d, nx, ny) for d, nx, ny in neighbors if abs(ny - jy) < jj_align and nx < jx]
                right = [(d, nx, ny) for d, nx, ny in neighbors if abs(ny - jy) < jj_align and nx > jx]
                up = [(d, nx, ny) for d, nx, ny in neighbors if abs(nx - jx) < jj_align and ny < jy]
                down = [(d, nx, ny) for d, nx, ny in neighbors if abs(nx - jx) < jj_align and ny > jy]
                # Pick closest in each direction
                for direction in [left, right, up, down]:
                    if direction:
                        direction.sort(key=lambda x: x[0])
                        d, nx, ny = direction[0]
                        if config["use_degree_constraint"]:
                            max_deg = _junc_branch_count.get((jx, jy), 8)
                            max_deg_n = _junc_branch_count.get((nx, ny), 8)
                            if junc_conn_count[(jx, jy)] >= max_deg + 2 or junc_conn_count[(nx, ny)] >= max_deg_n + 2:
                                continue
                        jj_connections.append((jx, jy, nx, ny))
                        junc_conn_count[(jx, jy)] += 1
                        junc_conn_count[(nx, ny)] += 1
        else:
            # No NN filter: accept all aligned pairs directly
            for dist, jx1, jy1, jx2, jy2 in jj_raw:
                if config["use_degree_constraint"]:
                    max_deg = _junc_branch_count.get((jx1, jy1), 8)
                    max_deg_n = _junc_branch_count.get((jx2, jy2), 8)
                    if junc_conn_count[(jx1, jy1)] >= max_deg + 2 or junc_conn_count[(jx2, jy2)] >= max_deg_n + 2:
                        continue
                jj_connections.append((jx1, jy1, jx2, jy2))
                junc_conn_count[(jx1, jy1)] += 1
                junc_conn_count[(jx2, jy2)] += 1

        # ---- Step 7a-skel: Non-aligned skeleton-verified JJ connections ----
        # Only enable when aligned JJ is sparse (< 15 connections) to avoid
        # over-merging in dense circuits where skeleton is unreliable.
        skel_jj_added = 0
        if len(jj_connections) < 15 and skeleton is not None and config["use_skel_jj"]:
            already_jj = set()
            for jx1, jy1, jx2, jy2 in jj_connections:
                already_jj.add((min(jx1,jx2), min(jy1,jy2), max(jx1,jx2), max(jy1,jy2)))
            for i in range(len(junctions)):
                for j in range(i + 1, len(junctions)):
                    jx1, jy1 = junctions[i]
                    jx2, jy2 = junctions[j]
                    key = (min(jx1,jx2), min(jy1,jy2), max(jx1,jx2), max(jy1,jy2))
                    if key in already_jj:
                        continue
                    dist = math.hypot(jx1 - jx2, jy1 - jy2)
                    if dist < 40 or dist > 400:
                        continue
                    # Skip if already aligned (handled by Step 7)
                    if abs(jy1 - jy2) < jj_align or abs(jx1 - jx2) < jj_align:
                        continue
                    comps_1 = jj_port_map.get((jx1, jy1), set())
                    comps_2 = jj_port_map.get((jx2, jy2), set())
                    would_short = False
                    for ci in comps_1 & comps_2:
                        ports_at_j1 = {p[1] for p in jj_port_detail.get((jx1, jy1), []) if p[0] == ci}
                        ports_at_j2 = {p[1] for p in jj_port_detail.get((jx2, jy2), []) if p[0] == ci}
                        if ports_at_j1 and ports_at_j2 and ports_at_j1 != ports_at_j2:
                            would_short = True
                            break
                    if would_short:
                        continue
                    non_gnd_1 = {ci for ci in comps_1 if components[ci]["name"] != "GND"}
                    non_gnd_2 = {ci for ci in comps_2 if components[ci]["name"] != "GND"}
                    if not non_gnd_1 and not non_gnd_2:
                        continue
                    if _verify_skeleton_any(skeleton, jx1, jy1, jx2, jy2, margin=6, min_ratio=0.50):
                        max_deg_1 = _junc_branch_count.get((jx1, jy1), 8)
                        max_deg_2 = _junc_branch_count.get((jx2, jy2), 8)
                        if junc_conn_count[(jx1, jy1)] < max_deg_1 + 2 and                        junc_conn_count[(jx2, jy2)] < max_deg_2 + 2:
                            jj_connections.append((jx1, jy1, jx2, jy2))
                            junc_conn_count[(jx1, jy1)] += 1
                            junc_conn_count[(jx2, jy2)] += 1
                            skel_jj_added += 1
        if skel_jj_added:
            print(f"  Junction->Junction (skel-nonaligned): +{skel_jj_added}")

        print(f"  Junction->Junction: {len(jj_connections)}")

        # ---- Step 7b: Line-of-sight port connections (sparse junctions only) ----
        los_added = 0
        if config["use_los"]:
            LOS_ALIGN = 25  # horizontal/vertical alignment tolerance (px)
            LOS_SKEL_MIN = int(200 * im_scale)  # distance beyond which skeleton verification is required
            los_candidates = []  # (dist, ia, pa, ib, pb)
            connected_set = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
            for ia in range(len(components)):
                ca = components[ia]
                for pa, (ax, ay) in enumerate(ca["ports"]):
                    for ib in range(ia + 1, len(components)):
                        cb = components[ib]
                        for pb, (bx, by) in enumerate(cb["ports"]):
                            if ca["name"] == "GND" and cb["name"] == "GND":
                                continue
                            same_h = abs(ay - by) < LOS_ALIGN and abs(ax - bx) > 5
                            same_v = abs(ax - bx) < LOS_ALIGN and abs(ay - by) > 5
                            if not (same_h or same_v):
                                continue
                            a_conn = (ia, pa) in connected_set
                            b_conn = (ib, pb) in connected_set
                            if a_conn and b_conn:
                                continue  # both already connected
                            # Check no component bbox blocks the direct line
                            blocked = False
                            for k, ck in enumerate(components):
                                if k == ia or k == ib:
                                    continue
                                kx1, ky1, kx2, ky2 = ck["xyxy"]
                                if same_h:
                                    x_min, x_max = min(ax, bx), max(ax, bx)
                                    if kx2 > x_min + 5 and kx1 < x_max - 5:
                                        if ky1 < ay + LOS_ALIGN and ky2 > ay - LOS_ALIGN:
                                            blocked = True
                                            break
                                else:  # same_v
                                    y_min, y_max = min(ay, by), max(ay, by)
                                    if ky2 > y_min + 5 and ky1 < y_max - 5:
                                        if kx1 < ax + LOS_ALIGN and kx2 > ax - LOS_ALIGN:
                                            blocked = True
                                            break
                            if blocked:
                                continue
                            d = math.hypot(ax - bx, ay - by)
                            # For longer LOS, require skeleton wire support
                            if d > LOS_SKEL_MIN and skeleton is not None and config["use_skeleton"]:
                                if not _verify_skeleton_path(skeleton, ax, ay, bx, by, margin=5, min_ratio=0.15):
                                    continue
                            los_candidates.append((d, ia, pa, ib, pb))

            # For each component pair, keep only the closest port pair
            best_pairs = {}
            for d, ia, pa, ib, pb in los_candidates:
                key = (min(ia, ib), max(ia, ib))
                if key not in best_pairs or d < best_pairs[key][0]:
                    best_pairs[key] = (d, ia, pa, ib, pb)

            for d, ia, pa, ib, pb in best_pairs.values():
                mx = (components[ia]["ports"][pa][0] + components[ib]["ports"][pb][0]) // 2
                my = (components[ia]["ports"][pa][1] + components[ib]["ports"][pb][1]) // 2
                p2j_connections.append((ia, pa, mx, my))
                p2j_connections.append((ib, pb, mx, my))
                connected_set.add((ia, pa))
                connected_set.add((ib, pb))
                los_added += 1
        if los_added:
            print(f"  Line-of-sight: +{los_added} connections")

        # ---- Step 7c: Force-connect remaining isolated ports ----
        forced = 0
        if config["use_force_connect"]:
            connected_set = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
            comp_junc_ports = defaultdict(set)
            for ci, pi, jx, jy in p2j_connections:
                comp_junc_ports[(ci, jx, jy)].add(pi)
            for ci, c in enumerate(components):
                unconnected = [pi for pi in range(len(c["ports"])) if (ci, pi) not in connected_set]
                if len(unconnected) == 1:
                    # Exactly one port isolated -> force to nearest valid junction
                    pi = unconnected[0]
                    px, py = c["ports"][pi]
                    best_j, best_d = None, 99999
                    for jx, jy in junctions:
                        d = math.hypot(px - jx, py - jy)
                        if d < best_d and d < 500:
                            # Skip if this junction already has any OTHER port of this component
                            existing = comp_junc_ports.get((ci, jx, jy), set())
                            if existing - {pi}:
                                continue
                            if d > 150 and skeleton is not None and config["use_skeleton"]:
                                if not _verify_skeleton_path(skeleton, px, py, jx, jy, margin=5, min_ratio=0.15):
                                    continue
                            if not crosses((px, py), (jx, jy), components, ci):
                                best_d = d
                                best_j = (jx, jy)
                    if best_j:
                        p2j_connections.append((ci, pi, best_j[0], best_j[1]))
                        connected_set.add((ci, pi))
                        comp_junc_ports[(ci, best_j[0], best_j[1])].add(pi)
                        forced += 1
        if forced:
            print(f"  Force-connected: +{forced} isolated ports")

        # ---- Step 7d: Close-port proximity rule ----
        # If two unconnected ports from different components are within 25px
        # (with 5px hand-drawn tolerance per axis), connect them directly.
        # This catches adjacent ports that P2J missed (e.g. C2.2/R2.2).
        close_added = 0
        if config["use_close_port"]:
            connected_now = set((ci, pi) for ci, pi, jx, jy in p2j_connections)
            for ia, ca in enumerate(components):
                for pa, (ax, ay) in enumerate(ca["ports"]):
                    if (ia, pa) in connected_now:
                        continue
                    for ib, cb in enumerate(components):
                        if ib <= ia:
                            continue
                        for pb, (bx, by) in enumerate(cb["ports"]):
                            if (ib, pb) in connected_now:
                                continue
                            dx = abs(ax - bx)
                            dy = abs(ay - by)
                            # Close ports: within 50px euclidean distance
                            if math.hypot(dx, dy) < 50:
                                # Don't wire two GNDs together
                                if ca["name"] == "GND" and cb["name"] == "GND":
                                    continue
                                mx, my = (ax+bx)//2, (ay+by)//2
                                p2j_connections.append((ia, pa, mx, my))
                                p2j_connections.append((ib, pb, mx, my))
                                connected_now.add((ia, pa))
                                connected_now.add((ib, pb))
                                if (mx, my) not in junctions:
                                    junctions.append((mx, my))
                                close_added += 1
                                break
                        if (ia, pa) in connected_now:
                            break
        if close_added:
            print(f"  Close-port proximity: +{close_added} connections")

        # ---- Step 8: Route all connections ----
        routes = []
        # Build set of port coordinates for direct LOS connection detection
        all_port_coords = set()
        for c in components:
            for px, py in c["ports"]:
                all_port_coords.add((int(px), int(py)))

        for ci, pi, jx, jy in p2j_connections:
            px, py = components[ci]["ports"][pi]
            is_port_target = (int(jx), int(jy)) in all_port_coords
            # Always try orthogonal routing first
            pts = route((px, py), (jx, jy), components, ci)
            if pts:
                routes.append(pts)
            elif is_port_target or not crosses((px, py), (jx, jy), components, ci):
                # Route failed but path is clear: use direct line as fallback
                routes.append([(px, py), (jx, jy)])
        for jx1, jy1, jx2, jy2 in jj_connections:
            pts = route((jx1, jy1), (jx2, jy2), components, -1)
            if pts:
                routes.append(pts)

        print(f"  Routes: {len(routes)}")

    # ---- Step 9: Build connection graph for LLM context ----
    # Union-find on ports connected through junctions
    port_parent = {}
    for ci, c in enumerate(components):
        for pi in range(len(c["ports"])):
            port_parent[(ci, pi)] = (ci, pi)

    def find_p(k):
        while port_parent[k] != k:
            port_parent[k] = port_parent[port_parent[k]]
            k = port_parent[k]
        return k

    def union_p(k1, k2):
        r1, r2 = find_p(k1), find_p(k2)
        if r1 != r2:
            port_parent[r2] = r1

    # Junction → connected ports
    jp_map = defaultdict(list)
    for ci, pi, jx, jy in p2j_connections:
        jp_map[(jx, jy)].append((ci, pi))

    for jkey, port_list in jp_map.items():
        for i in range(1, len(port_list)):
            union_p(port_list[0], port_list[i])

    # Junction→Junction merges
    for jx1, jy1, jx2, jy2 in jj_connections:
        ports_a = jp_map.get((jx1, jy1), [])
        ports_b = jp_map.get((jx2, jy2), [])
        if ports_a and ports_b:
            union_p(ports_a[0], ports_b[0])

    # LOS connections: merge bidirectional port pairs
    all_port_coords_map = {}
    for ci, c in enumerate(components):
        for pi, (px, py) in enumerate(c["ports"]):
            key = (int(px), int(py))
            all_port_coords_map[key] = (ci, pi)

    for ci, pi, jx, jy in p2j_connections:
        target = (int(jx), int(jy))
        if target in all_port_coords_map:
            cj, pj = all_port_coords_map[target]
            # Check reverse connection exists
            for cii, pii, jxx, jyy in p2j_connections:
                if cii == cj and pii == pj:
                    rev_target = (int(jxx), int(jyy))
                    rev_expected = (int(components[ci]["ports"][pi][0]),
                                   int(components[ci]["ports"][pi][1]))
                    if rev_target == rev_expected:
                        union_p((ci, pi), (cj, pj))
                        break

    # Build connected groups
    groups = defaultdict(set)
    for ci, c in enumerate(components):
        for pi in range(len(c["ports"])):
            root = find_p((ci, pi))
            groups[root].add((ci, pi))

    # Build detailed connection info with port labels
    conn_pairs = []      # [(comp_names, port_details)]
    conn_details = []    # human-readable per-group description
    for root, port_set in groups.items():
        comps_in_group = set(ci for ci, pi in port_set)
        if len(comps_in_group) >= 2:
            names = [f"{components[ci]['designator']}" for ci in comps_in_group]
            # Per-port detail
            port_list = []
            for ci, pi in sorted(port_set):
                c = components[ci]
                pname = c['designator']
                labels = PORT_LABELS.get(c['name'], [str(i) for i in range(len(c['ports']))])
                if c.get('label_swap'):
                    labels = list(reversed(labels))
                if pi < len(labels):
                    plabel = labels[pi]
                else:
                    plabel = str(pi)
                port_list.append(f"{pname}.{plabel}")
            conn_pairs.append(sorted(set(names)))
            conn_details.append(", ".join(sorted(port_list)))

    # ---- Post-connection fix: unshort 2-terminal components ----
    # If both ports of a 2-terminal component ended up in same group, remove the farther p2j
    for ci, c in enumerate(components):
        if len(c["ports"]) != 2:
            continue
        p0_conn = [(jx, jy) for cxi, pxi, jx, jy in p2j_connections if cxi == ci and pxi == 0]
        p1_conn = [(jx, jy) for cxi, pxi, jx, jy in p2j_connections if cxi == ci and pxi == 1]
        if p0_conn and p1_conn:
            # Check if both ports are in same connected group via Union-Find
            j0, j1 = p0_conn[0], p1_conn[0]
            root0 = find_p((ci, 0)) if (ci, 0) in port_parent else None
            root1 = find_p((ci, 1)) if (ci, 1) in port_parent else None
            if root0 is not None and root1 is not None and root0 == root1:
                # Both ports in same group → remove farther connection
                d0 = math.hypot(c["ports"][0][0] - j0[0], c["ports"][0][1] - j0[1])
                d1 = math.hypot(c["ports"][1][0] - j1[0], c["ports"][1][1] - j1[1])
                if d0 > d1:
                    p2j_connections = [x for x in p2j_connections if not (x[0] == ci and x[1] == 0)]
                else:
                    p2j_connections = [x for x in p2j_connections if not (x[0] == ci and x[1] == 1)]

    # Rebuild connection graph after fix
    port_parent = {}
    for ci, c in enumerate(components):
        for pi in range(len(c["ports"])):
            port_parent[(ci, pi)] = (ci, pi)
    jp_map_fix = defaultdict(list)
    for ci, pi, jx, jy in p2j_connections:
        jp_map_fix[(jx, jy)].append((ci, pi))
    for jkey, port_list in jp_map_fix.items():
        for i in range(1, len(port_list)):
            if port_list[0][0] != port_list[i][0]:
                ra, rb = find_p(port_list[0]), find_p(port_list[i])
                if ra != rb:
                    port_parent[rb] = ra
    for jx1, jy1, jx2, jy2 in jj_connections:
        ports_a = jp_map_fix.get((jx1, jy1), [])
        ports_b = jp_map_fix.get((jx2, jy2), [])
        if ports_a and ports_b:
            ra, rb = find_p(ports_a[0]), find_p(ports_b[0])
            if ra != rb:
                port_parent[rb] = ra
    groups = defaultdict(set)
    for ci, c in enumerate(components):
        for pi in range(len(c["ports"])):
            root = find_p((ci, pi))
            groups[root].add((ci, pi))
    conn_pairs = []
    conn_details = []
    for root, port_set in groups.items():
        comps_in_group = set(ci for ci, pi in port_set)
        if len(comps_in_group) >= 2:
            names = [f"{components[ci]['designator']}" for ci in comps_in_group]
            port_list = []
            for ci, pi in sorted(port_set):
                c = components[ci]
                pname = c['designator']
                labels = PORT_LABELS.get(c['name'], [str(i) for i in range(len(c['ports']))])
                if c.get('label_swap'):
                    labels = list(reversed(labels))
                plabel = labels[pi] if pi < len(labels) else str(pi)
                if pname == "GND":
                    port_list.append("GND")
                else:
                    port_list.append(f"{pname}.{plabel}")
            conn_pairs.append(sorted(set(names)))
            conn_details.append(", ".join(sorted(port_list)))

    # ---- Step 10: LLM evaluation ----
    comp_lines = []
    for c in components:
        d = c["designator"]
        nm = NM_CH.get(c["name"], c["name"])
        v = f"值={c['value']}" if c['value'] else "值未识别"
        comp_lines.append(f"{d}({nm},{v})")
    comp_str = "; ".join(comp_lines)

    conn_str = ""
    if conn_pairs:
        conn_lines = []
        for grp in conn_pairs:
            conn_lines.append("-".join(grp))
        conn_str = "; ".join(conn_lines)

    prompt = f"""你是资深电路设计评审专家。根据以下自动识别的电路信息，输出结构化评审报告。

【识别结果】
元器件({len(components)}个): {comp_str}
连接关系: {conn_str if conn_str else '未检测到连接'}

【评审要求】按以下三部分结构化输出：

## 一、合规性/规则检查 (Rule-Based Audit)
逐项检查并给出结论:
- 电源与地: 是否存在电源对地短路、电源反接、GND悬空
- 元件参数: 数值是否在合理范围(电阻1Ω~10MΩ, 电容1pF~10mF, 电压0.1V~1kV)
- 极性元件: 二极管/LED/极性电容方向是否正确
- 回路完整性: 是否形成闭合回路，有无悬空节点

## 二、设计缺陷与风险分析 (DFMEA思维)
- 识别潜在失效模式(开路/短路/过流/过压/反压)
- 评估严重度(S)、发生度(O)、探测度(D)，计算RPN=S×O×D
- 列出TOP3风险项

## 三、经验性原理评估 (Heuristic Review)
- 电路类型推测及依据
- 参数匹配性(如RC时间常数、分压比、限流合理性)
- 工程实践建议(保护电路、滤波、布局等)

## 总结
- 整体质量评分(1-10)
- 一句话改进建议"""

    if config["skip_llm"]:
        response = "(LLM skipped for experiment)"
    else:
        print(f"\n  LLM evaluating... (prompt={len(prompt)} chars)")
        response = ask(prompt)
        print(f"  LLM response received ({len(response)} chars)")

    # ---- Step 11: Draw annotated image ----
    out_img = Path(img_path).parent / (Path(img_path).stem + "_wired.jpg")
    _draw_result(img_path, components, text_values, junctions, routes, p2j_connections,
                 jj_connections, str(out_img))

    # ---- Rebuild junction numbering (after all synthetic junctions added) ----
    junctions = sorted(set((jx, jy) for jx, jy in junctions), key=lambda p: (p[1], p[0]))
    jid_map = {(jx, jy): f"J{i+1}" for i, (jx, jy) in enumerate(junctions)}

    # ---- Step 12: Save TXT ----
    out_txt = Path(img_path).parent / (Path(img_path).stem + "_result.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"电路图评价: {img_name}\n{'='*60}\n\n")
        f.write(f"[检测元件] {len(components)}个\n")
        for c in components:
            v = f"值={c['value']}" if c['value'] else "值未识别"
            nm = NM_CH.get(c["name"], c["name"])
            # Polarity/terminal info
            labels = PORT_LABELS.get(c["name"], [])
            pinfo = ""
            if labels and len(c["ports"]) == len(labels):
                pinfo = f" 端口:{','.join(labels)}"
            f.write(f"  {c['designator']:6s} {nm:10s} {v}{pinfo}\n")
        f.write(f"\n[OCR文字] {len(text_values)}个\n")
        for tv in text_values:
            f.write(f"  \"{tv['text']}\"\n")
        f.write(f"\n[节点] {len(junctions)}个\n")
        for jx, jy in junctions:
            f.write(f"  {jid_map.get((jx,jy),'?')}: ({jx},{jy})\n")
        f.write(f"\n[连接关系]\n")
        if conn_details:
            for i, detail in enumerate(conn_details):
                f.write(f"  连通组{i+1}: {detail}\n")
        elif conn_pairs:
            for grp in conn_pairs:
                f.write(f"  连通组: {' - '.join(grp)}\n")
        else:
            f.write("  (未检测到连接)\n")
        f.write(f"\n[LLM评价]\n{'='*60}\n{response}\n")

    print(f"  Output: {out_img.name}, {out_txt.name}")
    # Build raw_groups: port-level connectivity for evaluation
    raw_groups = [sorted(port_set) for port_set in groups.values()
                  if len(set(ci for ci, pi in port_set)) >= 2]
    return dict(components=components, text_values=text_values, junctions=junctions,
                routes=routes, conn_pairs=conn_pairs, raw_groups=raw_groups,
                evaluation=response)

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_result(img_path, components, text_values, junctions, routes,
                 p2j_connections, jj_connections, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    COLORS = {
        "Resistor": (0, 200, 0), "Capacitor": (200, 200, 0), "Inductor": (0, 200, 200),
        "Diode": (200, 0, 100), "LED": (0, 255, 255), "Zener Diode": (200, 0, 150),
        "V-DC": (200, 0, 0), "V-AC": (150, 0, 0),
        "I-DC": (0, 100, 0), "I-AC": (0, 150, 100),
        "GND": (0, 0, 200), "MOSFET-N": (100, 100, 0), "MOSFET-P": (100, 0, 100),
        "BJT-NPN": (150, 150, 0), "BJT-PNP": (150, 0, 150), "Op-Amp": (0, 100, 100),
        "Thyristor": (200, 100, 0), "Triac": (200, 100, 100),
        "Diac": (150, 100, 50), "Varistor": (100, 50, 200),
        "IC": (255, 150, 0), "NE555": (255, 120, 50), "Voltage-Regulator": (0, 180, 100),
    }

    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        color = COLORS.get(c["name"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = c["designator"]
        if c["value"]:
            label += "=" + c["value"]
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        # Red port dots with polarity labels
        port_labels = PORT_LABELS.get(c["name"], [])
        for pi, (px, py) in enumerate(c["ports"]):
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1)
            if pi < len(port_labels):
                pl = port_labels[pi]
                # Skip redundant GND label
                if pl != "GND":
                    cv2.putText(img, pl, (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                               0.3, (0, 0, 200), 1, cv2.LINE_AA)

    # Green junction dots with numbering
    jid_map = {}
    for i, (jx, jy) in enumerate(junctions):
        jid = f"J{i+1}"
        jid_map[(jx, jy)] = jid
        cv2.circle(img, (jx, jy), 5, (0, 255, 0), -1)
        cv2.circle(img, (jx, jy), 6, (255, 255, 255), 1)
        cv2.putText(img, jid, (jx + 7, jy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   0.35, (0, 180, 0), 1, cv2.LINE_AA)

    # Blue wire lines
    for pts in routes:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 0, 0), 2)

    # OCR labels
    for tv in text_values:
        x1, y1, x2, y2 = tv.get("xyxy", (0, 0, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 1)
        cv2.putText(img, f'OCR:"{tv["text"]}"', (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1, cv2.LINE_AA)

    # Legend — only show types present in this image
    used_types = set(c["name"] for c in components)
    y = 20
    for name, color in sorted(COLORS.items()):
        if name not in used_types:
            continue
        cv2.putText(img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        y += 15

    cv2.imwrite(out_path, img)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        # Default: process all circuit_*.jpg in project root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = sorted(Path(root).glob("circuit_*.jpg"))
        paths = [str(p) for p in paths]
    for p in paths:
        try:
            process_image(p)
        except Exception as e:
            print(f"ERROR processing {p}: {e}")
            import traceback
            traceback.print_exc()
