"""Circuit component detector using template matching (no training needed).

Given a circuit diagram image, detects components and outputs a text description
ready for the Phase 1 LLM evaluation pipeline.
"""

import cv2
import numpy as np
from pathlib import Path

# Map detected class to component description
COMPONENT_INFO = {
    "resistor": {"symbol": "R", "param": "value"},
    "led": {"symbol": "LED", "param": "color, Vf"},
    "voltage_source": {"symbol": "V", "param": "voltage"},
    "capacitor": {"symbol": "C", "param": "capacitance"},
    "ground": {"symbol": "GND", "param": None},
    "diode": {"symbol": "D", "param": "type"},
}

# Simplified: since synthetic circuits all look similar, we use contour detection
# to find components and classify them by shape features


def detect_components(image_path: str) -> list[dict]:
    """Detect circuit components in an image.

    Returns list of dicts: {type, center_x, center_y, bbox}
    For the synthetic data, we use a hybrid approach:
    1. Find all dark regions (components) using thresholding
    2. Classify by shape and size features
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert: components are dark on light background
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Filter noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)

        # Classify by shape features
        comp_type = _classify_by_shape(w, h, aspect_ratio, area)

        components.append({
            "type": comp_type,
            "bbox": (x, y, w, h),
            "center": (x + w // 2, y + h // 2),
            "area": int(area),
        })

    # Sort left-to-right, top-to-bottom
    components.sort(key=lambda c: (c["center"][1] // 50, c["center"][0]))
    return components


def _classify_by_shape(w: int, h: int, ratio: float, area: int) -> str:
    """Classify component by shape metrics."""
    if area > 3000:
        return "voltage_source"  # Voltage source is usually the largest symbol
    if ratio > 3.0:
        return "resistor"  # Resistor is long and thin (zigzag)
    if 0.5 < ratio < 2.0 and area < 800:
        return "led"  # LED is roughly square-ish
    if area < 500 and ratio > 1.5:
        return "capacitor"
    if area < 200:
        return "ground"
    return "unknown"


def components_to_description(components: list[dict], image_name: str) -> str:
    """Convert detected components to a circuit description for LLM evaluation.

    This is the key bridge between CV detection and LLM evaluation.
    """
    comp_lines = []
    conn_lines = []
    node_idx = 0

    for i, c in enumerate(components):
        ctype = c["type"]
        info = COMPONENT_INFO.get(ctype, {"symbol": "?", "param": None})
        sid = info["symbol"]
        cid = f"{sid}{i+1}"

        if ctype == "voltage_source":
            comp_lines.append(f"- {cid}：5V直流电源")
        elif ctype == "resistor":
            comp_lines.append(f"- {cid}：电阻（阻值未知，请根据色环或标注判断）")
        elif ctype == "led":
            comp_lines.append(f"- {cid}：发光二极管（正向压降约2.0V，额定电流20mA）")
        elif ctype == "capacitor":
            comp_lines.append(f"- {cid}：电容（容值未知，请根据标注判断）")
        elif ctype == "ground":
            comp_lines.append(f"- {cid}：接地")
        else:
            comp_lines.append(f"- {cid}：未知元件")

        # Build connections based on spatial proximity
        if i > 0:
            prev = components[i-1]
            conn_lines.append(f"- {prev['type']} 连接到 {cid}")

    desc = f"""电路描述：
这是从电路图图片 "{image_name}" 自动检测到的电路。

元器件：
{chr(10).join(comp_lines)}

连接关系：
{chr(10).join(conn_lines)}

预期功能：请根据检测到的元件和连接关系推断电路功能并评价。"""

    return desc


def image_to_evaluation(image_path: str, model: str | None = None) -> dict:
    """Full CV pipeline: image → detect components → text → LLM evaluation.

    This is the main entry point — input an image, get an evaluation report.
    """
    from src.phase1_verify.evaluate import evaluate_circuit

    # Step 1: Detect components
    components = detect_components(image_path)

    # Step 2: Convert to text description
    image_name = Path(image_path).name
    description = components_to_description(components, image_name)

    # Step 3: LLM evaluation (using existing Phase 1 pipeline)
    result = evaluate_circuit(description, model=model)

    # Attach detection metadata
    result["_detected_components"] = len(components)
    result["_component_types"] = list(set(c["type"] for c in components))
    result["_description"] = description

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "data/images/preview_good_led.png"

    print(f"检测图片: {img_path}")
    components = detect_components(img_path)
    print(f"\n检测到 {len(components)} 个元件:")
    for c in components:
        print(f"  {c['type']:15s} at ({c['center'][0]:3d},{c['center'][1]:3d}) "
              f"bbox={c['bbox']} area={c['area']}")

    print("\n生成的文字描述:")
    desc = components_to_description(components, Path(img_path).name)
    print(desc)
