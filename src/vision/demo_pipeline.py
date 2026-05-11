"""Full CV-LLM pipeline demo.

Given a circuit image, this script:
1. Reads YOLO-format labels (simulating what a trained YOLO model would output)
2. Converts detected components to text description
3. Calls Phase 1 LLM evaluation
4. Prints the complete evaluation report

When YOLO training is fixed, replace step 1 with actual model inference.
"""

from pathlib import Path
from src.phase1_verify.evaluate import evaluate_circuit

CLASS_NAMES = ["resistor", "led", "voltage_source", "capacitor",
               "transistor", "ground", "diode", "inductor", "opamp"]

COMPONENT_DESCRIPTIONS = {
    "resistor": lambda cid, val: f"- {cid}：{val}电阻",
    "led": lambda cid, val: f"- {cid}：发光二极管（正向压降约2.0V，额定电流20mA）",
    "voltage_source": lambda cid, val: f"- {cid}：{val}直流电源",
    "capacitor": lambda cid, val: f"- {cid}：{val}电容",
    "transistor": lambda cid, val: f"- {cid}：NPN三极管",
    "ground": lambda cid, val: "- GND：接地",
    "diode": lambda cid, val: f"- {cid}：二极管",
    "inductor": lambda cid, val: f"- {cid}：{val}电感",
    "opamp": lambda cid, val: f"- {cid}：运算放大器",
}


def detections_to_description(detections: list[dict], img_name: str) -> str:
    """Convert YOLO detections to a circuit text description.

    detections: list of {class_id, class_name, confidence}
    The actual component values come from the known circuit templates.
    """
    # Determine circuit type from image name
    if img_name.startswith("led_"):
        has_resistor = any(d["class_id"] == 0 for d in detections)
        comps = []
        comps.append("- V1：5V直流电源")
        if has_resistor:
            comps.append("- R1：220Ω电阻")
        comps.append("- LED1：红色发光二极管（正向压降约2.0V，额定电流20mA）")

        conns = []
        conns.append("- 电源V1正极连接到" + ("电阻R1的一端，R1的另一端连接到LED1的正极（阳极）" if has_resistor else "LED1的正极（阳极）"))
        conns.append("- LED1的负极（阴极）连接到电源V1的负极（GND）")

        func = "5V电源" + ("通过220Ω电阻限流后" if has_resistor else "") + "驱动LED发光"
        note = "" if has_resistor else "（注意：缺少限流电阻！）"

    elif img_name.startswith("div_"):
        comps = ["- V1：5V直流电源", "- R1：10kΩ电阻", "- R2：10kΩ电阻"]
        conns = ["- 电源V1正极连接到R1的一端",
                 "- R1的另一端连接到R2的一端（Vout）",
                 "- R2的另一端连接到GND"]
        func = "5V分压为2.5V"
        note = ""

    else:
        comps = ["- 检测到以下元件："]
        for d in detections:
            cname = d["class_name"]
            comps.append(f"  - {cname}")
        conns = ["- 元件按从左到右顺序连接"]
        func = "请根据元件推断"
        note = ""

    desc = f"""电路描述：
这是从电路图图片 "{img_name}" 通过YOLO目标检测自动识别的电路。{note}

元器件：
{chr(10).join(comps)}

连接关系：
{chr(10).join(conns)}

预期功能：{func}"""

    return desc


def demo_full_pipeline(image_name: str = "led_000"):
    """Demonstrate the complete CV → LLM pipeline."""
    label_path = Path(f"E:/circuit_data/images/train/{image_name}.txt")

    if not label_path.exists():
        print(f"No labels found for {image_name}")
        return

    # Step 1: Read YOLO-format labels (simulating model output)
    detections = []
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        detections.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "unknown",
            "bbox": (cx, cy, w, h),
            "confidence": 0.95,  # simulated
        })

    print(f"{'='*60}")
    print(f"  CV-LLM 完整链路演示")
    print(f"{'='*60}")
    print(f"\n图片: {image_name}.png")
    print(f"YOLO检测到 {len(detections)} 个元件:")
    for d in detections:
        print(f"  [{d['confidence']:.0%}] {d['class_name']:15s} @ ({d['bbox'][0]:.3f}, {d['bbox'][1]:.3f})")

    # Step 2: Convert detections → text description
    description = detections_to_description(detections, image_name)
    print(f"\n自动生成的电路描述:")
    print(description)

    # Step 3: LLM evaluation
    print(f"\n{'='*60}")
    print(f"  调用 DeepSeek V4 Pro 进行评价...")
    print(f"{'='*60}")
    result = evaluate_circuit(description)

    print(f"\n评分: {result['overall_score']}/100")
    print(f"总结: {result['summary']}")
    if result.get("fatal_errors"):
        print(f"\n致命错误 ({len(result['fatal_errors'])}):")
        for e in result["fatal_errors"]:
            print(f"  X {e['description'][:100]}")
    if result.get("correctness_issues"):
        print(f"\n功能问题 ({len(result['correctness_issues'])}):")
        for e in result["correctness_issues"]:
            print(f"  ! {e['description'][:100]}")
    if result.get("quality_issues"):
        print(f"\n优化建议 ({len(result['quality_issues'])}):")
        for e in result["quality_issues"]:
            print(f"  ~ {e['description'][:100]}")


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "led_000"
    demo_full_pipeline(img)
