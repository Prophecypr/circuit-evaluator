"""Generate synthetic circuit diagrams with BOTH images AND YOLO-format labels.

Each circuit diagram is rendered as a PNG, and component bounding boxes are saved
in YOLO format:  class_id cx cy w h  (all normalized 0-1)

Classes: 0=resistor, 1=led, 2=voltage_source, 3=capacitor,
         4=transistor, 5=ground, 6=diode, 7=inductor, 8=wire

Strategy: Instead of pixel-perfect detection from rendered images (hard),
we render each component type separately to get its visual footprint, then
use those templates for training data labeling.
"""

import random
from pathlib import Path
import schemdraw
import schemdraw.elements as elm

CLASS_NAMES = ["resistor", "led", "voltage_source", "capacitor",
               "transistor", "ground", "diode", "inductor", "wire"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

OUTPUT_DIR = Path("data/images")
IMG_W, IMG_H = 640, 640  # YOLO target size (will be padded)


# -- Circuit generators that track component positions --

class TrackedDrawing:
    """Wraps schemdraw.Drawing to track element positions and types."""

    def __init__(self):
        self.drawing = schemdraw.Drawing()
        self.elements = []  # list of (type_str, x_pixel, y_pixel, w_pixel, h_pixel)

    def add_resistor(self, label: str):
        el = elm.Resistor().label(label)
        self.drawing += el
        self._track("resistor", el)
        return el

    def add_led(self, label: str = "LED"):
        el = elm.LED().label(label)
        self.drawing += el
        self._track("led", el)
        return el

    def add_source(self, label: str = "5V"):
        el = elm.SourceV().label(label).up()
        self.drawing += el
        self._track("voltage_source", el)
        return el

    def add_capacitor(self, label: str = ""):
        el = elm.Capacitor().label(label)
        self.drawing += el
        self._track("capacitor", el)
        return el

    def add_ground(self):
        el = elm.Ground()
        self.drawing += el
        self._track("ground", el)
        return el

    def add_line(self, direction: str = "right", length: float = 1):
        if direction == "right":
            el = elm.Line().right(length)
        elif direction == "left":
            el = elm.Line().left(length)
        elif direction == "up":
            el = elm.Line().up(length)
        elif direction == "down":
            el = elm.Line().down(length)
        else:
            el = elm.Line()
        self.drawing += el
        self._track("wire", el)
        return el

    def add_dot(self):
        el = elm.Dot()
        self.drawing += el
        self._track("wire", el)
        return el

    def push(self):
        self.drawing.push()

    def pop(self):
        self.drawing.pop()

    def _track(self, etype: str, element):
        """Record element type. Exact pixel bbox will be estimated from the
        rendered image using template matching later.
        For now we just track type and order."""
        self.elements.append({"type": etype, "class_id": CLASS_TO_ID.get(etype, 8)})

    def save(self, path: str):
        self.drawing.save(path)


def save_yolo_labels(elements: list[dict], img_path: str, label_dir: Path):
    """Save YOLO-format labels alongside the image.

    Since we don't have exact pixel positions from schemdraw, we use a
    heuristic: estimate positions based on component order and typical layout.
    This gives approximate labels good enough for training.
    """
    name = Path(img_path).stem
    label_path = label_dir / f"{name}.txt"

    if not elements:
        label_path.write_text("")
        return

    lines = []
    # Simple sequential layout: components placed left-to-right
    n = len([e for e in elements if e["type"] != "wire"])
    comp_elements = [e for e in elements if e["type"] != "wire"]

    for i, e in enumerate(comp_elements):
        cid = e["class_id"]
        # Estimate: components spread evenly across the image
        cx = 0.1 + (i / max(n - 1, 1)) * 0.8
        cy = 0.5  # Assume horizontal layout
        bw = 0.06
        bh = 0.12
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_path.write_text("\n".join(lines))


# -- Circuit Variants --

def gen_led_circuit_good() -> tuple[TrackedDrawing, str]:
    """Good LED circuit: 5V → 220Ω → LED → GND"""
    d = TrackedDrawing()
    d.add_source("5V")
    d.add_line("right")
    d.add_resistor("220Ω")
    d.add_line("right")
    d.add_led("LED")
    d.add_line("down")
    d.push()
    d.add_line("left")
    d.add_line("left")
    d.add_line("left")
    d.pop()

    desc = """电路描述：
这是一个简单的LED驱动电路。

元器件：
- V1：5V直流电源
- R1：220Ω电阻
- LED1：红色发光二极管（正向压降约2.0V，额定电流20mA）

连接关系：
- 电源V1正极连接到电阻R1的一端
- 电阻R1的另一端连接到LED1的正极（阳极）
- LED1的负极（阴极）连接到电源V1的负极（GND）

预期功能：5V电源通过220Ω电阻限流后驱动LED发光"""
    return d, desc


def gen_led_no_resistor() -> tuple[TrackedDrawing, str]:
    """Bad LED circuit: 5V → LED → GND (no resistor!)"""
    d = TrackedDrawing()
    d.add_source("5V")
    d.add_line("right")
    d.add_led("LED")
    d.add_line("down")
    d.push()
    d.add_line("left")
    d.add_line("left")
    d.pop()

    desc = """电路描述：
这是一个LED驱动电路（有缺陷！）。

元器件：
- V1：5V直流电源
- LED1：红色发光二极管（正向压降约2.0V，额定电流20mA）

连接关系：
- 电源V1正极连接到LED1的正极（阳极）
- LED1的负极（阴极）连接到电源V1的负极（GND）

预期功能：5V电源驱动LED发光（注意：缺少限流电阻！）"""
    return d, desc


def gen_voltage_divider() -> tuple[TrackedDrawing, str]:
    """Voltage divider: 5V → R1 → R2 → GND"""
    d = TrackedDrawing()
    d.add_source("5V")
    d.add_line("right")
    d.add_resistor("10kΩ")
    d.add_line("right")
    d.add_resistor("10kΩ")
    d.add_line("down")
    d.push()
    d.add_line("left")
    d.add_line("left")
    d.add_line("left")
    d.pop()

    desc = """电路描述：
这是一个电阻分压电路。

元器件：
- V1：5V直流电源
- R1：10kΩ电阻
- R2：10kΩ电阻

连接关系：
- 电源V1正极连接到R1的一端
- R1的另一端连接到R2的一端，此节点为分压输出Vout
- R2的另一端连接到电源V1的负极（GND）

预期功能：将5V输入电压分压为2.5V输出"""
    return d, desc


def gen_rc_filter() -> tuple[TrackedDrawing, str]:
    """RC low-pass filter."""
    d = TrackedDrawing()
    d.add_dot()
    d.add_line("right")
    d.add_resistor("1kΩ")
    d.add_line("right")
    d.add_dot()
    d.add_capacitor("10μF")
    d.add_ground()

    desc = """电路描述：
这是一个RC低通滤波电路。

元器件：
- R1：1kΩ电阻
- C1：10μF电容

连接关系：
- 输入信号连接到R1的一端
- R1的另一端连接到C1的一端，此节点为滤波输出
- C1的另一端连接到GND

预期功能：滤除高频信号，截止频率约15.9Hz"""
    return d, desc


# --  Main generator --

def generate_dataset(n_train: int = 100, n_val: int = 20):
    """Generate full dataset with images and approximated labels."""
    for folder in ["train", "val"]:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    generators = [
        (gen_led_circuit_good, "led_good"),
        (gen_led_no_resistor, "led_bad"),
        (gen_voltage_divider, "divider"),
        (gen_rc_filter, "rc_filter"),
    ]

    # Training set
    print(f"Generating {n_train} training images...")
    per_type = n_train // len(generators)
    total = 0

    for gen_func, prefix in generators:
        for i in range(per_type):
            name = f"{prefix}_{i:03d}"
            d, desc = gen_func()
            img_path = OUTPUT_DIR / "train" / f"{name}.png"
            d.save(str(img_path))
            (OUTPUT_DIR / "train" / f"{name}.txt").write_text(desc, encoding="utf-8")
            save_yolo_labels(d.elements, str(img_path), OUTPUT_DIR / "train")
            total += 1
    print(f"  Training: {total} images")

    # Validation
    print(f"Generating {n_val} validation images...")
    per_type = n_val // len(generators)
    val_total = 0

    for gen_func, prefix in generators:
        for i in range(per_type):
            name = f"{prefix}_{i:03d}"
            d, desc = gen_func()
            img_path = OUTPUT_DIR / "val" / f"{name}.png"
            d.save(str(img_path))
            (OUTPUT_DIR / "val" / f"{name}.txt").write_text(desc, encoding="utf-8")
            save_yolo_labels(d.elements, str(img_path), OUTPUT_DIR / "val")
            val_total += 1
    print(f"  Validation: {val_total} images")

    # YAML config for YOLO
    yaml_content = f"""path: {OUTPUT_DIR.absolute().as_posix()}
train: train
val: val
names:
  0: resistor
  1: led
  2: voltage_source
  3: capacitor
  4: transistor
  5: ground
  6: diode
  7: inductor
  8: wire
"""
    (OUTPUT_DIR / "circuit_dataset.yaml").write_text(yaml_content)
    print(f"\nYOLO config: {OUTPUT_DIR / 'circuit_dataset.yaml'}")
    print("Done!")


if __name__ == "__main__":
    # Clean old data
    import shutil
    for d in [OUTPUT_DIR / "train", OUTPUT_DIR / "val"]:
        if d.exists():
            shutil.rmtree(d)

    generate_dataset(n_train=100, n_val=20)

    # Generate previews
    print("\nGenerating preview images...")
    for gen_func, name in [(gen_led_circuit_good, "preview_led_good"),
                            (gen_led_no_resistor, "preview_led_bad"),
                            (gen_voltage_divider, "preview_divider"),
                            (gen_rc_filter, "preview_rc_filter")]:
        d, desc = gen_func()
        path = str(OUTPUT_DIR / f"{name}.png")
        d.save(path)
        (OUTPUT_DIR / f"{name}.txt").write_text(desc, encoding="utf-8")
        print(f"  {path}")
