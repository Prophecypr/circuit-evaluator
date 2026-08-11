# 标准电路参考包与双域采集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 10 张原创 ANSI 锯齿电阻标准参考图、机器可读答案、三人手绘采集目录和支持双人独立标注的盲测数据包。

**Architecture:** 用一个不可变的语义电路规范同时驱动参考图、答案文件和一致性校验，避免图与 GT 人工复制后漂移。Matplotlib 负责代码原生 SVG/PNG 绘制；现有 Flask 标注服务增加盲测根目录、标注者隔离和数值框字段，最终由第三人裁决为 `schema_version=1.0` 的冻结 GT。

**Tech Stack:** Python 3、dataclasses、Matplotlib、Pillow、Pydantic 2、Flask、原生 HTML Canvas、pytest。

---

## Dependency and file structure

先完成 `docs/superpowers/plans/2026-08-01-publication-evaluation-infrastructure.md` 的 Tasks 1–8，因为本计划复用 `BlindGroundTruth`、BJT 家族契约和冻结清单。

- Create: `src/reference_pack/__init__.py` — 公开电路规范和生成入口。
- Create: `src/reference_pack/specs.py` — 10 个电路的唯一语义来源。
- Create: `src/reference_pack/layouts.py` — 元件位置与网络汇合点。
- Create: `src/reference_pack/render.py` — ANSI 电路符号、连线和数值渲染。
- Create: `generate_blind_reference_pack.py` — 生成 SVG、PNG、答案和协议文件。
- Create: `tests/test_reference_specs.py` — 元件、端口、网络和难度数量校验。
- Create: `tests/test_reference_render.py` — 产物数量、无设计号、页面边界和可重复性校验。
- Modify: `benchmark/server.py` — 支持 `BENCHMARK_ROOT` 与 `ANNOTATOR_ID`，隔离盲测标注。
- Modify: `benchmark/annotation_tool.html` — 增加数值字符串与数值框编辑。
- Create: `tests/test_blind_annotation_server.py` — 标注者隔离、数值字段和 schema 输出测试。
- Generate later: `benchmark/blind_reference_pack/v1/` — 版本化参考包和空采集目录。

### Task 1: Encode the ten circuits as the single semantic source

**Files:**
- Create: `src/reference_pack/__init__.py`
- Create: `src/reference_pack/specs.py`
- Create: `tests/test_reference_specs.py`

- [ ] **Step 1: Write failing semantic-invariant tests**

Create `tests/test_reference_specs.py`:

```python
from collections import Counter

from src.reference_pack.specs import CIRCUITS, validate_circuit


def test_reference_pack_has_confirmed_difficulty_balance():
    assert list(CIRCUITS) == [f"C{i:02d}" for i in range(1, 11)]
    assert Counter(circuit.difficulty for circuit in CIRCUITS.values()) == {
        "basic": 4, "medium": 4, "challenge": 2,
    }


def test_every_port_belongs_to_exactly_one_net():
    for circuit in CIRCUITS.values():
        validate_circuit(circuit)


def test_resistors_use_ansi_and_values_fit_current_ocr_scope():
    allowed = set("0123456789.kKmMΩμunpFVAHz-+")
    for circuit in CIRCUITS.values():
        for component in circuit.components:
            if component.class_name == "resistor":
                assert component.symbol == "ansi_zigzag"
            if component.value:
                assert set(component.value) <= allowed


def test_c09_uses_generic_bjt_family_with_bce_ports():
    transistor = next(
        component for component in CIRCUITS["C09"].components
        if component.class_name == "transistor.bjt"
    )
    assert transistor.ports == ("B", "C", "E")
```

- [ ] **Step 2: Run tests and verify the package is absent**

Run:

```powershell
pytest tests/test_reference_specs.py -v
```

Expected: collection fails with `ModuleNotFoundError: src.reference_pack`.

- [ ] **Step 3: Implement immutable circuit data types and validation**

Create `src/reference_pack/specs.py` with:

```python
from dataclasses import dataclass
from typing import Literal


Difficulty = Literal["basic", "medium", "challenge"]


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    class_name: str
    value: str | None
    ports: tuple[str, ...]
    symbol: str


@dataclass(frozen=True)
class CircuitSpec:
    id: str
    title: str
    difficulty: Difficulty
    components: tuple[ComponentSpec, ...]
    nets: dict[str, tuple[str, ...]]


PORTS = {
    "resistor": ("1", "2"),
    "capacitor.unpolarized": ("1", "2"),
    "inductor": ("1", "2"),
    "diode.light_emitting": ("+", "-"),
    "diode.zener": ("A", "K"),
    "voltage.dc": ("+", "-"),
    "gnd": ("GND",),
    "transistor.bjt": ("B", "C", "E"),
}


def component(component_id, class_name, value=None):
    symbol = "ansi_zigzag" if class_name == "resistor" else class_name
    return ComponentSpec(component_id, class_name, value, PORTS[class_name], symbol)


def circuit(circuit_id, title, difficulty, components, nets):
    return CircuitSpec(
        circuit_id, title, difficulty, tuple(components),
        {net_id: tuple(refs) for net_id, refs in nets.items()},
    )


def validate_circuit(spec):
    ids = [item.id for item in spec.components]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{spec.id}: duplicate component id")
    declared = {
        f"{item.id}.{label}"
        for item in spec.components
        for label in item.ports
    }
    listed = [ref for refs in spec.nets.values() for ref in refs]
    if len(listed) != len(set(listed)):
        raise ValueError(f"{spec.id}: port appears in multiple nets")
    if declared != set(listed):
        raise ValueError(f"{spec.id}: semantic net partition mismatch")
    if any(len(refs) < 2 for refs in spec.nets.values()):
        raise ValueError(f"{spec.id}: every net needs at least two ports")
```

- [ ] **Step 4: Add all ten exact circuit definitions**

Append this dictionary to `src/reference_pack/specs.py`:

```python
CIRCUITS = {
    "C01": circuit("C01", "单电阻直流回路", "basic", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "1kΩ"),
        component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N0": ("R1.2", "V1.-", "GND.GND")}),
    "C02": circuit("C02", "双电阻串联", "basic", [
        component("V1", "voltage.dc", "9V"), component("R1", "resistor", "1kΩ"),
        component("R2", "resistor", "2.2kΩ"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "R2.1"),
        "N0": ("R2.2", "V1.-", "GND.GND")}),
    "C03": circuit("C03", "双电阻并联", "basic", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "1kΩ"),
        component("R2", "resistor", "10kΩ"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1", "R2.1"),
        "N0": ("V1.-", "R1.2", "R2.2", "GND.GND")}),
    "C04": circuit("C04", "LED限流回路", "basic", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "330Ω"),
        component("LED1", "diode.light_emitting"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "LED1.+"),
        "N0": ("LED1.-", "V1.-", "GND.GND")}),
    "C05": circuit("C05", "带负载电压分压器", "medium", [
        component("V1", "voltage.dc", "9V"), component("R1", "resistor", "1kΩ"),
        component("R2", "resistor", "2.2kΩ"), component("R3", "resistor", "10kΩ"),
        component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "R2.1", "R3.1"),
        "N0": ("R2.2", "R3.2", "V1.-", "GND.GND")}),
    "C06": circuit("C06", "带泄放电阻的RC网络", "medium", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "10kΩ"),
        component("R2", "resistor", "2.2kΩ"),
        component("C1", "capacitor.unpolarized", "10uF"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "R2.1", "C1.1"),
        "N0": ("R2.2", "C1.2", "V1.-", "GND.GND")}),
    "C07": circuit("C07", "RLC串联回路", "medium", [
        component("V1", "voltage.dc", "12V"), component("R1", "resistor", "100Ω"),
        component("L1", "inductor", "10mH"),
        component("C1", "capacitor.unpolarized", "1uF"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "L1.1"),
        "N3": ("L1.2", "C1.1"), "N0": ("C1.2", "V1.-", "GND.GND")}),
    "C08": circuit("C08", "稳压二极管并联稳压", "medium", [
        component("V1", "voltage.dc", "12V"), component("R1", "resistor", "1kΩ"),
        component("ZD1", "diode.zener"), component("R2", "resistor", "2.2kΩ"),
        component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1"), "N2": ("R1.2", "ZD1.K", "R2.1"),
        "N0": ("ZD1.A", "R2.2", "V1.-", "GND.GND")}),
    "C09": circuit("C09", "BJT-LED驱动电路", "challenge", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "330Ω"),
        component("R2", "resistor", "10kΩ"), component("LED1", "diode.light_emitting"),
        component("Q1", "transistor.bjt"), component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1", "R2.1"), "N2": ("R1.2", "LED1.+"),
        "N3": ("LED1.-", "Q1.C"), "N4": ("R2.2", "Q1.B"),
        "N0": ("Q1.E", "V1.-", "GND.GND")}),
    "C10": circuit("C10", "惠斯通电桥", "challenge", [
        component("V1", "voltage.dc", "5V"), component("R1", "resistor", "100Ω"),
        component("R2", "resistor", "220Ω"), component("R3", "resistor", "330Ω"),
        component("R4", "resistor", "470Ω"), component("R5", "resistor", "1kΩ"),
        component("GND", "gnd"),
    ], {"N1": ("V1.+", "R1.1", "R3.1"), "NL": ("R1.2", "R2.1", "R5.1"),
        "NR": ("R3.2", "R4.1", "R5.2"),
        "N0": ("R2.2", "R4.2", "V1.-", "GND.GND")}),
}

for _spec in CIRCUITS.values():
    validate_circuit(_spec)
```

Create `src/reference_pack/__init__.py`:

```python
from .specs import CIRCUITS, CircuitSpec, ComponentSpec, validate_circuit

__all__ = ["CIRCUITS", "CircuitSpec", "ComponentSpec", "validate_circuit"]
```

- [ ] **Step 5: Run and commit semantic tests**

```powershell
pytest tests/test_reference_specs.py -v
git add src/reference_pack tests/test_reference_specs.py
git commit -m "feat: define blind reference circuit semantics"
```

Expected: all tests pass before the commit is created.

### Task 2: Implement ANSI circuit-symbol rendering primitives

**Files:**
- Create: `src/reference_pack/render.py`
- Create: `tests/test_reference_render.py`

- [ ] **Step 1: Write a failing resistor-symbol and text test**

Create `tests/test_reference_render.py`:

```python
from src.reference_pack.render import ReferenceCanvas


def test_resistor_is_zigzag_and_reference_hides_designator(tmp_path):
    canvas = ReferenceCanvas("test")
    ports = canvas.component("R17", "resistor", "2.2kΩ", (500, 300), "h")
    output = tmp_path / "resistor.svg"
    canvas.save(output)
    svg = output.read_text(encoding="utf-8")

    assert ports == {"1": (420, 300), "2": (580, 300)}
    assert canvas.symbol_records["R17"] == "ansi_zigzag"
    assert canvas.text_records == ["2.2kΩ"]
    assert "2.2kΩ" in svg
    assert "R17" not in svg
```

- [ ] **Step 2: Run the test and verify the renderer is absent**

Run:

```powershell
pytest tests/test_reference_render.py -v
```

Expected: import failure for `src.reference_pack.render`.

- [ ] **Step 3: Build the fixed page and ANSI resistor primitive**

Create `src/reference_pack/render.py` with a `ReferenceCanvas` using a 1000×700 logical canvas, A4-landscape aspect, black `2.4` point strokes, white background, equal axes and inverted y-axis. Set `matplotlib.rcParams["svg.fonttype"] = "none"` so UTF-8 values remain searchable text in SVG. Initialize `self.symbol_records = {}` and `self.text_records = []`; `component()` records the symbol style by hidden component ID, records only visible value strings in `text_records`, and never renders the ID. The resistor branch must use:

```python
def _resistor(self, center, orientation, value):
    x, y = center
    if orientation == "h":
        points = [
            (x - 80, y), (x - 55, y), (x - 42, y - 18),
            (x - 24, y + 18), (x - 6, y - 18), (x + 12, y + 18),
            (x + 30, y - 18), (x + 48, y + 18), (x + 60, y), (x + 80, y),
        ]
        ports = {"1": (x - 80, y), "2": (x + 80, y)}
        text_at = (x, y - 42)
    else:
        points = [
            (x, y - 80), (x, y - 55), (x - 18, y - 42),
            (x + 18, y - 24), (x - 18, y - 6), (x + 18, y + 12),
            (x - 18, y + 30), (x + 18, y + 48), (x, y + 60), (x, y + 80),
        ]
        ports = {"1": (x, y - 80), "2": (x, y + 80)}
        text_at = (x + 38, y)
    self.ax.plot(*zip(*points), color="black", linewidth=2.4, solid_capstyle="round")
    if value:
        self.ax.text(*text_at, value, fontsize=15, ha="center", va="center")
        self.text_records.append(value)
    return ports
```

The `component()` resistor dispatch must also execute `self.symbol_records[component_id] = "ansi_zigzag"` before returning these ports.

- [ ] **Step 4: Add every symbol needed by C01–C10**

Implement dispatch branches for `voltage.dc`, `gnd`, `capacitor.unpolarized`, `inductor`, `diode.light_emitting`, `diode.zener`, and `transistor.bjt`. Use these exact port contracts:

```python
PORT_GEOMETRY = {
    ("voltage.dc", "v"): {"+": (0, -70), "-": (0, 70)},
    ("gnd", "v"): {"GND": (0, -35)},
    ("capacitor.unpolarized", "h"): {"1": (-70, 0), "2": (70, 0)},
    ("capacitor.unpolarized", "v"): {"1": (0, -70), "2": (0, 70)},
    ("inductor", "h"): {"1": (-80, 0), "2": (80, 0)},
    ("inductor", "v"): {"1": (0, -80), "2": (0, 80)},
    ("diode.light_emitting", "h"): {"+": (-70, 0), "-": (70, 0)},
    ("diode.light_emitting", "v"): {"+": (0, -70), "-": (0, 70)},
    ("diode.zener", "h"): {"A": (-70, 0), "K": (70, 0)},
    ("diode.zener", "v"): {"A": (0, 70), "K": (0, -70)},
    ("transistor.bjt", "v"): {"B": (-70, 0), "C": (45, -65), "E": (45, 65)},
}
```

Use `matplotlib.patches.Arc`, `Circle`, `FancyArrowPatch`, and `Polygon`. Translate the geometry table with:

```python
def _ports(self, class_name, orientation, center):
    x, y = center
    return {
        label: (x + dx, y + dy)
        for label, (dx, dy) in PORT_GEOMETRY[(class_name, orientation)].items()
    }
```

Implement the remaining drawing dispatch with these exact primitives:

```python
def _voltage(self, center, value):
    x, y = center
    ports = self._ports("voltage.dc", "v", center)
    self.ax.add_patch(Circle((x, y), 42, fill=False, color="black", linewidth=2.4))
    self.ax.plot([x, x], [ports["+"][1], y - 42], color="black", linewidth=2.4)
    self.ax.plot([x, x], [y + 42, ports["-"][1]], color="black", linewidth=2.4)
    self.ax.text(x, y - 17, "+", fontsize=17, ha="center", va="center")
    self.ax.text(x, y + 18, "−", fontsize=17, ha="center", va="center")
    if value:
        self.ax.text(x + 62, y, value, fontsize=15, ha="left", va="center")
        self.text_records.append(value)
    return ports


def _ground(self, center):
    x, y = center
    ports = self._ports("gnd", "v", center)
    self.ax.plot([x, x], [y - 35, y], color="black", linewidth=2.4)
    for offset, half_width in ((0, 34), (11, 23), (22, 11)):
        self.ax.plot([x - half_width, x + half_width], [y + offset, y + offset], color="black", linewidth=2.4)
    return ports


def _capacitor(self, center, orientation, value):
    x, y = center
    ports = self._ports("capacitor.unpolarized", orientation, center)
    if orientation == "h":
        self.ax.plot([x - 70, x - 12], [y, y], color="black", linewidth=2.4)
        self.ax.plot([x + 12, x + 70], [y, y], color="black", linewidth=2.4)
        self.ax.plot([x - 12, x - 12], [y - 38, y + 38], color="black", linewidth=2.4)
        self.ax.plot([x + 12, x + 12], [y - 38, y + 38], color="black", linewidth=2.4)
        text_at = (x, y - 58)
    else:
        self.ax.plot([x, x], [y - 70, y - 12], color="black", linewidth=2.4)
        self.ax.plot([x, x], [y + 12, y + 70], color="black", linewidth=2.4)
        self.ax.plot([x - 38, x + 38], [y - 12, y - 12], color="black", linewidth=2.4)
        self.ax.plot([x - 38, x + 38], [y + 12, y + 12], color="black", linewidth=2.4)
        text_at = (x + 56, y)
    if value:
        self.ax.text(*text_at, value, fontsize=15, ha="center", va="center")
        self.text_records.append(value)
    return ports


def _inductor(self, center, orientation, value):
    x, y = center
    ports = self._ports("inductor", orientation, center)
    if orientation == "h":
        self.ax.plot([x - 80, x - 48], [y, y], color="black", linewidth=2.4)
        self.ax.plot([x + 48, x + 80], [y, y], color="black", linewidth=2.4)
        for arc_x in (x - 36, x - 12, x + 12, x + 36):
            self.ax.add_patch(Arc((arc_x, y), 24, 34, theta1=180, theta2=360, color="black", linewidth=2.4))
        text_at = (x, y - 48)
    else:
        self.ax.plot([x, x], [y - 80, y - 48], color="black", linewidth=2.4)
        self.ax.plot([x, x], [y + 48, y + 80], color="black", linewidth=2.4)
        for arc_y in (y - 36, y - 12, y + 12, y + 36):
            self.ax.add_patch(Arc((x, arc_y), 34, 24, theta1=90, theta2=270, color="black", linewidth=2.4))
        text_at = (x + 52, y)
    if value:
        self.ax.text(*text_at, value, fontsize=15, ha="center", va="center")
        self.text_records.append(value)
    return ports


def _diode(self, class_name, center, orientation, led=False):
    ports = self._ports(class_name, orientation, center)
    x, y = center
    if orientation == "h":
        self.ax.plot([x - 70, x - 30], [y, y], color="black", linewidth=2.4)
        self.ax.add_patch(Polygon([(x - 30, y - 30), (x - 30, y + 30), (x + 20, y)], fill=False, color="black", linewidth=2.4))
        self.ax.plot([x + 22, x + 22], [y - 34, y + 34], color="black", linewidth=2.4)
        self.ax.plot([x + 22, x + 70], [y, y], color="black", linewidth=2.4)
        if class_name == "diode.zener":
            self.ax.plot([x + 12, x + 22, x + 32], [y - 42, y - 34, y - 42], color="black", linewidth=2.4)
            self.ax.plot([x + 12, x + 22, x + 32], [y + 42, y + 34, y + 42], color="black", linewidth=2.4)
        if led:
            for offset in (0, 14):
                self.ax.add_patch(FancyArrowPatch((x + offset, y - 35), (x + 32 + offset, y - 67), arrowstyle="->", mutation_scale=12, linewidth=1.8))
    else:
        self.ax.plot([x, x], [y - 70, y - 30], color="black", linewidth=2.4)
        self.ax.add_patch(Polygon([(x - 30, y - 30), (x + 30, y - 30), (x, y + 20)], fill=False, color="black", linewidth=2.4))
        self.ax.plot([x - 34, x + 34], [y + 22, y + 22], color="black", linewidth=2.4)
        self.ax.plot([x, x], [y + 22, y + 70], color="black", linewidth=2.4)
        if class_name == "diode.zener":
            self.ax.plot([x - 42, x - 34, x - 42], [y + 12, y + 22, y + 32], color="black", linewidth=2.4)
            self.ax.plot([x + 42, x + 34, x + 42], [y + 12, y + 22, y + 32], color="black", linewidth=2.4)
        if led:
            for offset in (0, 14):
                self.ax.add_patch(FancyArrowPatch((x + 25, y - offset), (x + 58, y - 33 - offset), arrowstyle="->", mutation_scale=12, linewidth=1.8))
    return ports


def _bjt(self, center):
    x, y = center
    ports = self._ports("transistor.bjt", "v", center)
    self.ax.plot([x - 70, x - 18], [y, y], color="black", linewidth=2.4)
    self.ax.plot([x - 18, x - 18], [y - 45, y + 45], color="black", linewidth=2.4)
    self.ax.plot([x - 18, x + 45], [y - 28, y - 65], color="black", linewidth=2.4)
    self.ax.plot([x - 18, x + 45], [y + 28, y + 65], color="black", linewidth=2.4)
    self.ax.add_patch(FancyArrowPatch((x + 18, y + 48), (x + 43, y + 64), arrowstyle="->", mutation_scale=13, linewidth=1.8))
    return ports
```

`component()` must dispatch all eight supported classes to these methods and raise `ValueError` for any other class. `save(path)` must call `figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)` and close the figure.

- [ ] **Step 5: Run the renderer test and visually inspect the SVG**

```powershell
pytest tests/test_reference_render.py -v
```

Expected: the test passes. Open the generated SVG from pytest's temporary output only if debugging is needed; do not commit temporary files.

- [ ] **Step 6: Commit rendering primitives**

```powershell
git add src/reference_pack/render.py tests/test_reference_render.py
git commit -m "feat: render standard ANSI circuit symbols"
```

### Task 3: Define deterministic layouts and net routing

**Files:**
- Create: `src/reference_pack/layouts.py`
- Modify: `src/reference_pack/render.py`
- Modify: `tests/test_reference_render.py`

- [ ] **Step 1: Write layout-completeness tests**

Append:

```python
from src.reference_pack.layouts import LAYOUTS, validate_layout
from src.reference_pack.specs import CIRCUITS


def test_every_semantic_component_and_net_has_layout_data():
    for circuit_id, circuit in CIRCUITS.items():
        validate_layout(circuit, LAYOUTS[circuit_id])


def test_layouts_keep_symbols_inside_page():
    for layout in LAYOUTS.values():
        for x, y, orientation in layout["components"].values():
            assert 100 <= x <= 900
            assert 100 <= y <= 600
            assert orientation in {"h", "v"}
```

- [ ] **Step 2: Define all component positions and net hubs**

Create `src/reference_pack/layouts.py` with the following complete layout table:

```python
LAYOUTS = {
    "C01": {"components": {"V1": (180, 350, "v"), "R1": (500, 160, "h"), "GND": (500, 560, "v")},
            "hubs": {"N1": (300, 160), "N0": (500, 500)}},
    "C02": {"components": {"V1": (150, 350, "v"), "R1": (390, 150, "h"), "R2": (690, 150, "h"), "GND": (500, 560, "v")},
            "hubs": {"N1": (260, 150), "N2": (540, 150), "N0": (500, 500)}},
    "C03": {"components": {"V1": (150, 350, "v"), "R1": (440, 250, "h"), "R2": (440, 450, "h"), "GND": (760, 560, "v")},
            "hubs": {"N1": (280, 350), "N0": (720, 500)}},
    "C04": {"components": {"V1": (150, 350, "v"), "R1": (400, 160, "h"), "LED1": (700, 350, "v"), "GND": (500, 560, "v")},
            "hubs": {"N1": (270, 160), "N2": (700, 220), "N0": (500, 500)}},
    "C05": {"components": {"V1": (140, 350, "v"), "R1": (380, 150, "h"), "R2": (610, 320, "v"), "R3": (790, 320, "v"), "GND": (610, 580, "v")},
            "hubs": {"N1": (260, 150), "N2": (610, 210), "N0": (610, 520)}},
    "C06": {"components": {"V1": (140, 350, "v"), "R1": (380, 150, "h"), "R2": (610, 340, "v"), "C1": (790, 340, "v"), "GND": (610, 580, "v")},
            "hubs": {"N1": (260, 150), "N2": (610, 230), "N0": (610, 520)}},
    "C07": {"components": {"V1": (120, 350, "v"), "R1": (320, 150, "h"), "L1": (570, 150, "h"), "C1": (800, 350, "v"), "GND": (500, 580, "v")},
            "hubs": {"N1": (220, 150), "N2": (445, 150), "N3": (720, 150), "N0": (500, 520)}},
    "C08": {"components": {"V1": (130, 350, "v"), "R1": (370, 150, "h"), "ZD1": (610, 340, "v"), "R2": (800, 340, "v"), "GND": (610, 580, "v")},
            "hubs": {"N1": (250, 150), "N2": (610, 220), "N0": (610, 520)}},
    "C09": {"components": {"V1": (110, 350, "v"), "R1": (340, 140, "h"), "LED1": (610, 245, "v"), "Q1": (650, 420, "v"), "R2": (350, 360, "h"), "GND": (620, 600, "v")},
            "hubs": {"N1": (240, 220), "N2": (610, 160), "N3": (650, 330), "N4": (500, 420), "N0": (620, 550)}},
    "C10": {"components": {"V1": (100, 350, "v"), "R1": (360, 200, "v"), "R2": (360, 480, "v"), "R3": (720, 200, "v"), "R4": (720, 480, "v"), "R5": (540, 350, "h"), "GND": (540, 620, "v")},
            "hubs": {"N1": (540, 100), "NL": (360, 350), "NR": (720, 350), "N0": (540, 570)}},
}


def validate_layout(circuit, layout):
    component_ids = {component.id for component in circuit.components}
    if set(layout["components"]) != component_ids:
        raise ValueError(f"{circuit.id}: component layout mismatch")
    if set(layout["hubs"]) != set(circuit.nets):
        raise ValueError(f"{circuit.id}: net hub mismatch")
```

- [ ] **Step 3: Implement semantic star routing**

Add `draw_reference(circuit, layout)` to `src/reference_pack/render.py`. It must draw all components first and store `port_ref -> (x,y)`. For each semantic net, connect every member port to the declared hub with an orthogonal path `port -> (hub_x, port_y) -> hub`, remove consecutive duplicate points, draw the hub dot only when the net has three or more members, and set the title to `f"{circuit.id}  {circuit.title}"`. Return the canvas so tests can inspect its port map.

- [ ] **Step 4: Run layout and rendering tests**

```powershell
pytest tests/test_reference_specs.py tests/test_reference_render.py -v
```

Expected: all 10 layouts pass semantic coverage and page-boundary checks.

- [ ] **Step 5: Commit layouts**

```powershell
git add src/reference_pack/layouts.py src/reference_pack/render.py tests/test_reference_render.py
git commit -m "feat: lay out ten blind reference circuits"
```

### Task 4: Generate versioned references, answer keys and protocol files

**Files:**
- Create: `generate_blind_reference_pack.py`
- Modify: `tests/test_reference_render.py`
- Generate: `benchmark/blind_reference_pack/v1/`

- [ ] **Step 1: Write a failing pack-generation test**

Append a test that calls `generate_pack(tmp_path / "v1")` and asserts:

```python
assert len(list((output / "references").glob("C*_reference.svg"))) == 10
assert len(list((output / "references").glob("C*_reference.png"))) == 10
assert len(list((output / "answer_key").glob("C*_gt.json"))) == 10
assert (output / "collection_protocol.md").is_file()
assert (output / "annotation_protocol.md").is_file()
assert (output / "manifest_template.csv").is_file()
assert (output / "checksums.sha256").is_file()
```

Read every SVG and assert that component designators such as `R1`, `C1`, `V1`, `Q1`, `LED1` are absent while every non-null semantic value for that circuit is present.

- [ ] **Step 2: Implement `generate_pack()`**

Create `generate_blind_reference_pack.py` with:

```python
def generate_pack(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pack: {output_dir}")
    references = output_dir / "references"
    answer_key = output_dir / "answer_key"
    for directory in (references, answer_key):
        directory.mkdir(parents=True, exist_ok=True)

    for circuit_id, spec in CIRCUITS.items():
        canvas = draw_reference(spec, LAYOUTS[circuit_id])
        canvas.save(references / f"{circuit_id}_reference.svg")
        canvas = draw_reference(spec, LAYOUTS[circuit_id])
        canvas.save(references / f"{circuit_id}_reference.png")
        answer = {
            "circuit_id": circuit_id,
            "title": spec.title,
            "difficulty": spec.difficulty,
            "components": [asdict(component) for component in spec.components],
            "nets": spec.nets,
        }
        (answer_key / f"{circuit_id}_gt.json").write_text(
            json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    write_protocols(output_dir)
    write_checksums(output_dir)
    return output_dir
```

Implement `write_checksums()` using `src.evaluation.freeze.sha256_file`. `manifest_template.csv` must contain exactly the header:

```text
paper_id,circuit_id,participant_id,difficulty,domain,image_path,gt_path
```

The CLI must require `--output-dir` and default nowhere, so an accidental run cannot overwrite a prior pack.

- [ ] **Step 3: Put the confirmed collection rules into generated Markdown**

`collection_protocol.md` must state: 3 people × 10 circuits, blank A4, one circuit per page, black/dark pen, no ruler, no tracing, ANSI resistor, values only, visible junction dots, no model feedback, and pre-model redraw criteria. `annotation_protocol.md` must state: two independent annotators, third-person adjudication, actual visible drawing is labeled, IoU/ports/values/nets are all required, and no post-model GT edits.

- [ ] **Step 4: Run tests and generate the repository pack**

```powershell
pytest tests/test_reference_specs.py tests/test_reference_render.py -v
python generate_blind_reference_pack.py --output-dir benchmark/blind_reference_pack/v1
```

Expected: 10 SVG, 10 PNG, 10 answer JSON files and four protocol/manifest/checksum files are created in a previously absent directory.

- [ ] **Step 5: Commit the generator and generated reference pack**

```powershell
git add generate_blind_reference_pack.py src/reference_pack tests/test_reference_render.py benchmark/blind_reference_pack/v1
git commit -m "feat: generate blind circuit reference pack"
```

### Task 5: Perform machine and visual quality assurance on all references

**Files:**
- Modify only if verification exposes a concrete drawing defect.

- [ ] **Step 1: Run deterministic regeneration comparison**

Generate a second pack in a temporary directory, compare all answer JSON and SVG files byte-for-byte, and compare PNG SHA-256 values. Expected: all files match the committed pack.

- [ ] **Step 2: Render a contact sheet**

Use Matplotlib to assemble the 10 generated PNG files into a `5 × 2` contact sheet under a temporary directory. The contact sheet is for inspection only and must not be committed.

- [ ] **Step 3: Inspect every circuit against the answer key**

For C01–C10 verify: symbol count, ANSI resistor shape, value string, polarities, wire membership, junction dots, no unintentional crossing, no clipped content and no visible designator. Record one pass/fail row per circuit in the implementation handoff.

- [ ] **Step 4: Correct and re-verify any concrete defect**

Change only the corresponding `LAYOUTS` coordinate, hub, or rendering primitive; add a regression assertion for that defect; regenerate the pack in a new empty directory; replace only the generated files whose checksum changed; rerun Tasks 4 and 5 tests.

- [ ] **Step 5: Commit verified visual corrections if any**

```powershell
git add src/reference_pack tests/test_reference_render.py benchmark/blind_reference_pack/v1
git commit -m "fix: verify blind reference drawing layout"
```

If no source or generated file changes, do not create an empty commit.

### Task 6: Prepare controlled and handheld collection slots

**Files:**
- Create: `src/reference_pack/collection.py`
- Create: `tests/test_collection_manifest.py`
- Generate: `benchmark/blind_reference_pack/v1/collection/`

- [ ] **Step 1: Write the expected 60-row manifest test**

Create `tests/test_collection_manifest.py`:

```python
import csv

from src.reference_pack.collection import prepare_collection


def test_collection_has_thirty_papers_and_two_domains(tmp_path):
    manifest = prepare_collection(tmp_path, participants=("P01", "P02", "P03"))
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert len(rows) == 60
    assert len({row["paper_id"] for row in rows}) == 30
    assert {row["domain"] for row in rows} == {"controlled", "handheld"}
    assert all(row["image_path"].endswith(f"_{row['domain']}.jpg") for row in rows)
```

- [ ] **Step 2: Implement collection preparation**

Create `src/reference_pack/collection.py`. `prepare_collection(root, participants)` must create `images/controlled`, `images/handheld`, `annotations/A01`, `annotations/A02`, `annotations/ADJ`, `adjudicated`, and `logs`; write 60 rows ordered by circuit, participant, domain; derive difficulty from `CIRCUITS`; and use `adjudicated/{paper_id}_{domain}.json` as `gt_path`. Refuse non-empty roots.

- [ ] **Step 3: Add capture-log columns**

Create `logs/capture_log.csv` with columns:

```text
paper_id,circuit_id,participant_id,controlled_filename,handheld_filename,phone_model,width,height,captured_at,controlled_qc,handheld_qc,redraw_reason,notes
```

Create one row for every `paper_id` with empty capture fields and fixed IDs.

- [ ] **Step 4: Run tests and prepare collection directories**

```powershell
pytest tests/test_collection_manifest.py -v
python -c "from pathlib import Path; from src.reference_pack.collection import prepare_collection; prepare_collection(Path('benchmark/blind_reference_pack/v1/collection'), ('P01','P02','P03'))"
```

Expected: a 60-row manifest and 30-row capture log are created; image slots remain empty until human capture.

- [ ] **Step 5: Commit collection scaffolding**

```powershell
git add src/reference_pack/collection.py tests/test_collection_manifest.py benchmark/blind_reference_pack/v1/collection
git commit -m "feat: scaffold paired blind data collection"
```

### Task 7: Extend the existing annotation tool for values and annotator isolation

**Files:**
- Modify: `benchmark/server.py`
- Modify: `benchmark/annotation_tool.html`
- Create: `tests/test_blind_annotation_server.py`

- [ ] **Step 1: Write server isolation tests**

Create `tests/test_blind_annotation_server.py`. Import `benchmark.server`, monkeypatch `ROOT`, `DET_DIR`, `RESULT_DIR`, `FIX_DIR`, and a new `ANNOTATION_DIR` to `tmp_path`; use Flask's test client; post the same `sample.json` for annotators `A01` and `A02`; assert the two writes land in separate directories and neither overwrites the other. Post a component with:

```python
"value_gt": {"raw": "2.2 KΩ", "normalized": "2.2kΩ", "xyxy": [10, 20, 80, 45]}
```

Assert the saved JSON preserves all three fields.

- [ ] **Step 2: Add environment-scoped annotation paths**

At server startup, replace the fixed root with:

```python
APP_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BENCHMARK_ROOT", APP_ROOT)).resolve()
ANNOTATOR_ID = os.environ.get("ANNOTATOR_ID", "default")
if not ANNOTATOR_ID.replace("_", "").isalnum():
    raise ValueError("ANNOTATOR_ID must be alphanumeric with optional underscores")
DET_DIR = ROOT / "detections"
RESULT_DIR = ROOT / "result"
FIX_DIR = ROOT / "fixed"
ANNOTATION_DIR = ROOT / "annotations" / ANNOTATOR_ID
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
```

Construct Flask's `static_folder` from `ROOT`, but keep the index route as `send_file(APP_ROOT / "annotation_tool.html")` so the UI remains available when `BENCHMARK_ROOT` points at the collection directory. In blind mode, `/api/images` must recursively return `images/controlled/*` and `images/handheld/*` as paths relative to `ROOT`. Add `/api/annotation/<path:filename>` to read only the current `ANNOTATION_DIR`, and add `/api/save_annotation` to sanitize with `Path(filename).name` and write UTF-8 indented JSON to that directory. Never read another annotator's directory and never write adjudicated GT from these endpoints.

- [ ] **Step 3: Add value editing controls to the component editor**

Add one text input `ed_value`, one text input `ed_value_normalized`, and a `Set value box` button. Store the annotation as:

```javascript
edC.value_gt = {
  raw: document.getElementById('ed_value').value,
  normalized: document.getElementById('ed_value_normalized').value,
  xyxy: edC.value_gt && edC.value_gt.xyxy ? edC.value_gt.xyxy : null
};
```

The button enters a two-click mode: first click stores the top-left candidate, second click sorts both coordinates into `[x1,y1,x2,y2]`. Draw the value box in cyan and render the normalized string above it. Reject saving a non-empty value whose box is null.

- [ ] **Step 4: Save the complete blind annotation JSON**

Change the blind-save button to build and POST this exact shape to `/api/save_annotation`:

```javascript
const blindData = {
  image: data.image,
  width: img.width,
  height: img.height,
  components: data.components.map(c => ({
    designator: c.designator,
    name: c.name,
    xyxy: c.xyxy,
    ports: c.ports,
    labels: c.labels,
    value_gt: c.value_gt || null
  })),
  groups: groups.map(g => ({
    members: g.m.map(m => ({component_id: m.name, label: m.label}))
  }))
};
fetch('/api/save_annotation', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({filename: data.image.replace(/\.[^.]+$/, '') + '.json', data: blindData})
});
```

Keep the existing `/api/save` GT-text endpoint unchanged for the historical benchmark.

Change `load(n)` so it first requests the current annotator's saved JSON, then the historical detection JSON, and finally starts from an empty manual annotation:

```javascript
let key = n.replace(/\.[^.]+$/, '') + '.json';
let saved = await fetch('/api/annotation/' + encodeURIComponent(key));
if (saved.ok) {
  data = await saved.json();
} else {
  let detected = await fetch('/detections/' + key);
  data = detected.ok ? await detected.json() : {image: n, components: [], groups: []};
}
data.image = n;
groups = (data.groups || []).map(g => ({
  m: g.members.map(member => {
    let ci = data.components.findIndex(c => c.designator === member.component_id);
    return {ci: ci, pi: data.components[ci].labels.indexOf(member.label), name: member.component_id, label: member.label};
  })
}));
```

For the blind collection, leave `detections/` empty so both annotators work from the visible image rather than model pre-annotations.

- [ ] **Step 5: Run server and historical tool tests**

```powershell
pytest tests/test_blind_annotation_server.py tests/test_wiring_experiments.py -v
```

Expected: annotator writes are isolated and historical experiment tests remain green.

- [ ] **Step 6: Commit annotation support**

```powershell
git add benchmark/server.py benchmark/annotation_tool.html tests/test_blind_annotation_server.py
git commit -m "feat: add blind value annotation workflow"
```

### Task 8: Convert annotations and produce a disagreement report

**Files:**
- Create: `src/reference_pack/adjudication.py`
- Create: `tests/test_adjudication.py`

- [ ] **Step 1: Write conversion and disagreement tests**

Create `tests/test_adjudication.py` with one two-component annotation using this saved UI shape:

```python
annotation = {
    "image": "C01_P01_controlled.jpg", "width": 1000, "height": 700,
    "components": [
        {"designator": "R1", "name": "Resistor", "xyxy": [100, 100, 300, 180],
         "ports": [[100, 140], [300, 140]], "labels": ["1", "2"],
         "value_gt": {"raw": "1kΩ", "normalized": "1kΩ", "xyxy": [130, 60, 220, 95]}},
        {"designator": "V1", "name": "V-DC", "xyxy": [400, 200, 500, 300],
         "ports": [[450, 200], [450, 300]], "labels": ["+", "-"],
         "value_gt": {"raw": "5V", "normalized": "5V", "xyxy": [505, 220, 560, 250]}},
    ],
    "groups": [
        {"members": [{"component_id": "R1", "label": "1"}, {"component_id": "V1", "label": "+"}]},
        {"members": [{"component_id": "R1", "label": "2"}, {"component_id": "V1", "label": "-"}]},
    ],
}
```

Create a `ManifestRow` for `C01_P01/controlled`, call `annotation_to_ground_truth()`, and assert it returns a valid `BlindGroundTruth` with raw classes `resistor` and `voltage.dc`. Deep-copy the annotation, change `R1.value_gt.normalized` from `1kΩ` to `10kΩ`, call `compare_annotations()`, and assert exactly one `value_mismatch` row names `R1`. Call `agreement_metrics()` and assert `class_exact=2`, `bbox_iou_sum=2.0`, `value_exact=1`, `value_total=2`, `port_distance_sum=0.0`, and `net_exact=1`.

- [ ] **Step 2: Implement strict UI-to-GT conversion**

Create `src/reference_pack/adjudication.py`:

```python
import csv
import json
import math
from pathlib import Path

from src.evaluation.metrics import box_iou
from src.evaluation.schema import BlindGroundTruth, load_manifest


RAW_CLASS = {
    "Resistor": "resistor",
    "Capacitor": "capacitor.unpolarized",
    "Inductor": "inductor",
    "Diode": "diode",
    "LED": "diode.light_emitting",
    "Zener-Diode": "diode.zener",
    "V-DC": "voltage.dc",
    "GND": "gnd",
    "BJT": "transistor.bjt",
}


def annotation_to_ground_truth(annotation, manifest_row):
    ref_to_net = {}
    nets = {}
    for index, group in enumerate(annotation["groups"], 1):
        net_id = f"N{index}"
        refs = [f"{member['component_id']}.{member['label']}" for member in group["members"]]
        if len(refs) < 2:
            raise ValueError(f"{net_id} has fewer than two ports")
        nets[net_id] = refs
        for ref in refs:
            if ref in ref_to_net:
                raise ValueError(f"{ref} appears in multiple groups")
            ref_to_net[ref] = net_id

    components = []
    for item in annotation["components"]:
        component_id = item["designator"]
        ports = []
        for label, point in zip(item["labels"], item["ports"], strict=True):
            ref = f"{component_id}.{label}"
            if ref not in ref_to_net:
                raise ValueError(f"unassigned port: {ref}")
            ports.append({"label": label, "xy": point, "net_id": ref_to_net[ref]})
        value_gt = item.get("value_gt")
        value = None
        if value_gt and value_gt.get("raw"):
            if not value_gt.get("normalized") or not value_gt.get("xyxy"):
                raise ValueError(f"incomplete value annotation: {component_id}")
            value = value_gt
        components.append({
            "id": component_id,
            "class_name": RAW_CLASS[item["name"]],
            "xyxy": item["xyxy"],
            "value": value,
            "ports": ports,
        })

    return BlindGroundTruth.model_validate({
        "schema_version": "1.0",
        "image": {
            "paper_id": manifest_row.paper_id,
            "circuit_id": manifest_row.circuit_id,
            "participant_id": manifest_row.participant_id,
            "domain": manifest_row.domain,
            "width": annotation["width"],
            "height": annotation["height"],
        },
        "components": components,
        "nets": nets,
    })
```

- [ ] **Step 3: Implement deterministic disagreement extraction**

Append:

```python
def compare_annotations(left, right):
    disagreements = []
    left_components = {item["designator"]: item for item in left["components"]}
    right_components = {item["designator"]: item for item in right["components"]}
    for component_id in sorted(set(left_components) | set(right_components)):
        if component_id not in left_components or component_id not in right_components:
            disagreements.append({"component_id": component_id, "kind": "component_missing"})
            continue
        first, second = left_components[component_id], right_components[component_id]
        if first["name"] != second["name"]:
            disagreements.append({"component_id": component_id, "kind": "class_mismatch"})
        if box_iou(first["xyxy"], second["xyxy"]) < 0.90:
            disagreements.append({"component_id": component_id, "kind": "bbox_iou_below_090"})
        first_value = (first.get("value_gt") or {}).get("normalized", "")
        second_value = (second.get("value_gt") or {}).get("normalized", "")
        if first_value != second_value:
            disagreements.append({"component_id": component_id, "kind": "value_mismatch"})
        if first.get("labels") != second.get("labels"):
            disagreements.append({"component_id": component_id, "kind": "port_label_mismatch"})
        else:
            diagonal = math.hypot(left["width"], left["height"])
            for label, point_a, point_b in zip(first["labels"], first["ports"], second["ports"], strict=True):
                if math.dist(point_a, point_b) > 0.01 * diagonal:
                    disagreements.append({"component_id": component_id, "kind": f"port_distance:{label}"})

    def group_set(annotation):
        return {
            frozenset(f"{member['component_id']}.{member['label']}" for member in group["members"])
            for group in annotation["groups"]
        }

    if group_set(left) != group_set(right):
        disagreements.append({"component_id": "*", "kind": "net_mismatch"})
    return disagreements


def agreement_metrics(left, right):
    first = {item["designator"]: item for item in left["components"]}
    second = {item["designator"]: item for item in right["components"]}
    common = sorted(set(first) & set(second))
    diagonal = math.hypot(left["width"], left["height"])
    box_overlaps = [box_iou(first[item]["xyxy"], second[item]["xyxy"]) for item in common]
    value_components = [
        item for item in common
        if (first[item].get("value_gt") or {}).get("raw")
        or (second[item].get("value_gt") or {}).get("raw")
    ]
    port_distances = [
        math.dist(point_a, point_b) / diagonal
        for item in common
        if first[item].get("labels") == second[item].get("labels")
        for point_a, point_b in zip(first[item]["ports"], second[item]["ports"], strict=True)
    ]
    def groups(annotation):
        return {
            frozenset(f"{member['component_id']}.{member['label']}" for member in group["members"])
            for group in annotation["groups"]
        }
    return {
        "component_count_a": len(first),
        "component_count_b": len(second),
        "common_components": len(common),
        "class_exact": sum(first[item]["name"] == second[item]["name"] for item in common),
        "bbox_iou_sum": sum(box_overlaps),
        "bbox_iou_count": len(box_overlaps),
        "value_exact": sum(
            (first[item].get("value_gt") or {}).get("normalized", "")
            == (second[item].get("value_gt") or {}).get("normalized", "")
            for item in value_components
        ),
        "value_total": len(value_components),
        "port_distance_sum": sum(port_distances),
        "port_distance_count": len(port_distances),
        "net_exact": int(groups(left) == groups(right)),
    }
```

Add `write_disagreement_csv(left_dir, right_dir, output_csv, summary_json)` that compares same-named JSON files, writes columns `image,component_id,kind`, raises if either directory lacks a counterpart, sums the count fields returned by `agreement_metrics()`, and writes class agreement, mean bbox IoU, value exact agreement, mean normalized port distance and net exact agreement to `summary_json`. Add `convert_adjudicated(raw_dir, manifest_path)` that loads the manifest, opens a same-stem raw JSON for every row, calls `annotation_to_ground_truth()`, and writes `model_dump_json(indent=2)` to each row's declared `gt_path` while refusing to overwrite an existing GT.

- [ ] **Step 4: Run adjudication tests**

```powershell
pytest tests/test_adjudication.py tests/test_blind_schema.py -v
```

Expected: conversion produces valid schema and the changed value produces exactly one disagreement.

- [ ] **Step 5: Commit the adjudication tooling**

```powershell
git add src/reference_pack/adjudication.py tests/test_adjudication.py
git commit -m "feat: add blind annotation adjudication"
```

### Task 9: Collect, double-annotate and adjudicate the dataset

**Files:**
- Populate: `benchmark/blind_reference_pack/v1/collection/images/controlled/`
- Populate: `benchmark/blind_reference_pack/v1/collection/images/handheld/`
- Populate: `benchmark/blind_reference_pack/v1/collection/annotations/A01/`
- Populate: `benchmark/blind_reference_pack/v1/collection/annotations/A02/`
- Populate: `benchmark/blind_reference_pack/v1/collection/annotations/ADJ/`
- Populate: `benchmark/blind_reference_pack/v1/collection/adjudicated/`

- [ ] **Step 1: Complete all 30 paper drawings before model access**

P01–P03 each draw C01–C10 from the committed references. Perform quality control only against the reference diagrams. Record every redraw and reason in `capture_log.csv`.

- [ ] **Step 2: Capture both domains with one device and one operator**

For each paper, take `controlled` and `handheld` images using the confirmed protocol and exact manifest filenames. Fill phone model, resolution, capture time and QC columns. Do not run `process_image` at this stage.

- [ ] **Step 3: Annotate independently**

Run the server twice with distinct `ANNOTATOR_ID=A01` and `ANNOTATOR_ID=A02`. Both annotators label all 60 images without reading the other's files. Each must correct class, component box, ports, port labels, value raw/normalized/box and connection groups.

- [ ] **Step 4: Generate the adjudication disagreement list**

Run `write_disagreement_csv()` on `annotations/A01` and `annotations/A02`. The third person resolves every listed class, value, bbox, port and net field while viewing the image, then saves exactly one raw adjudication JSON per manifest row under `annotations/ADJ`.

- [ ] **Step 5: Convert and validate all 60 adjudicated GT files**

Run `convert_adjudicated()` from `annotations/ADJ` into the manifest-declared `adjudicated/` paths, then execute:

```powershell
python -c "from pathlib import Path; from src.evaluation.schema import load_manifest, load_ground_truth; rows=load_manifest(Path('benchmark/blind_reference_pack/v1/collection/manifest.csv'), require_pairs=True); [load_ground_truth(r.gt_path) for r in rows]; print(len(rows), len({r.paper_id for r in rows}))"
```

Expected: `60 30` and no validation exception.

- [ ] **Step 6: Commit only non-sensitive benchmark metadata**

Before collection, add these exact entries to `.gitignore` so participant handwriting and intermediate annotations remain local:

```text
benchmark/blind_reference_pack/v1/collection/images/
benchmark/blind_reference_pack/v1/collection/annotations/
benchmark/blind_reference_pack/v1/collection/adjudicated/
```

Commit references, protocols, schema, manifest template, empty collection manifest/capture log and `.gitignore`. Record the exact local collection path in the implementation handoff without copying images elsewhere.

### Task 10: Export YOLO test splits and freeze every blind input

**Files:**
- Create: `src/reference_pack/yolo_export.py`
- Create: `tests/test_yolo_export.py`
- Populate: `benchmark/blind_reference_pack/v1/collection/yolo/`
- Create: `benchmark/blind_reference_pack/v1/collection/freeze_manifest.json`

- [ ] **Step 1: Write a two-domain YOLO export test**

Create `tests/test_yolo_export.py` with one controlled and one handheld manifest row sharing a `paper_id`, valid GT containing a `resistor` box `[100,100,300,200]` in a `1000×500` image, and `class_names={0: "resistor"}`. Call `export_yolo()` and assert:

```python
assert (output / "controlled/images/C01_P01_controlled.jpg").is_file()
assert (output / "handheld/images/C01_P01_handheld.jpg").is_file()
assert (output / "controlled/labels/C01_P01_controlled.txt").read_text().strip() == "0 0.200000 0.300000 0.200000 0.200000"
assert (output / "controlled.yaml").is_file()
assert (output / "handheld.yaml").is_file()
```

- [ ] **Step 2: Implement the YOLO export**

Create `src/reference_pack/yolo_export.py`:

```python
import json
import shutil
from pathlib import Path

from src.evaluation.schema import load_ground_truth, load_manifest


def export_yolo(manifest_path, class_names, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite YOLO export: {output_dir}")
    normalized_names = {int(index): name for index, name in class_names.items()}
    if sorted(normalized_names) != list(range(len(normalized_names))):
        raise ValueError("detector class ids must be contiguous from zero")
    inverse = {name: index for index, name in normalized_names.items()}
    rows = load_manifest(manifest_path, require_pairs=True)

    for domain in ("controlled", "handheld"):
        image_dir = output_dir / domain / "images"
        label_dir = output_dir / domain / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for row in [item for item in rows if item.domain == domain]:
            gt = load_ground_truth(row.gt_path)
            destination = image_dir / row.image_path.name
            shutil.copy2(row.image_path, destination)
            labels = []
            for component in gt.components:
                if component.class_name not in inverse:
                    raise ValueError(f"model has no class for {component.class_name}")
                x1, y1, x2, y2 = component.xyxy
                width, height = gt.image.width, gt.image.height
                labels.append(
                    f"{inverse[component.class_name]} "
                    f"{(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
                    f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                )
            (label_dir / f"{row.image_path.stem}.txt").write_text(
                "\n".join(labels) + "\n", encoding="utf-8",
            )

        yaml_text = (
            f"path: {json.dumps(str((output_dir / domain).resolve()))}\n"
            "test: images\n"
            f"names: {json.dumps([normalized_names[index] for index in range(len(normalized_names))], ensure_ascii=False)}\n"
        )
        (output_dir / f"{domain}.yaml").write_text(yaml_text, encoding="utf-8")
    return output_dir
```

- [ ] **Step 3: Run YOLO export tests**

```powershell
pytest tests/test_yolo_export.py tests/test_blind_schema.py -v
```

Expected: both domains receive isolated image/label trees and the normalized label is exact.

- [ ] **Step 4: Export the real adjudicated set without running inference**

Before export, add these exact entries to `.gitignore`:

```text
benchmark/blind_reference_pack/v1/collection/yolo/
benchmark/blind_reference_pack/v1/collection/freeze_manifest.json
```

Load detector class names with `YOLO('runs/detect/cghd_61cls/weights/best.pt').names`, call `export_yolo()` for the collection manifest, and verify each domain contains 30 images and 30 labels. Do not call `.val()` yet.

- [ ] **Step 5: Build and immediately verify the freeze manifest**

Use `write_freeze_manifest()` from the infrastructure plan with:

```python
rows = load_manifest("benchmark/blind_reference_pack/v1/collection/manifest.csv", require_pairs=True)
data_files = [Path("benchmark/blind_reference_pack/v1/collection/manifest.csv")]
data_files += [row.image_path for row in rows]
data_files += [row.gt_path for row in rows]
data_files += [path for path in Path("benchmark/blind_reference_pack/v1/collection/yolo").rglob("*") if path.is_file()]
models = [
    Path("runs/detect/cghd_61cls/weights/best.pt"),
    Path("runs/ocr_crnn_machine/best.pt"),
]
write_freeze_manifest(
    "benchmark/blind_reference_pack/v1/collection/freeze_manifest.json",
    files=data_files,
    models=models,
    config=build_ablation_configs()["Ours"],
    git_revision=get_git_revision(),
)
verify_freeze_manifest(load_freeze_manifest(
    "benchmark/blind_reference_pack/v1/collection/freeze_manifest.json"
))
```

Expected: verification succeeds before either `run_blind_benchmark.py` or `run_detector_eval.py` is invoked.

- [ ] **Step 6: Commit exporter code, not local participant data**

```powershell
git add src/reference_pack/yolo_export.py tests/test_yolo_export.py .gitignore
git commit -m "feat: export frozen detector test splits"
```

Keep `collection/yolo/` and `freeze_manifest.json` local because they contain copies and hashes of participant data. Verify `git status --short` does not list either path after the Step 4 ignore rules are present.

### Task 11: Run the frozen development and blind experiments once

**Files:**
- Create through runners: `results/dev_ablation_preblind/`
- Create through runners: `results/blind_v1_detector_controlled/`
- Create through runners: `results/blind_v1_detector_handheld/`
- Create through runners: `results/blind_v1_pipeline/`

- [ ] **Step 1: Run the complete frozen development-set ablation first**

```powershell
python run_experiments.py --output-dir results/dev_ablation_preblind
```

Expected: one row for every scheduled development image and every ablation config, including C170, C171 and C274. Review only the development results. If code, model or config must change, do not run the blind commands; delete no data, create a new versioned freeze manifest after the approved change, and rerun this development gate.

- [ ] **Step 2: Verify the freeze again immediately before inference**

```powershell
python -c "import json; from pathlib import Path; from src.evaluation.freeze import verify_freeze_manifest; p=Path('benchmark/blind_reference_pack/v1/collection/freeze_manifest.json'); verify_freeze_manifest(json.loads(p.read_text(encoding='utf-8'))); print('freeze verified')"
```

Expected: `freeze verified`.

- [ ] **Step 3: Run detector mAP separately for both domains**

```powershell
python run_detector_eval.py --model runs/detect/cghd_61cls/weights/best.pt --data benchmark/blind_reference_pack/v1/collection/yolo/controlled.yaml --domain controlled --freeze-manifest benchmark/blind_reference_pack/v1/collection/freeze_manifest.json --output-dir results/blind_v1_detector_controlled
python run_detector_eval.py --model runs/detect/cghd_61cls/weights/best.pt --data benchmark/blind_reference_pack/v1/collection/yolo/handheld.yaml --domain handheld --freeze-manifest benchmark/blind_reference_pack/v1/collection/freeze_manifest.json --output-dir results/blind_v1_detector_handheld
```

Expected: both runs write overall Precision/Recall/mAP50/mAP50-95 plus per-class AP, with 30 test images in each domain.

- [ ] **Step 4: Run the end-to-end visual pipeline exactly once**

```powershell
python run_blind_benchmark.py --manifest benchmark/blind_reference_pack/v1/collection/manifest.csv --freeze-manifest benchmark/blind_reference_pack/v1/collection/freeze_manifest.json --output-dir results/blind_v1_pipeline
```

Expected: `per_image.csv` contains 60 rows and 30 unique `paper_id` values; `failures.csv` has only its header; `summary.json` contains domain, difficulty, participant and paired-delta sections; metadata shows `skip_llm=true`.

- [ ] **Step 5: Audit outputs without tuning on blind samples**

Check that controlled and handheld scores are separate, every reported confidence interval uses `paper_id`, and detector/pipeline metadata share the frozen hashes. Produce the six tables listed in the approved design from the generated CSV/JSON. Error examples may be categorized for the paper, but no threshold, correction rule, model weight, GT or post-processing code may be changed in response to these blind outputs.

- [ ] **Step 6: Archive the immutable experiment handoff**

Copy no participant images. Record the four result directories, Git revision, freeze-manifest SHA-256, detector metrics, OCR exact/CER/value-assignment metrics, port PCK/mean error, PC/GA/CNA/edge F1, diagram exact rate and controlled-to-handheld paired deltas in a versioned experiment report under `docs/experiments/`. Commit only the report and non-sensitive result tables after checking that no ignored collection image or annotation is staged.

## Exit criteria

This plan is complete only when:

- all 10 standard drawings are generated from the same semantic source as their answer keys;
- references contain ANSI zigzag resistors, values only, no handwritten designator prompts and no layout defect;
- 30 independent paper originals and 60 paired phone images pass pre-model QC;
- A01 and A02 annotations remain isolated until disagreement generation;
- all 60 adjudicated GT files validate against the publication schema;
- the data, models, Git revision and non-LLM config are frozen before the first blind run.
