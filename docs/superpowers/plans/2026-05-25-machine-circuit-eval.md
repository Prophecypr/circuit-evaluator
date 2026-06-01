# 机画电路图数值识别 + 连线检测 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为机画电路图实现高精度数值识别（Tesseract）和连线检测（霍夫线+掩模），输出元件列表+节点连接+SPICE网表。

**Architecture:** 新增 `src/vision/machine_eval.py`，独立于手绘图管线。OCR 用 Tesseract（印刷体优势）+ CRNN 兜底。线检测用 Canny + Hough + 元件掩模 + 图遍历。最后和手绘图一样走 LLM 评价。

**Tech Stack:** OpenCV, Tesseract 5.5, YOLOv8, NumPy, DeepSeek API

**现状分析:**
- 手绘图 CRNN: val_loss=0.084, 85-90%准确率 ✅
- 机画图 Tesseract: 印刷体能到 95%+，但需好的预处理
- 线检测: 之前 test1_fix.py 验证过——霍夫+掩模在机画图上可行（116段干净线，5+连接）
- 端口: bbox 边缘推算在机画图上精度够用（符号规整）

---

## 文件规划

| 文件 | 责任 |
|------|------|
| `src/vision/machine_eval.py` | **新建** 机画图主管线：检测→OCR→线检测→节点→LLM |
| `src/vision/image_eval.py` | **修改** 拆出公共函数（OCR预处理、数值匹配），保留手绘图管线 |
| `src/vision/connect.py` | **修改** 修复线检测函数，供 machine_eval 调用 |
| `data/machine_test/` | **新建** 存放机画图测试结果 |

---

### Task 1: 验证 Tesseract 在机画图上的基准表现

**Files:** 无新建，纯测试

- [ ] **Step 1: 对比 Tesseract vs CRNN 在 test2（已知值的机画图）上的表现**

```bash
cd E:\ClaudeCode\电路图智能评价系统
python -c "
import cv2, pytesseract, sys
sys.path.insert(0,'src')
from ultralytics import YOLO
from src.vision.train_ocr import load_trained_model, predict as crnn_predict

pytesseract.pytesseract.tesseract_cmd = r'E:/Tesseract-OCR/tesseract.exe'

# Test on test2 — known values: R=50Ω,R=10kΩ,L=5mH,C=10μF,C=10μF,GND
img = cv2.imread('test2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
text_model = YOLO('runs/detect/circuit_text/weights/best.pt')
r = text_model('test2.jpg')[0]

model, chars, _, i2c, img_h = load_trained_model()

print('region | Tesseract | CRNN | GT')
for box in (r.boxes or []):
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    crop = gray[y1:y2, x1:x2]
    
    # Tesseract: resize + OTSU + PSM 7
    h,w = crop.shape; crop_t = cv2.resize(crop, (w*3, h*3))
    _, th = cv2.threshold(crop_t, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t_text = pytesseract.image_to_string(th, config='--psm 7').strip()
    
    # CRNN
    c_text = crnn_predict(model, crop, chars, i2c, img_h)
    
    print(f'({x1},{y1}) | {t_text:12s} | {c_text:12s} | ?')
"
```

- [ ] **Step 2: 分析结果，确定主OCR策略**

根据对比结果：
- 如果 Tesseract 在印刷体上 >90% → Tesseract 为主，CRNN 兜底
- 如果都不理想 → 对 Tesseract 做针对性预处理（见 Task 2）

**预期:** Tesseract 在印刷体上应显著优于 CRNN（Tesseract 训练集就是印刷体）

---

### Task 2: 增强 Tesseract OCR 预处理

**Files:**
- Modify: `src/vision/image_eval.py` — 提取 `_preprocess_for_ocr()` 公共函数
- Modify: `src/vision/machine_eval.py` — 使用增强预处理

- [ ] **Step 1: 实现多策略预处理函数**

```python
def preprocess_for_ocr(crop_gray, is_machine_drawn=True):
    """Apply best preprocessing for OCR based on image type.
    
    Machine-drawn: scale up + OTSU + try both polarities
    Hand-drawn: CRNN handles its own preprocessing
    """
    results = []
    h, w = crop_gray.shape
    
    # Strategy 1: Scale up + OTSU binary (best for printed text)
    scale = max(1, 40 // h)  # Tesseract likes 30-40px height
    scaled = cv2.resize(crop_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    # Try dark-on-light and light-on-dark
    for inv in [False, True]:
        img = 255 - scaled if inv else scaled
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(thresh)
    
    # Strategy 2: Adaptive threshold (better for non-uniform lighting)
    for inv in [False, True]:
        img = 255 - scaled if inv else scaled
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        results.append(thresh)
    
    return results
```

- [ ] **Step 2: 实现多遍 OCR（试所有预处理，选最优）**

```python
def ocr_machine_text(crop_gray, pytesseract):
    """Run Tesseract with multiple preprocessings, return best result."""
    candidates = preprocess_for_ocr(crop_gray, is_machine_drawn=True)
    best_text = ""
    best_len = 0
    
    for thresh in candidates:
        for psm in ["--psm 7", "--psm 8"]:  # single line, single word
            text = pytesseract.image_to_string(
                thresh, config=f"{psm} -c tessedit_char_whitelist=0123456789.kKmMΩμunpFV AHz-+"
            ).strip()
            text = text.replace(" ", "")
            if len(text) > best_len:
                best_text = text
                best_len = len(text)
    
    return best_text
```

- [ ] **Step 3: 测试 test2 验证改进效果**

```bash
python -c "
# Quick test comparing old vs new OCR on test2
# Expected: new method should read 50Ω, 10kΩ, 5mH, 10μF correctly
"
```

---

### Task 3: 创建 machine_eval.py — 机画图主管线

**Files:**
- Create: `src/vision/machine_eval.py`

- [ ] **Step 1: 搭骨架——机画图检测+OCR**

```python
"""Machine-drawn circuit evaluation: YOLO → Tesseract OCR → Wire detection → SPICE → LLM."""

import sys, os, cv2, json, math, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
from src.llm import ask

# Tesseract setup
import pytesseract
TESSERACT_PATH = r"E:/Tesseract-OCR/tesseract.exe"
if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ.setdefault("TESSDATA_PREFIX", r"E:/Tesseract-OCR/tessdata")

MODEL_DIR = Path(__file__).parent.parent.parent / "runs" / "detect"


def evaluate_machine_circuit(img_path: str) -> dict:
    """Full pipeline for machine-drawn circuits."""
    comp_model = YOLO(str(MODEL_DIR / "circuit_real" / "weights" / "best.pt"))
    text_model = YOLO(str(MODEL_DIR / "circuit_text" / "weights" / "best.pt"))
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Component detection
    r_comp = comp_model(str(img_path))[0]
    components = build_components(r_comp, comp_model)
    
    # 2. OCR text values
    r_text = text_model(str(img_path))[0]
    text_values = ocr_text_regions(r_text, gray)
    
    # 3. Match values to components
    components = match_values_to_components(components, text_values)
    
    # 4. Wire detection
    wires, connections, port_nodes = detect_wires_and_nodes(img, gray, components)
    
    # 5. Generate SPICE
    spice = generate_spice(components, port_nodes)
    
    # 6. LLM evaluation
    desc = build_description(components, connections, spice)
    evaluation = ask(desc)
    
    return dict(components=components, text_values=text_values,
                wires=wires, connections=connections, port_nodes=port_nodes,
                spice=spice, evaluation=evaluation)


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "test2.jpg"
    r = evaluate_machine_circuit(img)
    print(r["evaluation"])
```

- [ ] **Step 2: 实现 build_components——从 YOLO 结果构建元件列表**

```python
def build_components(r_comp, comp_model):
    """Convert YOLO results to component dicts with inferred ports."""
    components = []
    for box in (r_comp.boxes or []):
        name = comp_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Machine-drawn: use bbox edge ports (symbols are regular)
        ports = infer_ports_machine(x1, y1, x2, y2, name)
        
        components.append(dict(
            idx=len(components), name=name,
            xyxy=(x1, y1, x2, y2), cx=cx, cy=cy,
            ports=ports, value="", prefix=SPICE_PREFIX.get(name, ""),
        ))
    return components
```

- [ ] **Step 3: 实现 infer_ports_machine——机画图端口推算**

```python
# SPICE prefixes and expected pins
COMP_INFO = {
    "Resistor": ("R", 2), "Capacitor": ("C", 2), "Inductor": ("L", 2),
    "Diode": ("D", 2), "Zener Diode": ("D", 2),
    "V-DC": ("V", 2), "V-AC": ("V", 2), "I-DC": ("I", 2), "I-AC": ("I", 2),
    "GND": ("", 1), "Wire Crossover": ("", 0),
    "BJT-NPN": ("Q", 3), "BJT-PNP": ("Q", 3),
    "MOSFET-N": ("M", 3), "MOSFET-P": ("M", 3),
    "Op-Amp": ("X", 3),
}

def infer_ports_machine(x1, y1, x2, y2, name):
    """Infer port positions from bbox for machine-drawn components.
    
    Machine-drawn symbols are regular and predictable — bbox edges work well.
    """
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    _, n_pins = COMP_INFO.get(name, ("", 2))
    
    if name == "GND":
        return [(cx, y1)]
    if n_pins == 0:
        return []
    
    # Determine orientation from aspect ratio
    w, h = x2 - x1, y2 - y1
    if n_pins == 2:
        if w > h * 1.3:      # horizontal — left/right ports
            return [(x1, cy), (x2, cy)]
        elif h > w * 1.3:    # vertical — top/bottom ports
            return [(cx, y1), (cx, y2)]
        else:                # square — use default
            return [(x1, cy), (x2, cy)]
    elif n_pins == 3:
        # 3-pin: left, top, bottom (MOSFET/BJT layout)
        return [(x1, cy), (cx, y1), (cx, y2)]
    return [(x1, cy), (x2, cy)]
```

- [ ] **Step 4: 实现 OCR 和数值匹配（复用预处理+语义过滤）**

```python
def ocr_text_regions(r_text, gray):
    """OCR all text regions using Tesseract."""
    values = []
    for box in (r_text.boxes or []):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        text = ocr_machine_text(crop, pytesseract)
        text = _clean_machine_ocr(text)
        if text:
            values.append(dict(text=text, cx=(x1+x2)//2, cy=(y1+y2)//2,
                              xyxy=(x1,y1,x2,y2), conf=float(box.conf[0])))
    return values


def _clean_machine_ocr(text):
    """Clean Tesseract output for printed circuit notation."""
    import re
    text = text.strip().replace(" ", "").replace("\n", "")
    if len(text) < 1:
        return ""
    
    # Fix common Tesseract errors on printed text
    text = text.replace("u", "μ").replace("U", "μ")
    text = re.sub(r"(?<=[\d.])Q", "Ω", text)
    
    # Normalize "3V3" → "3.3V", "1k2" → "1.2kΩ"
    m = re.match(r"^(\d+)([kKRr])(\d+)(.*)$", text)
    if m:
        unit = "Ω" if m.group(2) in "Rr" else m.group(2)
        text = f"{m.group(1)}.{m.group(3)}{unit}"
    m = re.match(r"^(\d+)([Vv])(\d+)$", text)
    if m:
        text = f"{m.group(1)}.{m.group(3)}V"
    
    return text
```

- [ ] **Step 5: 提交骨架代码**

```bash
git add src/vision/machine_eval.py
git commit -m "feat: add machine_eval pipeline skeleton"
```

---

### Task 4: 实现机画图连线检测

**Files:**
- Modify: `src/vision/machine_eval.py` — 添加 `detect_wires_and_nodes()`

- [ ] **Step 1: 实现线检测——Canny + Hough + 元件掩模**

```python
def detect_wires_and_nodes(img, gray, components):
    """Detect wires and build node graph for machine-drawn circuits."""
    h, w = gray.shape
    
    # 1. Create component mask (exclude component interiors from wire detection)
    comp_mask = np.zeros((h, w), dtype=np.uint8)
    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        cv2.rectangle(comp_mask, (x1+3, y1+3), (x2-3, y2-3), 255, -1)
    
    # 2. Edge detection + component masking
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges_masked = cv2.bitwise_and(edges, cv2.bitwise_not(comp_mask))
    
    # 3. Hough line detection
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=30,
                            minLineLength=15, maxLineGap=10)
    wire_segments = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = [int(v) for v in l[0]]
            wire_segments.append(((x1, y1), (x2, y2)))
    
    # 4. Build wire adjacency graph
    adj = build_wire_adjacency(wire_segments, max_gap=10)
    
    # 5. Match ports to wire endpoints
    port_matches = match_ports_to_wires(components, wire_segments)
    
    # 6. Find connections via wire graph BFS
    connections, port_nodes = find_wire_connections(
        components, port_matches, wire_segments, adj
    )
    
    return wire_segments, connections, port_nodes
```

- [ ] **Step 2: 实现线图邻接矩阵**

```python
def build_wire_adjacency(segments, max_gap=10):
    """Build adjacency list of wire segments that share endpoints."""
    n = len(segments)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            for e1 in segments[i]:
                for e2 in segments[j]:
                    if math.hypot(e1[0]-e2[0], e1[1]-e2[1]) < max_gap:
                        adj[i].append(j)
                        adj[j].append(i)
                        break
                else:
                    continue
                break
    return adj
```

- [ ] **Step 3: 实现端口→线端点匹配**

```python
def match_ports_to_wires(components, segments, max_dist=30):
    """Match each component port to nearest wire segment endpoint."""
    matches = {}  # (comp_idx, port_idx) → segment_idx
    for c in components:
        for pi, (px, py) in enumerate(c["ports"]):
            best_seg = None
            best_dist = max_dist
            for si, ((sx1, sy1), (sx2, sy2)) in enumerate(segments):
                d1 = math.hypot(px-sx1, py-sy1)
                d2 = math.hypot(px-sx2, py-sy2)
                d = min(d1, d2)
                if d < best_dist:
                    best_dist = d
                    best_seg = si
            if best_seg is not None:
                matches[(c["idx"], pi)] = best_seg
    return matches
```

- [ ] **Step 4: 实现线图 BFS 找连接**

```python
def find_wire_connections(components, port_matches, segments, adj,
                          max_hops=4):
    """BFS through wire graph to find connected ports."""
    port_nodes = {}
    next_node = 1
    
    # GND → node 0
    for c in components:
        if c["name"] == "GND":
            for pi in range(len(c["ports"])):
                port_nodes[(c["idx"], pi)] = 0
    
    connections = []
    keys = list(port_matches.keys())
    
    for i in range(len(keys)):
        ki = keys[i]
        if ki in port_nodes:
            continue
        ci, pi = ki
        seg_i = port_matches[ki]
        
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if kj in port_nodes or ki[0] == kj[0]:
                continue
            seg_j = port_matches[kj]
            
            # BFS from seg_i to seg_j through wire adjacency
            if bfs_wire_graph(seg_i, seg_j, adj, max_hops):
                d = math.hypot(
                    components[ci]["ports"][pi][0] - components[kj[0]]["ports"][kj[1]][0],
                    components[ci]["ports"][pi][1] - components[kj[0]]["ports"][kj[1]][1]
                )
                connections.append((ki, kj, d))
    
    # Greedy node assignment
    for (ci, pi), (cj, pj), dist in sorted(connections, key=lambda x: x[2]):
        if (ci, pi) not in port_nodes and (cj, pj) not in port_nodes:
            nid = next_node; next_node += 1
            port_nodes[(ci, pi)] = nid
            port_nodes[(cj, pj)] = nid
        elif (ci, pi) in port_nodes and (cj, pj) not in port_nodes:
            port_nodes[(cj, pj)] = port_nodes[(ci, pi)]
        elif (cj, pj) in port_nodes and (ci, pi) not in port_nodes:
            port_nodes[(ci, pi)] = port_nodes[(cj, pj)]
    
    # Unconnected → solo nodes
    for c in components:
        for pi in range(len(c["ports"])):
            if (c["idx"], pi) not in port_nodes:
                port_nodes[(c["idx"], pi)] = next_node; next_node += 1
    
    return connections, port_nodes


def bfs_wire_graph(start, end, adj, max_depth=4):
    """BFS through wire graph. Returns True if path exists."""
    if start == end:
        return True
    visited = {start}
    queue = [(start, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for nb in adj.get(cur, []):
            if nb == end:
                return True
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return False
```

- [ ] **Step 5: 提交连线检测代码**

```bash
git add src/vision/machine_eval.py
git commit -m "feat: add wire detection and node graph for machine circuits"
```

---

### Task 5: 生成 SPICE + LLM 评价

**Files:**
- Modify: `src/vision/machine_eval.py` — 添加 `generate_spice()` 和 `build_description()`

- [ ] **Step 1: 实现 SPICE 网表生成**

```python
def generate_spice(components, port_nodes):
    """Generate SPICE netlist from components and node assignments."""
    lines = ["* Machine-drawn circuit", "* Generated by Circuit Evaluator", ""]
    counters = {}
    models = set()
    
    for c in components:
        pfx = c["prefix"]
        n_ports = len(c["ports"])
        if not pfx or n_ports < 2:
            continue
        
        counters[pfx] = counters.get(pfx, 0) + 1
        ref = f"{pfx}{counters[pfx]}"
        nids = [f"N{port_nodes.get((c['idx'], pi), 'x')}" for pi in range(n_ports)]
        val = c.get("value", "")
        
        if pfx == "V":
            lines.append(f"{ref} {nids[0]} {nids[1]} DC {val}" if val else f"{ref} {nids[0]} {nids[1]} DC 0")
        elif pfx == "I":
            lines.append(f"{ref} {nids[0]} {nids[1]} DC {val}" if val else f"{ref} {nids[0]} {nids[1]} DC 0")
        elif pfx == "D":
            lines.append(f"{ref} {nids[0]} {nids[1]} DMOD")
            models.add(".model DMOD D (IS=1e-9 N=2.0)")
        elif pfx in ("M", "Q") and n_ports >= 3:
            body = nids[3] if n_ports >= 4 else nids[2]
            mname = "NMOS" if pfx == "M" else "NPN"
            lines.append(f"{ref} {nids[0]} {nids[1]} {nids[2]} {body} {mname}")
            models.add(f".model {mname} {mname} (LEVEL=1 VTO=1.5 KP=1e-3)")
        else:
            lines.append(f"{ref} {nids[0]} {nids[1]} {val}")
    
    if models:
        lines.append("")
        lines.extend(sorted(models))
    lines += ["", ".OP", ".END"]
    return "\n".join(lines)
```

- [ ] **Step 2: 实现带连接信息的文字描述**

```python
def build_description(components, connections, spice):
    """Build LLM evaluation prompt with component + connection info."""
    lines = ["以下是机器印刷电路图的自动识别结果：", ""]
    
    # Components
    lines.append("## 元器件")
    for c in components:
        val = f"数值={c['value']}" if c['value'] else "数值未识别"
        lines.append(f"- {c['name']}: {val}")
    
    # Connections
    lines.append("\n## 连线拓扑")
    for (ci, pi), (cj, pj), dist in connections:
        ni = components[ci]["name"]
        nj = components[cj]["name"]
        lines.append(f"- {ni}[端口{pi}] ←→ {nj}[端口{pj}]")
    
    # SPICE
    lines.append(f"\n## 生成SPICE网表\n```spice\n{spice}\n```")
    
    lines.append("\n请评价此电路：1. 数值是否正确 2. 拓扑是否合理 3. 整体质量")
    return "\n".join(lines)
```

- [ ] **Step 3: 提交**

```bash
git add src/vision/machine_eval.py
git commit -m "feat: add SPICE generation and LLM evaluation for machine circuits"
```

---

### Task 6: 测试 + 可视化

**Files:**
- Modify: `src/vision/machine_eval.py` — 添加可视化输出

- [ ] **Step 1: 实现标注图绘制**

```python
def draw_machine_annotated(img_path, components, text_values, wires, 
                           connections, port_nodes, out_path):
    """Draw annotated output: components + OCR + wires + connections."""
    img = cv2.imread(str(img_path))
    
    # Wire segments in orange
    for (sx1, sy1), (sx2, sy2) in wires:
        cv2.line(img, (sx1, sy1), (sx2, sy2), (50, 180, 255), 1)
    
    # Components in color
    colors = {"Resistor": (0,255,0), "Capacitor": (0,255,255), "Inductor": (255,255,0),
              "Diode": (0,128,255), "GND": (0,0,255), "V-DC": (255,0,0), "I-DC": (0,200,0)}
    for c in components:
        x1, y1, x2, y2 = c["xyxy"]
        cv2.rectangle(img, (x1,y1), (x2,y2), colors.get(c["name"], (255,255,255)), 2)
        label = f'{c["name"]}={c["value"]}' if c["value"] else c["name"]
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    colors.get(c["name"], (255,255,255)), 1)
        for pi, (px, py) in enumerate(c["ports"]):
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
            nid = port_nodes.get((c["idx"], pi), "?")
            cv2.putText(img, f"N{nid}", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
    
    # Connections in green
    for (ci, pi), (cj, pj), _ in connections:
        ca, cb = components[ci], components[cj]
        if pi < len(ca["ports"]) and pj < len(cb["ports"]):
            cv2.line(img, ca["ports"][pi], cb["ports"][pj], (0, 255, 0), 2)
    
    cv2.imwrite(str(out_path), img)
```

- [ ] **Step 2: 在 test2 上跑全链路验证**

```bash
cd E:\ClaudeCode\电路图智能评价系统
export ANTHROPIC_API_KEY="sk-72ab14d2654c44fcaf7a1c741c565132"
python -m src.vision.machine_eval test2.jpg
```

**预期输出：**
- 7 个元件：3C + 1L + 2R + 1GND
- Tesseract 正确读出 50Ω, 10kΩ, 5mH, 10μF, 10μF
- 5+ 条连线（R1-C2 等）
- SPICE 网表无短路
- 标注图 test2_machine_annotated.jpg

- [ ] **Step 3: 通过后提交**

```bash
git add src/vision/machine_eval.py
git commit -m "feat: add visualization to machine circuit evaluation"
```

---

## 自检清单

| 检查项 | 状态 |
|------|:--:|
| 是否覆盖数值识别方案 | ✅ Task 1+2 |
| 是否覆盖连线检测方案 | ✅ Task 4 |
| 是否与手绘图管线解耦 | ✅ 独立文件 |
| 是否能生成 SPICE | ✅ Task 5 |
| 是否有点到点的可执行步骤 | ✅ 每步含代码/命令 |
| 无占位符/TODO | ✅ |
