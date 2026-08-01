# 投稿级视觉评测基础设施 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立不会漏图、完全跳过 LLM、能分别评测元件、数值、端口、连接和端到端结果的投稿级实验基础设施。

**Architecture:** 保留 `run_experiments.py` 作为现有 50 张开发集的连线消融入口，并修复其图像发现逻辑。新增独立的 `src/evaluation/` 包和 `run_blind_benchmark.py`，使用版本化 JSON GT、IoU 元件匹配、端口语义匹配、配对 Bootstrap 与冻结清单评测新盲测集；检测 mAP 由单独的 Ultralytics test-split 入口计算。

**Tech Stack:** Python 3、pytest、Pydantic 2、NumPy、Ultralytics YOLO、现有 OpenCV/Matplotlib 视觉流水线、CSV/JSON。

---

## File structure

- Modify: `run_experiments.py` — 扩展名无关的开发集排程，缺失或重复文件立即失败。
- Modify: `tests/test_wiring_experiments.py` — 覆盖 `.jpg/.JPG/.jpeg/.png`、缺图、重复图和现有 50 张集成检查。
- Modify: `src/vision/unified_pipeline.py` — 把 `transistor.bjt` 统一为 BJT 家族，固定 B/C/E 端口。
- Modify: `gen_annotation_data.py` — 与主流水线共享同一 BJT 命名和端口顺序。
- Create: `tests/test_component_schema.py` — 防止主流水线和标注生成器再次漂移。
- Create: `src/evaluation/__init__.py` — 公开 GT、指标与冻结接口。
- Create: `src/evaluation/schema.py` — 版本化盲测 GT 与 manifest 校验。
- Create: `src/evaluation/metrics.py` — 元件、OCR、端口、连接、端到端指标。
- Create: `src/evaluation/reporting.py` — 分层汇总、配对差值和 Bootstrap 置信区间。
- Create: `src/evaluation/freeze.py` — 数据、模型、配置和 Git 版本哈希冻结与复核。
- Create: `run_blind_benchmark.py` — 只运行冻结的 `Ours` 非 LLM 配置并写独立结果目录。
- Create: `run_detector_eval.py` — 对 controlled/handheld 两个 YOLO test split 分别计算 mAP。
- Create: `tests/test_blind_schema.py` — GT 和 manifest 契约测试。
- Create: `tests/test_publication_metrics.py` — 指标的可手算单元测试。
- Create: `tests/test_blind_runner.py` — 无模型依赖的盲测入口测试。
- Create: `tests/test_freeze_manifest.py` — 文件篡改检测。

### Task 1: Make benchmark image discovery extension-aware and fail-fast

**Files:**
- Modify: `run_experiments.py:21-25,295-307`
- Modify: `tests/test_wiring_experiments.py`

- [ ] **Step 1: Write failing discovery tests**

Append these tests to `tests/test_wiring_experiments.py`:

```python
def _seed_benchmark_case(root, stem, suffix):
    (root / "result").mkdir(exist_ok=True)
    (root / "detections").mkdir(exist_ok=True)
    (root / "fixed").mkdir(exist_ok=True)
    (root / "result" / f"{stem}_gt.txt").write_text(
        "G1: R1.1, R2.2\n", encoding="utf-8",
    )
    (root / "detections" / f"{stem}.json").write_text(
        '{"components": []}', encoding="utf-8",
    )
    (root / f"{stem}{suffix}").write_bytes(b"image")


def test_get_image_list_accepts_supported_extensions(tmp_path):
    for stem, suffix in [
        ("a", ".jpg"), ("b", ".JPG"), ("c", ".jpeg"), ("d", ".png"),
    ]:
        _seed_benchmark_case(tmp_path, stem, suffix)

    rows = run_experiments.get_image_list(tmp_path)

    assert [row[0] for row in rows] == ["a", "b", "c", "d"]
    assert [Path(row[1]).suffix for row in rows] == [".jpg", ".JPG", ".jpeg", ".png"]


def test_get_image_list_rejects_missing_image(tmp_path):
    _seed_benchmark_case(tmp_path, "missing", ".jpg")
    (tmp_path / "missing.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="missing.*image"):
        run_experiments.get_image_list(tmp_path)


def test_get_image_list_rejects_duplicate_stem(tmp_path):
    _seed_benchmark_case(tmp_path, "duplicate", ".jpg")
    (tmp_path / "duplicate.jpeg").write_bytes(b"second image")

    with pytest.raises(RuntimeError, match="duplicate.*multiple images"):
        run_experiments.get_image_list(tmp_path)


def test_repository_benchmark_schedules_every_gt_file():
    rows = run_experiments.get_image_list(Path("benchmark"))
    gt_stems = {
        path.stem.removesuffix("_gt")
        for path in Path("benchmark/result").glob("*_gt.txt")
    }

    assert {row[0] for row in rows} == gt_stems
    assert {"C170_D1_P1", "C171_D1_P1", "C274_D1_P1"} <= gt_stems
```

Add `from pathlib import Path` to the test imports.

- [ ] **Step 2: Run the discovery tests and verify the old code fails**

Run:

```powershell
pytest tests/test_wiring_experiments.py -k "image_list or repository_benchmark" -v
```

Expected: `.jpeg/.JPG/.png` cases are omitted and the missing/duplicate cases do not raise the required errors.

- [ ] **Step 3: Implement strict image discovery**

Add near the benchmark path constants in `run_experiments.py`:

```python
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
```

Replace `get_image_list()` with:

```python
def get_image_list(benchmark_dir=BENCHMARK):
    """Return every GT-backed image, refusing silent omissions or duplicate stems."""
    benchmark_dir = Path(benchmark_dir)
    result_dir = benchmark_dir / "result"
    detections_dir = benchmark_dir / "detections"
    fixed_dir = benchmark_dir / "fixed"
    images = []

    for gt_file in sorted(result_dir.glob("*_gt.txt")):
        stem = gt_file.stem.removesuffix("_gt")
        candidates = sorted(
            path for path in benchmark_dir.iterdir()
            if path.is_file()
            and path.stem == stem
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidates:
            raise FileNotFoundError(f"{stem}: missing benchmark image")
        if len(candidates) > 1:
            joined = ", ".join(path.name for path in candidates)
            raise RuntimeError(f"{stem}: multiple images share the GT stem: {joined}")

        fixed_path = fixed_dir / f"{stem}.json"
        detected_path = detections_dir / f"{stem}.json"
        if fixed_path.exists():
            det_path = fixed_path
        elif detected_path.exists():
            det_path = detected_path
        else:
            raise FileNotFoundError(f"{stem}: missing detection JSON")

        images.append((stem, str(candidates[0]), str(gt_file), str(det_path)))
    return images
```

- [ ] **Step 4: Run the complete wiring experiment unit tests**

Run:

```powershell
pytest tests/test_wiring_experiments.py -v
```

Expected: all tests pass and the repository integration test schedules C170, C171 and C274.

- [ ] **Step 5: Commit the discovery fix**

```powershell
git add run_experiments.py tests/test_wiring_experiments.py
git commit -m "fix: schedule every benchmark image extension"
```

### Task 2: Normalize BJT and polar port contracts

**Files:**
- Modify: `src/vision/unified_pipeline.py:53-61,109-121,139-160,195-216,263-277,291-315,2577-2587`
- Modify: `gen_annotation_data.py:20-120`
- Modify: `benchmark/annotation_tool.html:202-228`
- Create: `tests/test_component_schema.py`

- [ ] **Step 1: Write a failing cross-file schema test**

Create `tests/test_component_schema.py`:

```python
import ast
from pathlib import Path

from src.vision import unified_pipeline as pipeline


def _assignment(path, name):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_bjt_family_has_one_name_and_one_port_order():
    annotation_names = _assignment("gen_annotation_data.py", "CGH_NAME_MAP")
    annotation_positions = _assignment("gen_annotation_data.py", "PORT_POSITIONS")
    annotation_labels = _assignment("gen_annotation_data.py", "PORT_LABELS")

    assert pipeline.CGH_NAME_MAP["transistor.bjt"] == "BJT"
    assert pipeline.CGH_TO_PORT_KEY["transistor.bjt"] == "BJT"
    assert pipeline.PORT_LABELS["BJT"] == ["B", "C", "E"]
    assert pipeline.PORT_POSITIONS["BJT"] == [
        (0.0, 0.5), (0.7, 0.0), (0.7, 1.0),
    ]
    assert annotation_names["transistor.bjt"] == "BJT"
    assert annotation_labels["BJT"] == ["B", "C", "E"]
    assert annotation_positions["BJT"] == [
        (0.0, 0.5), (0.7, 0.0), (0.7, 1.0),
    ]
    assert annotation_labels["Zener-Diode"] == ["A", "K"]
    annotation_html = Path("benchmark/annotation_tool.html").read_text(encoding="utf-8")
    assert "'BJT':['B','C','E']" in annotation_html
    assert "'Zener-Diode':['A','K']" in annotation_html
```

- [ ] **Step 2: Run the test and verify the pipeline naming mismatch**

Run:

```powershell
pytest tests/test_component_schema.py -v
```

Expected: FAIL because the main pipeline maps `transistor.bjt` to `BJT-PNP` and the annotation generator uses a different B/E/C ordering.

- [ ] **Step 3: Apply the single BJT family contract**

In `src/vision/unified_pipeline.py`, make these exact replacements:

```python
# In CGH_NAME_MAP and CGH_TO_PORT_KEY
"transistor.bjt": "BJT",

# In PORT_LABELS
"BJT": ["B", "C", "E"],

# In PORT_POSITIONS
"BJT": [(0.0, 0.5), (0.7, 0.0), (0.7, 1.0)],

# In ANTI_PATTERNS
"BJT": [r"[ΩFVHzμAH]$"],

# In DESIG and NM_CH
"BJT": "Q",
"BJT": "BJT三极管",

# In _draw_result COLORS
"BJT": (150, 150, 0),
```

Remove the now-unreachable `BJT-NPN` and `BJT-PNP` entries from those same dictionaries. In `gen_annotation_data.py`, keep the existing `"transistor.bjt": "BJT"` mapping but change both the comment and labels to:

```python
"BJT": [(0.0, 0.5), (0.7, 0.0), (0.7, 1.0)],  # B, C, E
"BJT": ["B", "C", "E"],
"Zener-Diode": ["A", "K"],
```

In `benchmark/annotation_tool.html`, change `defLabels` to use `'Zener-Diode':['A','K']` and `'BJT':['B','C','E']`. Keep LED as `['+','-']`.

- [ ] **Step 4: Run schema and visual tests**

Run:

```powershell
pytest tests/test_component_schema.py tests/test_wire_mask.py tests/test_wiring_experiments.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the BJT contract**

```powershell
git add src/vision/unified_pipeline.py gen_annotation_data.py benchmark/annotation_tool.html tests/test_component_schema.py
git commit -m "fix: unify BJT family port semantics"
```

### Task 3: Add the versioned blind-GT and manifest schema

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/schema.py`
- Create: `tests/test_blind_schema.py`

- [ ] **Step 1: Write failing GT and manifest contract tests**

Create `tests/test_blind_schema.py` with:

```python
import json

import pytest

from src.evaluation.schema import BlindGroundTruth, load_manifest


def valid_gt():
    return {
        "schema_version": "1.0",
        "image": {
            "paper_id": "C01_P01", "circuit_id": "C01",
            "participant_id": "P01", "domain": "controlled",
            "width": 1000, "height": 700,
        },
        "components": [
            {
                "id": "R1", "class_name": "resistor", "xyxy": [100, 100, 300, 180],
                "value": {"raw": "1kΩ", "normalized": "1kΩ", "xyxy": [130, 60, 220, 95]},
                "ports": [
                    {"label": "1", "xy": [100, 140], "net_id": "N1"},
                    {"label": "2", "xy": [300, 140], "net_id": "N0"},
                ],
            },
            {
                "id": "V1", "class_name": "voltage.dc", "xyxy": [400, 200, 500, 300],
                "value": {"raw": "5V", "normalized": "5V", "xyxy": [505, 220, 560, 250]},
                "ports": [
                    {"label": "+", "xy": [450, 200], "net_id": "N1"},
                    {"label": "-", "xy": [450, 300], "net_id": "N0"},
                ],
            },
        ],
        "nets": {"N1": ["R1.1", "V1.+"], "N0": ["R1.2", "V1.-"]},
    }


def test_ground_truth_accepts_complete_port_partition():
    gt = BlindGroundTruth.model_validate(valid_gt())
    assert gt.image.paper_id == "C01_P01"
    assert gt.nets["N1"] == ["R1.1", "V1.+"]


def test_ground_truth_rejects_port_missing_from_net():
    data = valid_gt()
    data["nets"]["N1"].remove("R1.1")
    with pytest.raises(ValueError, match="net membership"):
        BlindGroundTruth.model_validate(data)


def test_manifest_requires_paired_domains(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(valid_gt()), encoding="utf-8")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "paper_id,circuit_id,participant_id,difficulty,domain,image_path,gt_path\n"
        f"C01_P01,C01,P01,basic,controlled,{image_path.name},{gt_path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controlled.*handheld"):
        load_manifest(manifest, require_pairs=True)
```

- [ ] **Step 2: Run tests and confirm the evaluation package is absent**

Run:

```powershell
pytest tests/test_blind_schema.py -v
```

Expected: collection fails with `ModuleNotFoundError: src.evaluation`.

- [ ] **Step 3: Implement the complete Pydantic schema**

Create `src/evaluation/schema.py`:

```python
import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


Domain = Literal["controlled", "handheld"]
Difficulty = Literal["basic", "medium", "challenge"]


class ValueGT(BaseModel):
    raw: str
    normalized: str
    xyxy: tuple[float, float, float, float]


class PortGT(BaseModel):
    label: str
    xy: tuple[float, float]
    net_id: str


class ComponentGT(BaseModel):
    id: str
    class_name: str
    xyxy: tuple[float, float, float, float]
    value: ValueGT | None = None
    ports: list[PortGT] = Field(min_length=1)


class ImageGT(BaseModel):
    paper_id: str
    circuit_id: str
    participant_id: str
    domain: Domain
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class BlindGroundTruth(BaseModel):
    schema_version: Literal["1.0"]
    image: ImageGT
    components: list[ComponentGT] = Field(min_length=1)
    nets: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_net_membership(self):
        component_ids = [component.id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be unique")

        declared = {}
        for component in self.components:
            labels = [port.label for port in component.ports]
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicate port label on {component.id}")
            for port in component.ports:
                declared[f"{component.id}.{port.label}"] = port.net_id

        listed = {}
        for net_id, refs in self.nets.items():
            if len(refs) < 2:
                raise ValueError(f"{net_id} must contain at least two ports")
            for ref in refs:
                if ref in listed:
                    raise ValueError(f"{ref} appears in multiple nets")
                listed[ref] = net_id

        if declared != listed:
            raise ValueError("net membership does not match component ports")
        return self


class ManifestRow(BaseModel):
    paper_id: str
    circuit_id: str
    participant_id: str
    difficulty: Difficulty
    domain: Domain
    image_path: Path
    gt_path: Path


def load_ground_truth(path):
    return BlindGroundTruth.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_manifest(path, require_pairs=True):
    path = Path(path).resolve()
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            raw["image_path"] = (path.parent / raw["image_path"]).resolve()
            raw["gt_path"] = (path.parent / raw["gt_path"]).resolve()
            row = ManifestRow.model_validate(raw)
            if not row.image_path.is_file() or not row.gt_path.is_file():
                raise FileNotFoundError(f"missing image or GT for {row.paper_id}/{row.domain}")
            rows.append(row)

    keys = [(row.paper_id, row.domain) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate paper_id/domain row in manifest")
    if require_pairs:
        by_paper = {}
        for row in rows:
            by_paper.setdefault(row.paper_id, set()).add(row.domain)
        for paper_id, domains in by_paper.items():
            if domains != {"controlled", "handheld"}:
                raise ValueError(f"{paper_id} requires controlled and handheld domains")
    return rows
```

Create `src/evaluation/__init__.py`:

```python
from .schema import BlindGroundTruth, load_ground_truth, load_manifest

__all__ = ["BlindGroundTruth", "load_ground_truth", "load_manifest"]
```

- [ ] **Step 4: Run the schema tests**

Run:

```powershell
pytest tests/test_blind_schema.py -v
```

Expected: all three tests pass.

- [ ] **Step 5: Commit the schema**

```powershell
git add src/evaluation tests/test_blind_schema.py
git commit -m "feat: define versioned blind benchmark schema"
```

### Task 4: Implement publication metrics with hand-checkable tests

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_publication_metrics.py`

- [ ] **Step 1: Write failing normalization, CER, port and graph tests**

Create `tests/test_publication_metrics.py`:

```python
from src.evaluation.metrics import (
    component_neighbor_accuracy, edge_counts, levenshtein, normalize_value,
    port_distance_counts,
)


def test_value_normalization_is_fixed_and_unit_safe():
    assert normalize_value(" 2.2 KΩ ") == "2.2kΩ"
    assert normalize_value("10μF") == "10uF"
    assert normalize_value("10µF") == "10uF"
    assert normalize_value("1MΩ") == "1MΩ"


def test_levenshtein_has_known_distance():
    assert levenshtein("10uF", "100uF") == 1
    assert levenshtein("330Ω", "330Ω") == 0


def test_port_pck_counts_are_based_on_image_diagonal():
    counts = port_distance_counts([0.0, 10.0, 30.0], image_diagonal=1000.0)
    assert counts["port_total"] == 3
    assert counts["port_distance_matched"] == 3
    assert counts["pck_001_correct"] == 2
    assert counts["pck_002_correct"] == 2
    assert counts["mean_normalized_port_error"] == pytest.approx(40 / 3000)


def test_edge_counts_are_exact():
    gt = {("R1.1", "R2.1"), ("R2.2", "V1.+")}
    pred = {("R1.1", "R2.1"), ("R1.2", "V1.-")}
    assert edge_counts(gt, pred) == {"edge_tp": 1, "edge_fp": 1, "edge_fn": 1}


def test_component_neighbor_accuracy_requires_exact_neighbor_sets():
    gt = [{"R1.1", "R2.1", "C1.1"}, {"R1.2", "V1.+"}]
    pred = [{"R1.1", "R2.1"}, {"R1.2", "V1.+"}]
    assert component_neighbor_accuracy(gt, pred) == 0.25
```

- [ ] **Step 2: Run tests and verify the metrics module is absent**

Run:

```powershell
pytest tests/test_publication_metrics.py -v
```

Expected: collection fails because `src.evaluation.metrics` does not exist.

- [ ] **Step 3: Implement deterministic primitive metrics**

Create `src/evaluation/metrics.py` with these foundations:

```python
import math
import re
from itertools import combinations

from src.vision.unified_pipeline import PORT_LABELS


RAW_CLASS = {
    "Resistor": "resistor",
    "Capacitor": "capacitor.unpolarized",
    "Polarized-Capacitor": "capacitor.polarized",
    "Inductor": "inductor",
    "Diode": "diode",
    "LED": "diode.light_emitting",
    "Zener Diode": "diode.zener",
    "V-DC": "voltage.dc",
    "GND": "gnd",
    "BJT": "transistor.bjt",
}


def normalize_value(value):
    value = re.sub(r"\s+", "", value or "")
    value = value.replace("μ", "u").replace("µ", "u")
    value = re.sub(r"(?<=\d)K(?=Ω|$)", "k", value)
    return value


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def box_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_components(predictions, ground_truth, iou_threshold=0.5):
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        pred_class = prediction.get("raw_name") or RAW_CLASS.get(prediction.get("name"))
        for gt_index, target in enumerate(ground_truth):
            if pred_class == target.class_name:
                overlap = box_iou(prediction["xyxy"], target.xyxy)
                if overlap >= iou_threshold:
                    candidates.append((overlap, pred_index, gt_index))
    matches = {}
    used_gt = set()
    for _, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index not in matches and gt_index not in used_gt:
            matches[pred_index] = gt_index
            used_gt.add(gt_index)
    return matches


def group_to_edges(groups):
    return {
        tuple(sorted(pair))
        for group in groups
        for pair in combinations(sorted(set(group)), 2)
    }


def edge_counts(gt_edges, pred_edges):
    return {
        "edge_tp": len(gt_edges & pred_edges),
        "edge_fp": len(pred_edges - gt_edges),
        "edge_fn": len(gt_edges - pred_edges),
    }


def component_neighbor_accuracy(gt_groups, pred_groups):
    def neighbors(groups):
        result = {}
        for group in groups:
            component_ids = {ref.rsplit(".", 1)[0] for ref in group}
            for component_id in component_ids:
                result.setdefault(component_id, set()).update(component_ids - {component_id})
        return result

    expected = neighbors(gt_groups)
    actual = neighbors(pred_groups)
    component_ids = set(expected) | set(actual)
    return safe_ratio(
        sum(expected.get(item, set()) == actual.get(item, set()) for item in component_ids),
        len(component_ids),
    )


def port_distance_counts(distances, image_diagonal, total=None):
    total = len(distances) if total is None else total
    return {
        "port_total": total,
        "port_distance_matched": len(distances),
        "pck_001_correct": sum(distance <= 0.01 * image_diagonal for distance in distances),
        "pck_002_correct": sum(distance <= 0.02 * image_diagonal for distance in distances),
        "mean_normalized_port_error": safe_ratio(sum(distances), len(distances) * image_diagonal),
    }


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def prf(tp, fp, fn):
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1
```

- [ ] **Step 4: Add one exact end-to-end evaluator test**

Append this self-contained test to `tests/test_publication_metrics.py`:

```python
from src.evaluation.metrics import evaluate_blind_result
from src.evaluation.schema import BlindGroundTruth


def test_perfect_pipeline_result_scores_one_everywhere():
    gt = BlindGroundTruth.model_validate({
        "schema_version": "1.0",
        "image": {"paper_id": "C01_P01", "circuit_id": "C01", "participant_id": "P01",
                  "domain": "controlled", "width": 1000, "height": 700},
        "components": [
            {"id": "R1", "class_name": "resistor", "xyxy": [100, 100, 300, 180],
             "value": {"raw": "1kΩ", "normalized": "1kΩ", "xyxy": [130, 60, 220, 95]},
             "ports": [{"label": "1", "xy": [100, 140], "net_id": "N1"},
                       {"label": "2", "xy": [300, 140], "net_id": "N0"}]},
            {"id": "V1", "class_name": "voltage.dc", "xyxy": [400, 200, 500, 300],
             "value": {"raw": "5V", "normalized": "5V", "xyxy": [505, 220, 560, 250]},
             "ports": [{"label": "+", "xy": [450, 200], "net_id": "N1"},
                       {"label": "-", "xy": [450, 300], "net_id": "N0"}]},
        ],
        "nets": {"N1": ["R1.1", "V1.+"], "N0": ["R1.2", "V1.-"]},
    })
    result = {
        "components": [
            {"raw_name": "resistor", "name": "Resistor", "xyxy": [100, 100, 300, 180],
             "value": "1kΩ", "ports": [[100, 140], [300, 140]]},
            {"raw_name": "voltage.dc", "name": "V-DC", "xyxy": [400, 200, 500, 300],
             "value": "5V", "ports": [[450, 200], [450, 300]]},
        ],
        "text_values": [
            {"text": "1kΩ", "xyxy": [130, 60, 220, 95]},
            {"text": "5V", "xyxy": [505, 220, 560, 250]},
        ],
        "raw_groups": [{(0, 0), (1, 0)}, {(0, 1), (1, 1)}],
    }

    metrics = evaluate_blind_result(result, gt)

    assert metrics["component_f1"] == 1.0
    assert metrics["ocr_exact_accuracy"] == 1.0
    assert metrics["value_exact_accuracy"] == 1.0
    assert metrics["value_assignment_accuracy"] == 1.0
    assert metrics["cer"] == 0.0
    assert metrics["pck_001"] == 1.0
    assert metrics["port_label_accuracy"] == 1.0
    assert metrics["edge_f1"] == 1.0
    assert metrics["port_correct_rate"] == 1.0
    assert metrics["comp_neighbor_accuracy"] == 1.0
    assert metrics["diagram_exact"] == 1
```

- [ ] **Step 5: Implement `evaluate_blind_result`**

Add this function below the primitives in `src/evaluation/metrics.py`:

```python
def evaluate_blind_result(result, gt):
    predictions = result["components"]
    matches = match_components(predictions, gt.components)
    component_tp = len(matches)
    component_fp = len(predictions) - component_tp
    component_fn = len(gt.components) - component_tp

    value_targets = [
        (component.id, component.value)
        for component in gt.components
        if component.value is not None
    ]
    value_total = len(value_targets)
    value_exact = 0

    text_predictions = result.get("text_values", [])
    text_candidates = []
    for prediction_index, prediction in enumerate(text_predictions):
        for target_index, (_, target) in enumerate(value_targets):
            overlap = box_iou(prediction.get("xyxy", (0, 0, 0, 0)), target.xyxy)
            if overlap >= 0.30:
                text_candidates.append((overlap, prediction_index, target_index))
    text_matches = {}
    used_text_targets = set()
    for _, prediction_index, target_index in sorted(text_candidates, reverse=True):
        if prediction_index not in text_matches and target_index not in used_text_targets:
            text_matches[prediction_index] = target_index
            used_text_targets.add(target_index)

    ocr_characters = sum(len(normalize_value(target.normalized)) for _, target in value_targets)
    ocr_edits = ocr_characters
    ocr_exact = 0
    for prediction_index, target_index in text_matches.items():
        expected = normalize_value(value_targets[target_index][1].normalized)
        actual = normalize_value(text_predictions[prediction_index].get("text", ""))
        ocr_exact += actual == expected
        ocr_edits += levenshtein(actual, expected) - len(expected)
    port_distances = []
    port_label_correct = 0
    port_label_total = sum(len(component.ports) for component in gt.components)
    pred_port_refs = {}

    for pred_index, gt_index in matches.items():
        prediction = predictions[pred_index]
        target = gt.components[gt_index]
        if target.value is not None:
            expected = normalize_value(target.value.normalized)
            actual = normalize_value(prediction.get("value", ""))
            value_exact += actual == expected
            value_edits += levenshtein(actual, expected) - len(expected)

        labels = PORT_LABELS.get(prediction.get("name"), [])
        target_by_label = {port.label: port for port in target.ports}
        for port_index, point in enumerate(prediction.get("ports", [])):
            label = labels[port_index] if port_index < len(labels) else str(port_index)
            target_port = target_by_label.get(label)
            if target_port is None:
                continue
            port_label_correct += 1
            port_distances.append(math.hypot(
                point[0] - target_port.xy[0], point[1] - target_port.xy[1],
            ))
            pred_port_refs[(pred_index, port_index)] = f"{target.id}.{target_port.label}"

    predicted_groups = []
    for raw_group in result.get("raw_groups", []):
        refs = {
            pred_port_refs[(component_index, port_index)]
            for component_index, port_index in raw_group
            if (component_index, port_index) in pred_port_refs
        }
        if len(refs) >= 2:
            predicted_groups.append(refs)

    gt_groups = [set(refs) for refs in gt.nets.values()]
    gt_edges = group_to_edges(gt_groups)
    pred_edges = group_to_edges(predicted_groups)
    counts = edge_counts(gt_edges, pred_edges)
    component_precision, component_recall, component_f1 = prf(
        component_tp, component_fp, component_fn,
    )
    edge_precision, edge_recall, edge_f1 = prf(
        counts["edge_tp"], counts["edge_fp"], counts["edge_fn"],
    )
    diagonal = math.hypot(gt.image.width, gt.image.height)
    port_counts = port_distance_counts(port_distances, diagonal, total=port_label_total)
    group_correct = sum(group in predicted_groups for group in gt_groups)
    groups_exact = set(map(frozenset, gt_groups)) == set(map(frozenset, predicted_groups))

    metrics = {
        "component_tp": component_tp,
        "component_fp": component_fp,
        "component_fn": component_fn,
        "component_precision": component_precision,
        "component_recall": component_recall,
        "component_f1": component_f1,
        "ocr_total": value_total,
        "ocr_exact": ocr_exact,
        "ocr_exact_accuracy": safe_ratio(ocr_exact, value_total),
        "ocr_edits": ocr_edits,
        "ocr_characters": ocr_characters,
        "value_total": value_total,
        "value_exact": value_exact,
        "value_exact_accuracy": safe_ratio(value_exact, value_total),
        "value_assignment_accuracy": safe_ratio(value_exact, value_total),
        "cer": safe_ratio(ocr_edits, ocr_characters),
        "port_label_total": port_label_total,
        "port_label_correct": port_label_correct,
        "port_label_accuracy": safe_ratio(port_label_correct, port_label_total),
        **port_counts,
        **counts,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "port_correct_rate": edge_recall,
        "comp_neighbor_accuracy": component_neighbor_accuracy(gt_groups, predicted_groups),
        "group_total": len(gt_groups),
        "group_correct": group_correct,
        "group_accuracy": safe_ratio(group_correct, len(gt_groups)),
    }
    metrics["pck_001"] = safe_ratio(port_counts["pck_001_correct"], port_counts["port_total"])
    metrics["pck_002"] = safe_ratio(port_counts["pck_002_correct"], port_counts["port_total"])
    metrics["diagram_exact"] = int(
        component_fp == 0
        and component_fn == 0
        and value_exact == value_total
        and port_label_correct == port_label_total
        and groups_exact
    )
    return metrics
```

- [ ] **Step 6: Run metrics tests**

Run:

```powershell
pytest tests/test_publication_metrics.py tests/test_blind_schema.py -v
```

Expected: all tests pass, including the perfect-prediction test.

- [ ] **Step 7: Commit the metrics**

```powershell
git add src/evaluation/metrics.py tests/test_publication_metrics.py
git commit -m "feat: add publication recognition metrics"
```

### Task 5: Add stratified summaries and paired-domain confidence intervals

**Files:**
- Create: `src/evaluation/reporting.py`
- Modify: `tests/test_publication_metrics.py`

- [ ] **Step 1: Write a deterministic Bootstrap test**

Append:

```python
from src.evaluation.reporting import bootstrap_mean_ci, paired_domain_delta


def test_bootstrap_ci_is_deterministic_and_ordered():
    first = bootstrap_mean_ci([0.2, 0.4, 0.8], seed=7, n_resamples=1000)
    second = bootstrap_mean_ci([0.2, 0.4, 0.8], seed=7, n_resamples=1000)
    assert first == second
    assert first[0] <= first[1] <= first[2]


def test_paired_delta_uses_paper_id_not_sixty_independent_images():
    rows = [
        {"paper_id": "A", "domain": "controlled", "edge_f1": 0.9},
        {"paper_id": "A", "domain": "handheld", "edge_f1": 0.7},
        {"paper_id": "B", "domain": "controlled", "edge_f1": 0.8},
        {"paper_id": "B", "domain": "handheld", "edge_f1": 0.5},
    ]
    result = paired_domain_delta(rows, "edge_f1", seed=3, n_resamples=1000)
    assert result["n_pairs"] == 2
    assert result["mean_delta"] == pytest.approx(-0.25)
```

- [ ] **Step 2: Verify the reporting module is absent**

Run:

```powershell
pytest tests/test_publication_metrics.py -k "bootstrap or paired" -v
```

Expected: import failure for `src.evaluation.reporting`.

- [ ] **Step 3: Implement reporting helpers**

Create `src/evaluation/reporting.py`:

```python
from collections import defaultdict

import numpy as np


REPORT_METRICS = (
    "component_f1", "ocr_exact_accuracy", "cer", "value_exact_accuracy",
    "value_assignment_accuracy", "pck_001", "pck_002",
    "mean_normalized_port_error", "port_label_accuracy", "edge_precision",
    "edge_recall", "edge_f1", "port_correct_rate", "comp_neighbor_accuracy",
    "group_accuracy", "diagram_exact",
)


def bootstrap_mean_ci(values, seed=20260801, n_resamples=10000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None, None
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(n_resamples, values.size), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(values.mean()), float(np.quantile(means, 0.975))


def summarize(rows, group_fields=("domain",)):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        summary = dict(zip(group_fields, key))
        summary["n_images"] = len(group)
        summary["n_papers"] = len({row["paper_id"] for row in group})
        for metric in REPORT_METRICS:
            low, mean, high = bootstrap_mean_ci([row[metric] for row in group])
            summary[metric] = mean
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        output.append(summary)
    return output


def paired_domain_delta(rows, metric, seed=20260801, n_resamples=10000):
    by_paper = defaultdict(dict)
    for row in rows:
        by_paper[row["paper_id"]][row["domain"]] = row[metric]
    deltas = [
        domains["handheld"] - domains["controlled"]
        for domains in by_paper.values()
        if set(domains) == {"controlled", "handheld"}
    ]
    low, mean, high = bootstrap_mean_ci(deltas, seed=seed, n_resamples=n_resamples)
    return {"metric": metric, "n_pairs": len(deltas), "ci_low": low, "mean_delta": mean, "ci_high": high}
```

- [ ] **Step 4: Run reporting tests**

Run:

```powershell
pytest tests/test_publication_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit reporting**

```powershell
git add src/evaluation/reporting.py tests/test_publication_metrics.py
git commit -m "feat: add paired benchmark reporting"
```

### Task 6: Add the non-LLM blind benchmark runner

**Files:**
- Create: `run_blind_benchmark.py`
- Create: `tests/test_blind_runner.py`

- [ ] **Step 1: Write a failing runner test with a fake pipeline**

Create `tests/test_blind_runner.py`:

```python
import json

import run_blind_benchmark


def test_runner_separates_domains_and_forces_llm_off(tmp_path, monkeypatch):
    rows = [
        type("Row", (), {
            "paper_id": "C01_P01", "circuit_id": "C01", "participant_id": "P01",
            "difficulty": "basic", "domain": domain,
            "image_path": tmp_path / f"{domain}.jpg", "gt_path": tmp_path / f"{domain}.json",
        })()
        for domain in ("controlled", "handheld")
    ]
    for row in rows:
        row.image_path.write_bytes(b"image")
        row.gt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_blind_benchmark, "load_manifest", lambda _path, require_pairs: rows)
    monkeypatch.setattr(run_blind_benchmark, "load_ground_truth", lambda _path: object())

    def fake_process(_path, config):
        assert config["skip_llm"] is True
        assert config["save_artifacts"] is False
        return {"components": [], "raw_groups": []}

    monkeypatch.setattr(run_blind_benchmark, "process_image", fake_process)
    monkeypatch.setattr(
        run_blind_benchmark, "evaluate_blind_result",
        lambda _result, _gt: {
            "component_f1": 1.0, "ocr_exact_accuracy": 1.0, "cer": 0.0,
            "value_exact_accuracy": 1.0, "value_assignment_accuracy": 1.0,
            "pck_001": 1.0, "pck_002": 1.0, "mean_normalized_port_error": 0.0,
            "port_label_accuracy": 1.0,
            "edge_precision": 1.0, "edge_recall": 1.0, "edge_f1": 1.0,
            "port_correct_rate": 1.0, "comp_neighbor_accuracy": 1.0,
            "group_accuracy": 1.0, "diagram_exact": 1,
        },
    )

    output = tmp_path / "run"
    run_blind_benchmark.run("manifest.csv", output)

    per_image = (output / "per_image.csv").read_text(encoding="utf-8")
    assert "controlled" in per_image and "handheld" in per_image
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired_deltas"][0]["n_pairs"] == 1
```

- [ ] **Step 2: Run the runner test and verify the module is absent**

Run:

```powershell
pytest tests/test_blind_runner.py -v
```

Expected: collection fails because `run_blind_benchmark.py` does not exist.

- [ ] **Step 3: Implement the runner with an isolated output directory**

Create `run_blind_benchmark.py`. Use the existing `resolve_output_dir()` and `build_ablation_configs()["Ours"]`, then write these four files: `per_image.csv`, `summary.json`, `failures.csv`, and `run_metadata.json`. Start with these imports and constants:

```python
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from run_experiments import build_ablation_configs, get_git_revision, resolve_output_dir
from src.evaluation.metrics import evaluate_blind_result
from src.evaluation.reporting import REPORT_METRICS, paired_domain_delta, summarize
from src.evaluation.schema import load_ground_truth, load_manifest
from src.vision.unified_pipeline import process_image


DETECTOR_WEIGHTS = Path("runs/detect/cghd_61cls/weights/best.pt")
OCR_WEIGHTS = Path("runs/ocr_crnn_machine/best.pt")
FAILURE_FIELDS = ("paper_id", "domain", "error_type", "message")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

Use this core loop:

```python
def run(manifest_path, output_dir):
    output_dir = resolve_output_dir(output_dir)
    manifest_rows = load_manifest(manifest_path, require_pairs=True)
    config = dict(build_ablation_configs()["Ours"])
    if config["skip_llm"] is not True or config["save_artifacts"] is not False:
        raise RuntimeError("blind benchmark must be LLM-free and artifact-isolated")

    rows = []
    failures = []
    for item in manifest_rows:
        try:
            gt = load_ground_truth(item.gt_path)
            result = process_image(str(item.image_path), config=dict(config))
            if result is None:
                raise RuntimeError("process_image returned None")
            metrics = evaluate_blind_result(result, gt)
            rows.append({
                "paper_id": item.paper_id,
                "circuit_id": item.circuit_id,
                "participant_id": item.participant_id,
                "difficulty": item.difficulty,
                "domain": item.domain,
                **metrics,
            })
        except Exception as error:
            failures.append({
                "paper_id": item.paper_id,
                "domain": item.domain,
                "error_type": type(error).__name__,
                "message": str(error),
            })

    write_csv(output_dir / "failures.csv", failures, FAILURE_FIELDS)
    if failures:
        raise RuntimeError(f"blind benchmark failed for {len(failures)} images")

    write_csv(output_dir / "per_image.csv", rows, tuple(rows[0]))
    summary = {
        "by_domain": summarize(rows, ("domain",)),
        "by_domain_difficulty": summarize(rows, ("domain", "difficulty")),
        "by_domain_participant": summarize(rows, ("domain", "participant_id")),
        "paired_deltas": [paired_domain_delta(rows, metric) for metric in REPORT_METRICS],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_revision": get_git_revision(),
            "config": config,
            "manifest": str(Path(manifest_path).resolve()),
            "paper_count": len({row["paper_id"] for row in rows}),
            "image_count": len(rows),
        }, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir
```

Add an `argparse` CLI with required `--manifest` and `--output-dir` arguments. The CLI must call `run(arguments.manifest, arguments.output_dir)` until Task 8 adds the required freeze argument.

- [ ] **Step 4: Run runner and existing experiment tests**

Run:

```powershell
pytest tests/test_blind_runner.py tests/test_wiring_experiments.py -v
```

Expected: all tests pass and no root-level historical result is modified.

- [ ] **Step 5: Commit the runner**

```powershell
git add run_blind_benchmark.py tests/test_blind_runner.py
git commit -m "feat: add LLM-free blind benchmark runner"
```

### Task 7: Add detector mAP evaluation for each capture domain

**Files:**
- Create: `run_detector_eval.py`
- Modify: `tests/test_blind_runner.py`

- [ ] **Step 1: Write a test using a fake Ultralytics model**

Append this test to `tests/test_blind_runner.py`:

```python
from types import SimpleNamespace

import run_detector_eval


def test_detector_entry_writes_overall_and_per_class_metrics(tmp_path, monkeypatch):
    calls = {}

    class FakeModel:
        names = {0: "resistor"}

        def val(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(B)": 0.9,
                    "metrics/recall(B)": 0.8,
                    "metrics/mAP50(B)": 0.85,
                    "metrics/mAP50-95(B)": 0.6,
                },
                box=SimpleNamespace(ap_class_index=[0], ap50=[0.85], ap=[0.60]),
            )

    monkeypatch.setattr(run_detector_eval, "YOLO", lambda _path: FakeModel())
    output = tmp_path / "detector-run"

    metrics = run_detector_eval.evaluate_detector(
        "model.pt", "controlled.yaml", output, "controlled",
    )

    assert calls["split"] == "test"
    assert calls["conf"] == 0.001
    assert calls["plots"] is True
    assert calls["project"] == str(output)
    assert calls["name"] == "controlled"
    assert metrics["map50"] == 0.85
    assert metrics["per_class"] == [
        {"class_id": 0, "class_name": "resistor", "ap50": 0.85, "ap50_95": 0.60},
    ]
    saved = json.loads((output / "detector_metrics.json").read_text(encoding="utf-8"))
    assert saved["precision"] == 0.9
    assert saved["per_class"] == metrics["per_class"]
```

- [ ] **Step 2: Run the detector-entry test and verify failure**

Run:

```powershell
pytest tests/test_blind_runner.py -k detector -v
```

Expected: import failure for `run_detector_eval`.

- [ ] **Step 3: Implement the detector-only evaluation entry**

Create `run_detector_eval.py`:

```python
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def evaluate_detector(model_path, data_yaml, output_dir, domain):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(model_path))
    result = model.val(
        data=str(data_yaml), split="test", conf=0.001, iou=0.7,
        plots=True, project=str(output_dir), name=domain, exist_ok=False,
    )
    metrics = {name: float(result.results_dict[key]) for name, key in KEYS.items()}
    per_class = [
        {
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)],
            "ap50": float(result.box.ap50[index]),
            "ap50_95": float(result.box.ap[index]),
        }
        for index, class_id in enumerate(result.box.ap_class_index)
    ]
    (output_dir / "detector_metrics.json").write_text(
        json.dumps({"domain": domain, **metrics, "per_class": per_class}, indent=2),
        encoding="utf-8",
    )
    return {**metrics, "per_class": per_class}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", choices=("controlled", "handheld"), required=True)
    arguments = parser.parse_args()
    evaluate_detector(arguments.model, arguments.data, arguments.output_dir, arguments.domain)
```

- [ ] **Step 4: Run the detector-entry test**

Run:

```powershell
pytest tests/test_blind_runner.py -k detector -v
```

Expected: PASS without loading the real model.

- [ ] **Step 5: Commit detector evaluation**

```powershell
git add run_detector_eval.py tests/test_blind_runner.py
git commit -m "feat: add domain-specific detector evaluation"
```

### Task 8: Freeze and verify code, models, configuration and data

**Files:**
- Create: `src/evaluation/freeze.py`
- Create: `tests/test_freeze_manifest.py`
- Modify: `run_blind_benchmark.py`
- Modify: `run_detector_eval.py`
- Modify: `tests/test_blind_runner.py`

- [ ] **Step 1: Write a file-tamper test**

Create `tests/test_freeze_manifest.py`:

```python
import pytest

from src.evaluation.freeze import build_freeze_manifest, verify_freeze_manifest


def test_freeze_manifest_detects_changed_file(tmp_path):
    tracked = tmp_path / "image.jpg"
    tracked.write_bytes(b"original")
    manifest = build_freeze_manifest(
        files=[tracked], models=[], config={"skip_llm": True}, git_revision="abc123",
    )
    verify_freeze_manifest(manifest)
    tracked.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_freeze_manifest(manifest)
```

- [ ] **Step 2: Run the test and verify the freeze module is absent**

Run:

```powershell
pytest tests/test_freeze_manifest.py -v
```

Expected: import failure for `src.evaluation.freeze`.

- [ ] **Step 3: Implement hashing and verification**

Create `src/evaluation/freeze.py`:

```python
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_freeze_manifest(files, models, config, git_revision):
    paths = [Path(path).resolve() for path in [*files, *models]]
    return {
        "schema_version": "1.0",
        "git_revision": git_revision,
        "config": config,
        "files": {str(path): sha256_file(path) for path in sorted(paths)},
    }


def verify_freeze_manifest(manifest):
    if manifest["config"].get("skip_llm") is not True:
        raise RuntimeError("freeze config must disable LLM")
    for raw_path, expected in manifest["files"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"missing frozen file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {path}")


def write_freeze_manifest(path, files, models, config, git_revision):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite freeze manifest: {path}")
    manifest = build_freeze_manifest(files, models, config, git_revision)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_freeze_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 4: Require a verified freeze manifest before a blind run**

Add `--freeze-manifest` to `run_blind_benchmark.py`, change the signature to `run(manifest_path, output_dir, freeze_manifest_path)`, and execute this preflight before creating the output directory:

```python
config = dict(build_ablation_configs()["Ours"])
manifest_rows = load_manifest(manifest_path, require_pairs=True)
freeze = load_freeze_manifest(freeze_manifest_path)
verify_freeze_manifest(freeze)
if freeze["git_revision"] != get_git_revision():
    raise RuntimeError("frozen Git revision does not match HEAD")
if freeze["config"] != config:
    raise RuntimeError("frozen recognition config does not match Ours")

required_paths = {
    str(Path(manifest_path).resolve()),
    str(DETECTOR_WEIGHTS.resolve()),
    str(OCR_WEIGHTS.resolve()),
    *(str(row.image_path.resolve()) for row in manifest_rows),
    *(str(row.gt_path.resolve()) for row in manifest_rows),
}
missing = required_paths - set(freeze["files"])
if missing:
    raise RuntimeError(f"freeze manifest omits {len(missing)} required inputs")

output_dir = resolve_output_dir(output_dir)
```

Update `tests/test_blind_runner.py` to create a temporary freeze JSON containing the manifest, both fake images, both fake GT files and two temporary model files; monkeypatch the model path constants in `run_blind_benchmark`; pass that freeze path as the third argument; add one assertion that changing either fake image makes the runner fail before `process_image` is called.

Apply the same gate to `run_detector_eval.py`: add a required `--freeze-manifest` argument and change `evaluate_detector()` to accept it. Before creating the output directory, run:

```python
freeze = load_freeze_manifest(freeze_manifest_path)
verify_freeze_manifest(freeze)
domain_root = Path(data_yaml).resolve().parent / domain
required = {
    str(Path(model_path).resolve()),
    str(Path(data_yaml).resolve()),
    *(str(path.resolve()) for path in domain_root.rglob("*") if path.is_file()),
}
missing = required - set(freeze["files"])
if missing:
    raise RuntimeError(f"freeze manifest omits {len(missing)} detector inputs")
```

Extend the fake-detector test so a changed label file causes failure before `YOLO()` is constructed.

- [ ] **Step 5: Run freeze and runner tests**

Run:

```powershell
pytest tests/test_freeze_manifest.py tests/test_blind_runner.py -v
```

Expected: all tests pass; both runners refuse a missing or stale freeze manifest, and the recognition runner also rejects an LLM-enabled config.

- [ ] **Step 6: Commit freeze enforcement**

```powershell
git add src/evaluation/freeze.py run_blind_benchmark.py run_detector_eval.py tests/test_freeze_manifest.py tests/test_blind_runner.py
git commit -m "feat: freeze blind benchmark inputs"
```

### Task 9: Run the complete infrastructure verification

**Files:**
- Verify only; no source change unless a preceding test exposes a defect.

- [ ] **Step 1: Run focused visual and evaluation tests**

```powershell
pytest tests/test_wiring_experiments.py tests/test_wire_mask.py tests/test_component_schema.py tests/test_blind_schema.py tests/test_publication_metrics.py tests/test_blind_runner.py tests/test_freeze_manifest.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete repository suite**

```powershell
pytest tests/ -v
```

Expected: all non-LLM tests pass. Any pre-existing LLM-provider failures must be recorded separately and must not be reclassified as recognition failures.

- [ ] **Step 3: Validate the real 50-image schedule without running models**

```powershell
python -c "from pathlib import Path; import run_experiments as r; rows=r.get_image_list(Path('benchmark')); print(len(rows)); print([x[0] for x in rows if x[0] in {'C170_D1_P1','C171_D1_P1','C274_D1_P1'}])"
```

Expected: the printed count equals the number of `benchmark/result/*_gt.txt` files and the printed list contains all three named circuits.

- [ ] **Step 4: Record verification without an empty commit**

If no source changed, record the passing commands in the implementation handoff and do not create a commit. If a test exposes a defect, return to the task that owns the failing file, add a regression test there, repeat that task's exact `git add` command, and commit with that task's stated message before rerunning this complete verification.

## Execution order and exit criteria

Execute Tasks 1–9 in order. This plan is complete when:

- no GT-backed benchmark image can be silently omitted;
- BJT uses one family name and B/C/E ordering across detection, annotation and evaluation;
- blind GT and manifest validation reject incomplete or unpaired samples;
- all recognition metrics have hand-checkable unit tests;
- controlled and handheld results are separate and paired by `paper_id`;
- a blind run cannot begin without verified file, model, config and Git hashes;
- the complete run remains LLM-free.
