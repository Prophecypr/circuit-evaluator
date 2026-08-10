# Wiring Edge Error Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local tool that assigns one actionable root-cause category to every strict-jj FP/FN on the 50-image development benchmark and produces tabular, visual, and Markdown diagnostics without using LLM, OCR, or the sealed 42-image test set.

**Architecture:** Put topology identity and edge-set construction in a small pure module shared by the existing evaluator and the new analyzer. Put mutually exclusive FN/FP attribution in a second pure module, while a CLI module owns validation, UTF-8/Unicode-safe I/O, skeleton evidence, rendering, provenance, and report generation. The analyzer reads cached strict-jj predictions and traces, so only lightweight skeleton extraction is recomputed from source images.

**Tech Stack:** Python 3.13, dataclasses, pathlib, csv/json/hashlib, OpenCV, NumPy, pytest, existing `unified_pipeline` skeleton helpers.

---

## File map

- Create `src/vision/wiring_topology.py`: component matching, port identity mapping, named networks, and undirected edge inventories.
- Create `src/vision/wiring_error_analysis.py`: trace indexing, FN attribution, FP merge-root attribution, and deterministic summaries.
- Create `src/vision/wiring_error_cli.py`: input validation, skeleton evidence, CSV/JSON/Markdown output, rendering, metadata, and CLI entry point.
- Modify `run_experiments.py`: reuse `wiring_topology` so evaluation and diagnosis cannot drift.
- Create `tests/vision/test_wiring_topology.py`: unit tests for matching, port identity, and edge inventory.
- Create `tests/vision/test_wiring_error_analysis.py`: unit tests for every FN category and FP/cascade behavior.
- Create `tests/vision/test_wiring_error_cli.py`: validation, output, Unicode image, and integration tests.
- Modify `docs/wiring_reliability_experiment_2026-08-10.md`: link the completed attribution report and record the selected next root cause after the real run.

### Task 1: Extract one shared topology and edge-inventory definition

**Files:**
- Create: `src/vision/wiring_topology.py`
- Create: `tests/vision/test_wiring_topology.py`
- Modify: `run_experiments.py:120-340`
- Test: `tests/test_wiring_experiments.py`

- [ ] **Step 1: Write failing tests for deterministic identity mapping and edge expansion**

```python
# tests/vision/test_wiring_topology.py
from src.vision.wiring_topology import build_edge_inventory, groups_to_edges


def test_groups_to_edges_is_undirected_and_order_independent():
    groups = [{"R1.1", "R2.2", "C1.+"}]
    assert groups_to_edges(groups) == {
        ("C1.+", "R1.1"),
        ("C1.+", "R2.2"),
        ("R1.1", "R2.2"),
    }


def test_build_edge_inventory_preserves_unmatched_component_and_port_sets():
    pipeline = {
        "components": [{"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]}],
        "raw_groups": [[[0, 0]]],
    }
    detections = [
        {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]], "labels": ["1"], "designator": "R1"},
        {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]], "labels": ["2"], "designator": "R2"},
    ]
    inventory = build_edge_inventory(pipeline, [[("R1", "1"), ("R2", "2")]], detections)
    assert inventory.gt_edges == {("R1.1", "R2.2")}
    assert inventory.pred_edges == set()
    assert inventory.unmatched_detection_components == {1}
    assert inventory.unmapped_gt_ports == {"R2.2"}
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `python -m pytest tests/vision/test_wiring_topology.py -q`

Expected: collection fails with `ModuleNotFoundError: src.vision.wiring_topology`.

- [ ] **Step 3: Implement the pure topology module**

```python
# src/vision/wiring_topology.py
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EdgeInventory:
    gt_groups: tuple[frozenset[str], ...]
    pred_groups: tuple[frozenset[str], ...]
    gt_edges: frozenset[tuple[str, str]]
    pred_edges: frozenset[tuple[str, str]]
    tp_edges: frozenset[tuple[str, str]]
    fp_edges: frozenset[tuple[str, str]]
    fn_edges: frozenset[tuple[str, str]]
    pipeline_port_ids: dict[tuple[int, int], str]
    port_points: dict[str, tuple[int, int]]
    component_matches: dict[int, int]
    unmatched_detection_components: frozenset[int]
    unmapped_gt_ports: frozenset[str]


def groups_to_edges(groups):
    edges = set()
    for group in groups:
        ports = sorted(set(group))
        edges.update((ports[i], ports[j]) for i in range(len(ports)) for j in range(i + 1, len(ports)))
    return edges


def match_components(pipeline_components, detection_components, iou_threshold=0.3):
    matches, used = {}, set()
    for pipeline_index, pipeline_component in enumerate(pipeline_components):
        px1, py1, px2, py2 = pipeline_component.get("xyxy", [0, 0, 0, 0])
        best = (0.0, None)
        for detection_index, detection_component in enumerate(detection_components):
            if detection_index in used:
                continue
            dx1, dy1, dx2, dy2 = detection_component["xyxy"]
            ix1, iy1, ix2, iy2 = max(px1, dx1), max(py1, dy1), min(px2, dx2), min(py2, dy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            intersection = (ix2 - ix1) * (iy2 - iy1)
            union = (px2 - px1) * (py2 - py1) + (dx2 - dx1) * (dy2 - dy1) - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold and iou > best[0]:
                best = (iou, detection_index)
        if best[1] is not None:
            matches[pipeline_index] = best[1]
            used.add(best[1])
    return matches


def build_edge_inventory(pipeline_result, gt_groups, detection_components, port_match_radius=60):
    pipeline_components = pipeline_result["components"]
    matches = match_components(pipeline_components, detection_components)
    pipeline_port_ids, port_points = {}, {}
    for pipeline_index, component in enumerate(pipeline_components):
        detection_index = matches.get(pipeline_index)
        if detection_index is None:
            continue
        detection = detection_components[detection_index]
        pipeline_ports, detection_ports = component.get("ports", []), detection.get("ports", [])
        if len(pipeline_ports) == len(detection_ports):
            port_pairs = [(index, index) for index in range(len(pipeline_ports))]
        else:
            port_pairs = []
            for pipeline_port_index, (px, py) in enumerate(pipeline_ports):
                candidates = [
                    (math.hypot(px-dpx, py-dpy), detection_port_index)
                    for detection_port_index, (dpx, dpy) in enumerate(detection_ports)
                    if math.hypot(px-dpx, py-dpy) < port_match_radius
                ]
                if candidates:
                    port_pairs.append((pipeline_port_index, min(candidates)[1]))
        for pipeline_port_index, detection_port_index in port_pairs:
            labels = detection.get("labels", [])
            label = labels[detection_port_index] if detection_port_index < len(labels) else "?"
            port_id = f"{detection['designator']}.{label}"
            pipeline_port_ids[(pipeline_index, pipeline_port_index)] = port_id
            port_points[port_id] = tuple(map(int, pipeline_ports[pipeline_port_index]))

    gt_named = tuple(
        frozenset(f"{designator}.{label}" for designator, label in group)
        for group in gt_groups
    )
    pred_named = []
    for raw_group in pipeline_result.get("raw_groups", []):
        named = frozenset(
            pipeline_port_ids[(int(component_index), int(port_index))]
            for component_index, port_index in raw_group
            if (int(component_index), int(port_index)) in pipeline_port_ids
        )
        if len(named) >= 2:
            pred_named.append(named)
    gt_edges, pred_edges = groups_to_edges(gt_named), groups_to_edges(pred_named)
    all_gt_ports = set().union(*gt_named) if gt_named else set()
    return EdgeInventory(
        gt_groups=gt_named,
        pred_groups=tuple(pred_named),
        gt_edges=frozenset(gt_edges),
        pred_edges=frozenset(pred_edges),
        tp_edges=frozenset(gt_edges & pred_edges),
        fp_edges=frozenset(pred_edges - gt_edges),
        fn_edges=frozenset(gt_edges - pred_edges),
        pipeline_port_ids=pipeline_port_ids,
        port_points=port_points,
        component_matches=matches,
        unmatched_detection_components=frozenset(set(range(len(detection_components))) - set(matches.values())),
        unmapped_gt_ports=frozenset(all_gt_ports - set(pipeline_port_ids.values())),
    )
```

The module must not import OpenCV or access files.

- [ ] **Step 4: Make `run_experiments.evaluate()` consume `EdgeInventory`**

```python
from src.vision.wiring_topology import build_edge_inventory, match_components


def evaluate(pipeline_result, gt_groups, det_comps):
    inventory = build_edge_inventory(
        pipeline_result, gt_groups, det_comps, port_match_radius=PORT_MATCH_RADIUS,
    )
    gt_sets = list(inventory.gt_groups)
    pred_groups = list(inventory.pred_groups)
    gt_edges = set(inventory.gt_edges)
    pred_edges = set(inventory.pred_edges)
    tp_edges = len(inventory.tp_edges)
    fp_edges = len(inventory.fp_edges)
    fn_edges = len(inventory.fn_edges)
    # Keep the existing group, neighbor, and PRF calculations unchanged below this point.
```

- [ ] **Step 5: Run topology and experiment regression tests**

Run: `python -m pytest tests/vision/test_wiring_topology.py tests/test_wiring_experiments.py -q`

Expected: all tests pass and existing experiment metric fixtures remain unchanged.

- [ ] **Step 6: Commit the shared metric definition**

```powershell
git add -- src/vision/wiring_topology.py run_experiments.py tests/vision/test_wiring_topology.py tests/test_wiring_experiments.py
git commit -m "refactor: share wiring edge inventory"
```

### Task 2: Validate and load exactly the cached 50-image strict-jj run

**Files:**
- Create: `src/vision/wiring_error_cli.py`
- Create: `tests/vision/test_wiring_error_cli.py`

- [ ] **Step 1: Write failing validation tests**

```python
# tests/vision/test_wiring_error_cli.py
import json
import pytest
from src.vision.wiring_error_cli import validate_inputs


def test_validate_inputs_requires_exact_expected_count(tmp_path):
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "image_count": 49,
        "failure_count": 0,
        "configs": {"strict_jj": {"skip_llm": True, "skip_ocr": True}},
        "final_42_image_test_used": False,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 50.*found 49"):
        validate_inputs(tmp_path, expected_count=50)


def test_validate_inputs_rejects_llm_or_final42_use(tmp_path):
    metadata = {
        "image_count": 50,
        "failure_count": 0,
        "configs": {"strict_jj": {"skip_llm": False, "skip_ocr": True}},
        "final_42_image_test_used": False,
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="skip_llm"):
        validate_inputs(tmp_path, expected_count=50)
```

- [ ] **Step 2: Run the tests and verify the loader is missing**

Run: `python -m pytest tests/vision/test_wiring_error_cli.py -q`

Expected: collection fails because `wiring_error_cli` does not exist.

- [ ] **Step 3: Implement fail-closed input validation and case loading**

```python
# src/vision/wiring_error_cli.py
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class AnalysisCase:
    stem: str
    image_path: Path
    gt_path: Path
    detection_path: Path
    prediction_path: Path
    trace_path: Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(run_dir, benchmark_dir="benchmark", expected_count=50):
    run_dir, benchmark_dir = Path(run_dir), Path(benchmark_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("image_count") != expected_count:
        raise RuntimeError(f"expected {expected_count} images, found {metadata.get('image_count')}")
    if metadata.get("failure_count") != 0:
        raise RuntimeError("input run contains failures")
    strict = metadata.get("configs", {}).get("strict_jj", {})
    if strict.get("skip_llm") is not True:
        raise RuntimeError("strict_jj must use skip_llm=true")
    if strict.get("skip_ocr") is not True:
        raise RuntimeError("strict_jj must use skip_ocr=true")
    if metadata.get("final_42_image_test_used") is not False:
        raise RuntimeError("sealed final 42-image test set was used")
    # Join prediction, trace, GT, detection, and exactly one supported image by stem.
    # Reject missing files, duplicates, extra predictions, and a final count other than expected_count.
    return metadata, tuple(cases)
```

- [ ] **Step 4: Add tests for missing trace, duplicate image stem, and output overwrite refusal**

```python
def test_validate_inputs_rejects_missing_trace(seeded_run):
    seeded_run.trace_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing trace"):
        validate_inputs(seeded_run.run_dir, seeded_run.benchmark_dir, expected_count=1)


def test_prepare_output_refuses_nonempty_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output(output, resume=False)
```

- [ ] **Step 5: Run validation tests**

Run: `python -m pytest tests/vision/test_wiring_error_cli.py -q`

Expected: all validation tests pass.

- [ ] **Step 6: Commit validated loading**

```powershell
git add -- src/vision/wiring_error_cli.py tests/vision/test_wiring_error_cli.py
git commit -m "feat: validate wiring attribution inputs"
```

### Task 3: Build trace indices and classify every FN once

**Files:**
- Create: `src/vision/wiring_error_analysis.py`
- Create: `tests/vision/test_wiring_error_analysis.py`
- Modify: `src/vision/wiring_error_cli.py`

- [ ] **Step 1: Write one parametrized failing test for all seven FN categories**

```python
# tests/vision/test_wiring_error_analysis.py
import pytest
from src.vision.wiring_error_analysis import FnEvidence, classify_fn


@pytest.mark.parametrize(("evidence", "expected"), [
    (FnEvidence(component_matched=False), "component_unmatched"),
    (FnEvidence(component_matched=True, port_mapped=False), "port_unmatched"),
    (FnEvidence(component_matched=True, port_mapped=True, skeleton_near_port=False), "no_port_skeleton"),
    (FnEvidence(component_matched=True, port_mapped=True, skeleton_near_port=True, trace_reached_network=False), "skeleton_break"),
    (FnEvidence(component_matched=True, port_mapped=True, skeleton_near_port=True,
                trace_reached_network=True, rejection_reason="ambiguous_crossing"), "candidate_rejected"),
    (FnEvidence(component_matched=True, port_mapped=True, skeleton_near_port=True,
                trace_reached_network=True, candidate_generated=False), "candidate_not_generated"),
    (FnEvidence(component_matched=True, port_mapped=True, skeleton_near_port=True,
                trace_reached_network=True, candidate_generated=True), "network_split_unresolved"),
])
def test_classify_fn_uses_mutually_exclusive_priority(evidence, expected):
    assert classify_fn(evidence).category == expected
```

- [ ] **Step 2: Run the test and verify the analysis module is missing**

Run: `python -m pytest tests/vision/test_wiring_error_analysis.py -q`

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement explicit FN evidence and priority classification**

```python
# src/vision/wiring_error_analysis.py
from dataclasses import dataclass


@dataclass(frozen=True)
class FnEvidence:
    component_matched: bool
    port_mapped: bool = False
    skeleton_near_port: bool | None = None
    trace_reached_network: bool | None = None
    candidate_generated: bool | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ErrorAttribution:
    error_type: str
    edge: tuple[str, str]
    category: str
    secondary_reason: str = ""
    root_event_id: str = ""
    is_root: bool = True


def classify_fn(evidence, edge=("", "")):
    if not evidence.component_matched:
        return ErrorAttribution("FN", edge, "component_unmatched")
    if not evidence.port_mapped:
        return ErrorAttribution("FN", edge, "port_unmatched")
    if evidence.skeleton_near_port is False:
        return ErrorAttribution("FN", edge, "no_port_skeleton")
    if evidence.trace_reached_network is False:
        return ErrorAttribution("FN", edge, "skeleton_break")
    if evidence.rejection_reason:
        return ErrorAttribution("FN", edge, "candidate_rejected", evidence.rejection_reason)
    if evidence.candidate_generated is False:
        return ErrorAttribution("FN", edge, "candidate_not_generated")
    return ErrorAttribution("FN", edge, "network_split_unresolved")
```

- [ ] **Step 4: Add deterministic trace indexing and lightweight skeleton evidence**

```python
def index_trace_events(trace_payload):
    indexed = {}
    for event_index, event in enumerate(trace_payload.get("events", [])):
        source = event.get("source", {})
        key = (source.get("component_index"), source.get("port_index"))
        indexed.setdefault(key, []).append({"event_id": f"E{event_index:05d}", **event})
    return indexed


def skeleton_near_port(skeleton, point, radius=15):
    x, y = map(int, point)
    height, width = skeleton.shape[:2]
    crop = skeleton[max(0, y-radius):min(height, y+radius+1),
                    max(0, x-radius):min(width, x+radius+1)]
    return bool(crop.size and (crop > 0).any())
```

In `wiring_error_cli.py`, decode Chinese paths with `Path.read_bytes()` plus `cv2.imdecode`, then call the same `_extract_component_masked_skeleton()` helper and scaling policy used by `strict_jj`. Do not call `process_image`, YOLO, OCR, or LLM.

- [ ] **Step 5: Test priority, trace indexing, and skeleton evidence**

Run: `python -m pytest tests/vision/test_wiring_error_analysis.py -q`

Expected: all FN and skeleton tests pass.

- [ ] **Step 6: Commit FN attribution**

```powershell
git add -- src/vision/wiring_error_analysis.py src/vision/wiring_error_cli.py tests/vision/test_wiring_error_analysis.py
git commit -m "feat: attribute wiring false negatives"
```

### Task 4: Identify network-merge roots and cascade FP edges

**Files:**
- Modify: `src/vision/wiring_error_analysis.py`
- Modify: `tests/vision/test_wiring_error_analysis.py`

- [ ] **Step 1: Write failing tests for a single merge root and cascade edges**

```python
def test_network_merge_marks_one_root_and_remaining_cross_edges_as_cascade():
    gt_groups = [frozenset({"R1.1", "R2.1"}), frozenset({"C1.1", "C2.1"})]
    pred_groups = [frozenset({"R1.1", "R2.1", "C1.1", "C2.1"})]
    physical_edges = [
        {"event_id": "E00001", "kind": "p2j", "ports": ("R1.1", "R2.1")},
        {"event_id": "E00002", "kind": "jj", "ports": ("R2.1", "C1.1")},
        {"event_id": "E00003", "kind": "p2j", "ports": ("C1.1", "C2.1")},
    ]
    rows = attribute_fp_edges(gt_groups, pred_groups, physical_edges)
    roots = [row for row in rows if row.is_root]
    cascades = [row for row in rows if row.category == "cascade_fp"]
    assert [(row.category, row.root_event_id) for row in roots] == [("wrong_junction_merge", "E00002")]
    assert cascades
    assert {row.root_event_id for row in cascades} == {"E00002"}


def test_ambiguous_merge_has_explicit_unattributed_root():
    rows = attribute_fp_edges(
        [frozenset({"R1.1", "R2.1"}), frozenset({"C1.1", "C2.1"})],
        [frozenset({"R1.1", "R2.1", "C1.1", "C2.1"})],
        [],
    )
    assert any(row.category == "unattributed_merge" and row.is_root for row in rows)
```

- [ ] **Step 2: Run the two tests and verify the function is missing**

Run: `python -m pytest tests/vision/test_wiring_error_analysis.py -k "merge or cascade" -q`

Expected: fails because `attribute_fp_edges` is undefined.

- [ ] **Step 3: Implement physical accepted-edge reconstruction**

```python
def build_physical_edges(prediction, trace_index, port_ids):
    edges = []
    for component_index, port_index, junction_x, junction_y in prediction.get("p2j_connections", []):
        edges.append({
            "kind": "p2j",
            "source": port_ids.get((component_index, port_index), ""),
            "target": f"J({junction_x},{junction_y})",
        })
    for x1, y1, x2, y2 in prediction.get("jj_connections", []):
        edges.append({"kind": "jj", "source": f"J({x1},{y1})", "target": f"J({x2},{y2})"})
    # Attach the earliest coordinate-compatible accepted trace event and stable event_id.
    # P2P/LOS/close-port events come from accepted trace events even when not in p2j/jj lists.
    return tuple(edges)
```

- [ ] **Step 4: Implement bridge-based root selection**

For each predicted network containing ports from at least two GT networks, build its physical port/junction graph. Remove each accepted physical edge in trace order and recompute connected components. An edge is a root candidate when its removal separates the different GT-network labels. Choose the unique earliest candidate; map `p2j`, `p2p`/`los`/`close_port`, `jj`, crossing evidence, and component-crossing evidence to the categories in the design. If no unique candidate exists, create one `unattributed_merge` root. Label every other FP port-pair in that merged network `cascade_fp` with the selected `root_event_id`.

- [ ] **Step 5: Run the complete analysis unit suite**

Run: `python -m pytest tests/vision/test_wiring_error_analysis.py -q`

Expected: every FN, merge-root, local FP, and cascade test passes.

- [ ] **Step 6: Commit FP root attribution**

```powershell
git add -- src/vision/wiring_error_analysis.py tests/vision/test_wiring_error_analysis.py
git commit -m "feat: identify wiring merge root causes"
```

### Task 5: Produce deterministic files, diagnostics, and provenance

**Files:**
- Modify: `src/vision/wiring_error_cli.py`
- Modify: `tests/vision/test_wiring_error_cli.py`

- [ ] **Step 1: Write a failing integration test for all required artifacts**

```python
def test_run_analysis_writes_reconciled_artifacts(seeded_run, tmp_path):
    output = run_analysis(
        seeded_run.run_dir,
        seeded_run.benchmark_dir,
        tmp_path / "analysis",
        expected_count=1,
        expected_counts={"tp": 1, "fp": 1, "fn": 1},
        worst_count=1,
    )
    assert (output / "edge_errors.csv").is_file()
    assert (output / "image_summary.csv").is_file()
    assert (output / "category_summary.json").is_file()
    assert (output / "wiring_error_report.md").is_file()
    assert (output / "run_metadata.json").is_file()
    assert len(list((output / "annotated_worst10").glob("*.jpg"))) == 1
```

- [ ] **Step 2: Run the integration test and verify artifact generation is missing**

Run: `python -m pytest tests/vision/test_wiring_error_cli.py::test_run_analysis_writes_reconciled_artifacts -q`

Expected: fails because `run_analysis` is undefined.

- [ ] **Step 3: Implement CSV/JSON aggregation with fail-closed reconciliation**

```python
EDGE_FIELDS = [
    "image", "error_type", "port_a", "port_b", "category", "secondary_reason",
    "is_root", "root_event_id", "pred_network", "gt_network_a", "gt_network_b",
]


def reconcile_counts(image_rows, error_rows, expected_counts):
    actual = {
        "tp": sum(int(row["edge_tp"]) for row in image_rows),
        "fp": sum(row["error_type"] == "FP" for row in error_rows),
        "fn": sum(row["error_type"] == "FN" for row in error_rows),
    }
    if actual != expected_counts:
        raise RuntimeError(f"edge reconciliation failed: expected {expected_counts}, got {actual}")
    return actual
```

Write CSV with `encoding="utf-8-sig"`, JSON/Markdown with `encoding="utf-8"`, stable image/category ordering, and atomic temporary-file replacement inside the dedicated output directory.

- [ ] **Step 4: Implement Unicode-safe diagnostic rendering**

Read images with `np.frombuffer(path.read_bytes(), np.uint8)` and `cv2.imdecode`. Write with `cv2.imencode(path.suffix, image)[1].tofile(str(path))`. Draw TP green, FP orange, FN red, root evidence purple, and a legend inside the image. Choose the worst images deterministically by `(edge_f1 ascending, fp+fn descending, stem ascending)`.

- [ ] **Step 5: Write metadata and resume provenance validation**

`run_metadata.json` records source run path, source metadata SHA-256, all 50 prediction/trace/GT/detection/image hashes, Git revision, expected and actual TP/FP/FN, category counts, generation time, `llm_used=false`, `ocr_used=false`, and `final_42_image_test_used=false`. `--resume` is accepted only when the entire stored provenance object matches the current inputs.

- [ ] **Step 6: Run CLI tests including a Chinese absolute path**

Run: `python -m pytest tests/vision/test_wiring_error_cli.py -q`

Expected: validation, overwrite, resume, reconciliation, artifact, and Unicode rendering tests all pass.

- [ ] **Step 7: Commit reporting and rendering**

```powershell
git add -- src/vision/wiring_error_cli.py tests/vision/test_wiring_error_cli.py
git commit -m "feat: report wiring error attribution"
```

### Task 6: Add CLI command and run the real 50-image local analysis

**Files:**
- Modify: `src/vision/wiring_error_cli.py`
- Modify: `tests/vision/test_wiring_error_cli.py`
- Output only: `results/wiring_error_attribution_20260811/`

- [ ] **Step 1: Write a failing CLI parser test**

```python
def test_cli_defaults_to_sealed_safe_local_analysis():
    args = build_parser().parse_args([])
    assert args.run_dir == Path("results/wiring_reliability_full_20260810_merged")
    assert args.benchmark_dir == Path("benchmark")
    assert args.output_dir == Path("results/wiring_error_attribution_20260811")
    assert args.expected_count == 50
    assert args.expected_tp == 305
    assert args.expected_fp == 537
    assert args.expected_fn == 1002
```

- [ ] **Step 2: Run the parser test and verify it fails**

Run: `python -m pytest tests/vision/test_wiring_error_cli.py::test_cli_defaults_to_sealed_safe_local_analysis -q`

Expected: fails because `build_parser` is undefined.

- [ ] **Step 3: Implement the command-line entry point**

```python
def build_parser():
    parser = argparse.ArgumentParser(description="Attribute wiring FP/FN root causes")
    parser.add_argument("--run-dir", type=Path, default=Path("results/wiring_reliability_full_20260810_merged"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/wiring_error_attribution_20260811"))
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--expected-tp", type=int, default=305)
    parser.add_argument("--expected-fp", type=int, default=537)
    parser.add_argument("--expected-fn", type=int, default=1002)
    parser.add_argument("--worst-count", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_analysis(
        args.run_dir, args.benchmark_dir, args.output_dir,
        expected_count=args.expected_count,
        expected_counts={"tp": args.expected_tp, "fp": args.expected_fp, "fn": args.expected_fn},
        worst_count=args.worst_count, resume=args.resume,
    )
```

- [ ] **Step 4: Run focused and full regression tests before real data**

Run: `python -m pytest tests/vision/test_wiring_topology.py tests/vision/test_wiring_error_analysis.py tests/vision/test_wiring_error_cli.py tests/test_wiring_experiments.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run the 50-image attribution locally**

Run: `python -m src.vision.wiring_error_cli`

Expected terminal summary:

```text
images=50 failures=0
TP=305 FP=537 FN=1002 reconciled=true
LLM=false OCR=false final42=false
```

- [ ] **Step 6: Verify required files and exact row reconciliation**

Run:

```powershell
python -c "import csv,json,pathlib; p=pathlib.Path(r'results/wiring_error_attribution_20260811'); rows=list(csv.DictReader((p/'edge_errors.csv').open(encoding='utf-8-sig'))); m=json.loads((p/'run_metadata.json').read_text(encoding='utf-8')); print(len(rows),m['actual_counts'],len(list((p/'annotated_worst10').glob('*.jpg'))))"
```

Expected: `1539 {'tp': 305, 'fp': 537, 'fn': 1002} 10`.

- [ ] **Step 7: Inspect all 10 diagnostic overlays**

Open every file under `results/wiring_error_attribution_20260811/annotated_worst10/`. Confirm that line colors match the legend, root markers are visible, and no Chinese-path read/write failures occurred. Record any clearly wrong automatic categories in `wiring_error_report.md` under an `人工抽查` section without changing GT.

- [ ] **Step 8: Commit the runnable CLI**

```powershell
git add -- src/vision/wiring_error_cli.py tests/vision/test_wiring_error_cli.py
git commit -m "feat: run local wiring error analysis"
```

Do not add the `results/` directory to Git.

### Task 7: Publish the diagnosis and select exactly one next algorithm target

**Files:**
- Modify: `docs/wiring_reliability_experiment_2026-08-10.md`
- Read only: `results/wiring_error_attribution_20260811/category_summary.json`
- Read only: `results/wiring_error_attribution_20260811/wiring_error_report.md`

- [ ] **Step 1: Add the reconciled attribution table to the experiment document**

Append a section containing every category's root count, affected edge count, affected image count, and percentage. State the largest actionable root cause exactly as generated; do not choose a different category by visual impression alone.

- [ ] **Step 2: Define the next-stage gate from measured data**

The document must identify one and only one primary algorithm target. The next change is accepted only if a fresh 50-image comparison has edge F1 greater than `0.2838529548627269` and edge Precision no more than 1.0 percentage point below `0.36223277909738716`. The sealed 42 images remain untouched until that gate passes and code/config are locked.

- [ ] **Step 3: Run final verification**

Run:

```powershell
python -m pytest tests/vision/test_wiring_topology.py tests/vision/test_wiring_error_analysis.py tests/vision/test_wiring_error_cli.py tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py tests/vision/ocr_v2 -q
python -m py_compile src/vision/wiring_topology.py src/vision/wiring_error_analysis.py src/vision/wiring_error_cli.py run_experiments.py
git diff --check
```

Expected: all tests and compilation pass; `git diff --check` prints no error.

- [ ] **Step 4: Commit the measured diagnosis**

```powershell
git add -- docs/wiring_reliability_experiment_2026-08-10.md
git commit -m "docs: record wiring error root causes"
```

- [ ] **Step 5: Report the next action without starting it silently**

Report the dominant category, its share, the exact local output directory, test evidence, and the single proposed algorithm change. Begin the second-stage algorithm design only after confirming that the proposed change directly addresses the measured dominant category.
