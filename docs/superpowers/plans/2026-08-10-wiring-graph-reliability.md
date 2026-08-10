# Wiring Graph Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise publication-oriented wiring-edge F1 by making Terminal semantics correct, exposing every graph decision, and replacing permissive distance connections with skeleton-verified P2J/JJ construction.

**Architecture:** Keep YOLO, CRNN, and the detector-first junction flow unchanged. Add a focused `src/vision/wiring_graph.py` module containing pure trace, candidate, event, color, and rendering helpers; the existing `unified_pipeline.py` calls those helpers behind independent config switches. Extend the experiment runner with cumulative 50-image ablations and preserve the sealed 42-image final test.

**Tech Stack:** Python 3.13, OpenCV, NumPy, Ultralytics YOLO, pytest, JSON/CSV experiment artifacts.

---

## File map

- Create `src/vision/wiring_graph.py`: pure wiring events, terminal conversion, skeleton tracing, strict P2J/JJ decisions, network colors, and graph rendering.
- Create `tests/vision/test_wiring_graph.py`: synthetic unit tests for all six behaviors.
- Modify `src/vision/unified_pipeline.py`: config gates and integration with the pure wiring helpers.
- Modify `src/vision/joint_benchmark.py`: persist wiring traces and draw colored accepted networks.
- Modify `run_experiments.py`: cumulative wiring-reliability ablation configurations and stage summaries.
- Modify `tests/test_wiring_experiments.py`: configuration isolation and result-column tests.
- Modify `tests/vision/test_joint_benchmark.py`: trace sidecar and graph-rendering integration tests.
- Create `docs/wiring_reliability_experiment_2026-08-10.md`: final 50-image results and retained/rejected changes.

### Task 0: Freeze the verified joint-benchmark baseline

**Files:**
- Commit existing: `run_experiments.py`
- Commit existing: `run_joint_benchmark.py`
- Commit existing: `src/vision/unified_pipeline.py`
- Commit existing: `src/vision/joint_benchmark.py`
- Commit existing: `tests/test_wiring_experiments.py`
- Commit existing: `tests/vision/test_joint_benchmark.py`
- Commit existing: `docs/joint_benchmark_seed3407_freeze_2026-08-10.json`
- Commit existing: `docs/superpowers/plans/2026-08-10-joint-benchmark.md`
- Commit existing: `docs/superpowers/specs/2026-08-10-joint-benchmark-design.md`

- [ ] **Step 1: Verify the current baseline before changing wiring behavior**

Run:

```powershell
python -m pytest tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py tests/vision/ocr_v2 -q
python -m py_compile src/vision/joint_benchmark.py run_joint_benchmark.py run_experiments.py src/vision/unified_pipeline.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 2: Commit only the verified baseline files**

```powershell
git add -- run_experiments.py run_joint_benchmark.py src/vision/unified_pipeline.py src/vision/joint_benchmark.py tests/test_wiring_experiments.py tests/vision/test_joint_benchmark.py docs/joint_benchmark_seed3407_freeze_2026-08-10.json docs/superpowers/plans/2026-08-10-joint-benchmark.md docs/superpowers/specs/2026-08-10-joint-benchmark-design.md
git commit -m "feat: add strict joint vision benchmark"
```

Expected: unrelated LLM scoring-pack and reference-pack files remain unstaged.

### Task 1: Add trace events and deterministic network colors

**Files:**
- Create: `src/vision/wiring_graph.py`
- Create: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Write failing trace and color tests**

```python
from src.vision.wiring_graph import WiringTrace, network_color


def test_wiring_trace_counts_acceptance_by_stage_and_reason():
    trace = WiringTrace(enabled=True)
    trace.record("p2j", "p2j", True, "continuous_skeleton_path")
    trace.record("p2j", "p2j", False, "no_skeleton_path")
    assert trace.summary() == {
        "p2j": {
            "candidates": 2,
            "accepted": 1,
            "rejected": 1,
            "reasons": {"continuous_skeleton_path": 1, "no_skeleton_path": 1},
        }
    }


def test_network_color_is_stable_and_network_specific():
    assert network_color(["C1.+", "R1.1"]) == network_color(["R1.1", "C1.+"])
    assert network_color(["C1.+", "R1.1"]) != network_color(["C1.-", "R1.2"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/vision/test_wiring_graph.py -q`

Expected: collection fails because `src.vision.wiring_graph` does not exist.

- [ ] **Step 3: Implement the minimal event and color API**

```python
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any


@dataclass
class WiringEvent:
    stage: str
    kind: str
    accepted: bool
    reason: str
    source: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


class WiringTrace:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: list[WiringEvent] = []

    def record(self, stage, kind, accepted, reason, source=None, target=None, **evidence):
        if self.enabled:
            self.events.append(WiringEvent(stage, kind, accepted, reason, source or {}, target or {}, evidence))

    def summary(self):
        grouped = defaultdict(list)
        for event in self.events:
            grouped[event.stage].append(event)
        return {
            stage: {
                "candidates": len(events),
                "accepted": sum(event.accepted for event in events),
                "rejected": sum(not event.accepted for event in events),
                "reasons": dict(Counter(event.reason for event in events)),
            }
            for stage, events in grouped.items()
        }

    def to_dict(self):
        return {"events": [asdict(event) for event in self.events], "summary": self.summary()}


def network_color(port_ids):
    token = "|".join(sorted(str(port_id) for port_id in port_ids)).encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return tuple(64 + byte % 160 for byte in digest[:3])
```

- [ ] **Step 4: Run GREEN and regression tests**

Run:

```powershell
python -m pytest tests/vision/test_wiring_graph.py -q
python -m pytest tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the trace foundation**

```powershell
git add -- src/vision/wiring_graph.py tests/vision/test_wiring_graph.py
git commit -m "feat: add wiring decision trace foundation"
```

### Task 2: Model Terminal as a one-port component

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Modify: `src/vision/unified_pipeline.py:41-54, 1391-1400`
- Test: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Write failing Terminal conversion tests**

```python
from src.vision.wiring_graph import terminal_component


def test_terminal_component_has_one_center_port_and_stable_semantics():
    component = terminal_component([10, 20, 30, 40], 0.8, index=4)
    assert component["name"] == "Terminal"
    assert component["ports"] == [(20, 30)]
    assert component["port_labels"] == ["T"]
    assert component["raw_name"] == "terminal"


def test_terminal_detection_is_not_added_to_junction_centers_when_enabled():
    from src.vision.wiring_graph import classify_connection_detection
    classified = classify_connection_detection("terminal", [10, 20, 30, 40], 0.8, True, 0)
    assert classified["junction"] is None
    assert classified["component"]["name"] == "Terminal"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/vision/test_wiring_graph.py -q`

Expected: imports fail because the Terminal helpers are missing.

- [ ] **Step 3: Implement Terminal helpers and pipeline config**

Add `use_terminal_components=True` to `DEFAULT_CONFIG`. Implement `terminal_component()` and `classify_connection_detection()` so a `junction` returns its center and no component, while an enabled `terminal` returns a one-port component and no junction. In the YOLO loop, append the returned component with the next index and retain the raw detection for audit.

Required component fields:

```python
{
    "idx": index,
    "name": "Terminal",
    "display": "Terminal",
    "raw_name": "terminal",
    "xyxy": tuple(bbox),
    "cx": center_x,
    "cy": center_y,
    "conf": confidence,
    "value": "",
    "ports": [(center_x, center_y)],
    "port_labels": ["T"],
    "designator": "",
    "label_swap": False,
}
```

Also add `"Terminal": ["T"]` to `PORT_LABELS`, `"Terminal": [(0.5, 0.5)]` to `PORT_POSITIONS`, and `"Terminal": "T"` to `DESIG`.

- [ ] **Step 4: Run GREEN and the full relevant suite**

```powershell
python -m pytest tests/vision/test_wiring_graph.py -q
python -m pytest tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
```

Expected: all tests pass; legacy configs with `use_terminal_components=False` retain the previous behavior.

- [ ] **Step 5: Commit Terminal semantics**

```powershell
git add -- src/vision/wiring_graph.py src/vision/unified_pipeline.py tests/vision/test_wiring_graph.py
git commit -m "fix: model terminals as one-port components"
```

### Task 3: Add strict fallback policy and stage counters

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Modify: `src/vision/unified_pipeline.py:1855-2030`
- Test: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Write failing strict candidate tests**

```python
from src.vision.wiring_graph import accept_p2j_candidate


def test_strict_p2j_rejects_distance_only_candidate():
    decision = accept_p2j_candidate(distance=20, path_found=False, crosses_component=False, strict=True)
    assert decision == (False, "no_skeleton_path")


def test_legacy_p2j_keeps_distance_candidate_for_ablation():
    decision = accept_p2j_candidate(distance=20, path_found=False, crosses_component=False, strict=False)
    assert decision == (True, "legacy_distance_fallback")


def test_p2j_rejects_component_crossing_even_with_skeleton():
    decision = accept_p2j_candidate(distance=20, path_found=True, crosses_component=True, strict=True)
    assert decision == (False, "crosses_component")
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/vision/test_wiring_graph.py -q`

Expected: import fails for `accept_p2j_candidate`.

- [ ] **Step 3: Implement the policy and integrate tracing**

Add `use_wiring_trace=True` and `use_strict_p2j=True` config keys. Construct one `WiringTrace` per image. Route initial P2J, second-pass fallback, aggressive P2P, LOS, force-connect, and close-port decisions through `accept_p2j_candidate`; record every candidate with a stable stage and reason. Under strict mode, Step 6c cannot add a junction connection without `path_found=True`. Legacy mode preserves current behavior for ablation.

- [ ] **Step 4: Run GREEN and verify output contract**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
```

Expected: all tests pass and `_make_pipeline_result` exposes `wiring_trace` with an empty trace accepted for compatibility.

- [ ] **Step 5: Commit strict fallback policy**

```powershell
git add -- src/vision/wiring_graph.py src/vision/unified_pipeline.py tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py
git commit -m "fix: require evidence for fallback port connections"
```

### Task 4: Trace outward from ports to the first reachable anchor

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Modify: `src/vision/unified_pipeline.py:971-1170, 1855-1933`
- Test: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Write failing synthetic skeleton tests**

```python
import numpy as np
from src.vision.wiring_graph import trace_port_to_anchor


def test_trace_port_to_anchor_returns_first_outward_anchor():
    skeleton = np.zeros((21, 41), dtype=np.uint8)
    skeleton[10, 5:36] = 255
    result = trace_port_to_anchor(
        skeleton=skeleton,
        port=(5, 10),
        component_center=(0, 10),
        anchors=[("J1", (20, 10)), ("J2", (35, 10))],
        search_radius=2,
        anchor_radius=2,
        max_steps=100,
    )
    assert result["anchor_id"] == "J1"
    assert result["reason"] == "continuous_skeleton_path"


def test_trace_port_to_anchor_rejects_anchor_behind_component():
    skeleton = np.zeros((21, 41), dtype=np.uint8)
    skeleton[10, 2:20] = 255
    result = trace_port_to_anchor(
        skeleton=skeleton,
        port=(10, 10),
        component_center=(15, 10),
        anchors=[("behind", (2, 10))],
        search_radius=2,
        anchor_radius=2,
        max_steps=100,
    )
    assert result is None
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/vision/test_wiring_graph.py -q`

Expected: import fails for `trace_port_to_anchor`.

- [ ] **Step 3: Implement direction-aware BFS**

The helper finds the closest skeleton pixel to the port, computes the outward unit vector `port - component_center`, rejects initial steps whose dot product is negative, and performs BFS over 8-neighbor skeleton pixels. It returns immediately on the first anchor within `anchor_radius`, including path length and visited-pixel count. Equal-depth anchors are ordered by anchor ID for determinism. No arbitrary nearest-junction jump occurs after a dead end.

Add `use_outward_skeleton_trace=True`. Under this key, P2J candidates come from the first anchor returned by `trace_port_to_anchor`; if no anchor is found, strict mode records `no_skeleton_path` and leaves the port unconnected.

- [ ] **Step 4: Run GREEN and a single-image smoke test**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
python -c "from src.vision.unified_pipeline import process_image; r=process_image('benchmark/C274_D2_P1.jpeg', config={'skip_llm':True,'save_artifacts':False}); print(r['wiring_trace']['summary'])"
```

Expected: tests pass, the smoke test exits 0, and `p2j_trace` appears in the summary.

- [ ] **Step 5: Commit outward tracing**

```powershell
git add -- src/vision/wiring_graph.py src/vision/unified_pipeline.py tests/vision/test_wiring_graph.py
git commit -m "feat: trace ports outward along wire skeletons"
```

### Task 5: Enforce strict JJ paths and crossing semantics

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Modify: `src/vision/unified_pipeline.py:2032-2172`
- Test: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Write failing path and crossing tests**

```python
import numpy as np
from src.vision.wiring_graph import strict_jj_decision


def test_strict_jj_accepts_continuous_aligned_wire():
    skeleton = np.zeros((31, 51), dtype=np.uint8)
    skeleton[15, 5:46] = 255
    accepted, reason = strict_jj_decision(skeleton, (5, 15), (45, 15), detected_junctions=[(5, 15), (45, 15)])
    assert (accepted, reason) == (True, "continuous_skeleton_path")


def test_strict_jj_rejects_blank_gap():
    skeleton = np.zeros((31, 51), dtype=np.uint8)
    skeleton[15, 5:20] = 255
    skeleton[15, 30:46] = 255
    accepted, reason = strict_jj_decision(skeleton, (5, 15), (45, 15), detected_junctions=[(5, 15), (45, 15)])
    assert (accepted, reason) == (False, "no_skeleton_path")


def test_crossing_without_detected_center_does_not_merge_directions():
    skeleton = np.zeros((51, 51), dtype=np.uint8)
    skeleton[25, 5:46] = 255
    skeleton[5:46, 25] = 255
    accepted, reason = strict_jj_decision(
        skeleton, (5, 25), (25, 5), detected_junctions=[(5, 25), (25, 5)],
    )
    assert (accepted, reason) == (False, "ambiguous_crossing")


def test_crossing_with_detected_center_can_turn_and_merge():
    skeleton = np.zeros((51, 51), dtype=np.uint8)
    skeleton[25, 5:46] = 255
    skeleton[5:46, 25] = 255
    accepted, reason = strict_jj_decision(
        skeleton, (5, 25), (25, 5), detected_junctions=[(5, 25), (25, 5), (25, 25)],
    )
    assert (accepted, reason) == (True, "continuous_skeleton_path")
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/vision/test_wiring_graph.py -q`

Expected: import fails for `strict_jj_decision`.

- [ ] **Step 3: Implement strict JJ and crossing decisions**

Add `use_strict_jj=True` and `use_crossing_semantics=True`. `strict_jj_decision` traces the skeleton between the two detected endpoints. At a degree-3-or-higher skeleton pixel, a direction change greater than 45 degrees is allowed only when a detected junction lies within the crossing tolerance. Aligned traversal may continue straight through an undetected crossing without merging the perpendicular wire. The pipeline applies nearest-neighbor filtering only after strict decisions accept candidates.

- [ ] **Step 4: Run GREEN and relevant regressions**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit strict JJ logic**

```powershell
git add -- src/vision/wiring_graph.py src/vision/unified_pipeline.py tests/vision/test_wiring_graph.py
git commit -m "fix: require continuous junction graph paths"
```

### Task 6: Render colored predicted networks and persist trace sidecars

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Modify: `src/vision/unified_pipeline.py:54-80, 2327-2504`
- Modify: `src/vision/joint_benchmark.py:464-500, 650-770`
- Test: `tests/vision/test_wiring_graph.py`
- Test: `tests/vision/test_joint_benchmark.py`

- [ ] **Step 1: Write failing network-render and sidecar tests**

```python
def test_build_network_render_data_assigns_same_color_within_group():
    from src.vision.wiring_graph import build_network_render_data
    data = build_network_render_data(
        raw_groups=[[(0, 0), (1, 0)], [(0, 1), (2, 0)]],
        components=[
            {"designator": "R1", "ports": [(0, 0), (10, 0)]},
            {"designator": "C1", "ports": [(20, 0)]},
            {"designator": "GND", "ports": [(30, 0)]},
        ],
    )
    assert data["port_colors"][(0, 0)] == data["port_colors"][(1, 0)]
    assert data["port_colors"][(0, 0)] != data["port_colors"][(0, 1)]


def test_joint_runner_writes_one_trace_sidecar_per_prediction(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    output = tmp_path / "output"

    def fake_process(_image_path, config):
        return {
            "components": [],
            "text_values": [],
            "junctions": [],
            "routes": [],
            "conn_pairs": [],
            "raw_groups": [],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [],
            "raw_text_detections": [],
            "wiring_trace": {
                "events": [],
                "summary": {"p2j": {"candidates": 1, "accepted": 0, "rejected": 1}},
            },
        }

    run_joint_benchmark(
        benchmark_dir=benchmark,
        cghd_root=cghd_root,
        output_dir=output,
        ocr_model_path="model.pt",
        process_fn=fake_process,
        render=False,
    )

    trace_path = output / "wiring_traces" / "sample.json"
    assert trace_path.is_file()
    assert json.loads(trace_path.read_text(encoding="utf-8"))["summary"]["p2j"]["rejected"] == 1
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py -q
```

Expected: missing render helper and sidecar assertion failure.

- [ ] **Step 3: Implement graph render data and sidecars**

`build_network_render_data()` maps each `(component_index, port_index)` to a deterministic BGR color based on the sorted `designator.port_label` identities in its final group. Extend the annotated renderer to draw accepted P2J/JJ segments before boxes and use colored port circles. Create `wiring_traces/` beside `predictions/` and write one `<stem>.json` from `result["wiring_trace"]`. Add per-stage totals to `summary.json`.

- [ ] **Step 4: Run GREEN and visually inspect C170**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py -q
python run_joint_benchmark.py --benchmark benchmark --cghd-root E:\circuit_image\cghd-zenodo-16 --ocr-model runs\ocr_crnn_hand_v2\best.pt --output results\wiring_trace_smoke_20260810 --expected-count 50 --prediction-cache results\joint_benchmark_seed3407_20260810
```

Expected: 50 trace sidecars, 50 annotated images, zero failures, and C170 shows different networks in distinct colors. Because the cached predictions predate the trace contract, use a one-image live pipeline smoke before treating trace counts as complete.

- [ ] **Step 5: Commit observability integration**

```powershell
git add -- src/vision/wiring_graph.py src/vision/unified_pipeline.py src/vision/joint_benchmark.py tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py
git commit -m "feat: visualize and persist wiring graph decisions"
```

### Task 7: Add cumulative ablations and run the 50-image benchmark

**Files:**
- Modify: `run_experiments.py:29-46, 344-505`
- Modify: `tests/test_wiring_experiments.py`
- Create: `docs/wiring_reliability_experiment_2026-08-10.md`

- [ ] **Step 1: Write failing cumulative-config tests**

```python
def test_wiring_reliability_configs_are_cumulative_and_llm_free():
    from run_experiments import build_wiring_reliability_configs
    configs = build_wiring_reliability_configs()
    assert list(configs) == [
        "frozen_baseline", "observability", "terminal", "strict_fallback",
        "outward_trace", "strict_jj", "crossing_semantics",
    ]
    assert all(config["skip_llm"] is True for config in configs.values())
    assert configs["frozen_baseline"]["use_terminal_components"] is False
    assert configs["terminal"]["use_terminal_components"] is True
    assert configs["strict_fallback"]["use_strict_p2j"] is True
    assert configs["outward_trace"]["use_outward_skeleton_trace"] is True
    assert configs["strict_jj"]["use_strict_jj"] is True
    assert configs["crossing_semantics"]["use_crossing_semantics"] is True
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_wiring_experiments.py -q`

Expected: import fails for `build_wiring_reliability_configs`.

- [ ] **Step 3: Implement cumulative configs and stage-summary columns**

Add `build_wiring_reliability_configs()` without changing the existing general ablation builder. Add CLI option `--suite wiring-reliability` and output columns for edge TP/FP/FN/F1 plus JSON-encoded stage summaries. Metadata records the 50-case assertion, git revision, OCR hash, detector hash, config dictionary, and `final_42_image_test_used=false`.

- [ ] **Step 4: Run GREEN and a three-image diagnostic suite**

```powershell
python -m pytest tests/test_wiring_experiments.py tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py -q
python run_experiments.py --suite wiring-reliability --images C170_D1_P1,C171_D1_P1,C274_D2_P1 --output-dir results\wiring_reliability_smoke_20260810
```

Expected: seven configs × three images complete, LLM is skipped, and every row contains edge metrics and stage counts.

- [ ] **Step 5: Run the full 50-image cumulative experiment**

```powershell
python run_experiments.py --suite wiring-reliability --expected-count 50 --output-dir results\wiring_reliability_full_20260810
```

Expected: seven configs × 50 images, zero failures, no final-42 input, and isolated output artifacts.

- [ ] **Step 6: Select the retained default from full-50 evidence**

For each cumulative row, compare micro edge F1 against the immediately preceding row. Keep a behavior enabled by default only when full-50 F1 increases. Record precision/recall changes and rejection reasons in `docs/wiring_reliability_experiment_2026-08-10.md`; never select based only on the three-image smoke test.

- [ ] **Step 7: Run final verification**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_joint_benchmark.py tests/test_wiring_experiments.py tests/vision/ocr_v2 -q
python -m py_compile src/vision/wiring_graph.py src/vision/unified_pipeline.py src/vision/joint_benchmark.py run_experiments.py
git diff --check
```

Expected: all tests pass, compilation exits 0, and no whitespace errors.

- [ ] **Step 8: Commit experiment support and report**

```powershell
git add -- run_experiments.py tests/test_wiring_experiments.py docs/wiring_reliability_experiment_2026-08-10.md
git commit -m "test: add wiring reliability ablation suite"
```

Do not stage model weights, result image directories, API keys, benchmark source images, or unrelated user files.
