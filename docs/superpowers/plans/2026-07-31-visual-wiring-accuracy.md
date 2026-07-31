# Visual Wiring Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the visual wiring benchmark reproducible and reduce false wire evidence by excluding component interiors before CCL and skeleton extraction, without calling an LLM.

**Architecture:** `run_experiments.py` will resolve every ablation into a full immutable config, write all outputs into a caller-selected run directory, and record the resolved settings. `unified_pipeline.py` will build one component-masked binary wire image that both CCL and skeleton extraction consume, preserving the existing port and graph logic.

**Tech Stack:** Python 3.13, pytest, OpenCV, NumPy, Ultralytics pipeline (not invoked by unit tests).

---

### Task 1: Reproducible, LLM-free ablation configurations

**Files:**
- Create: `tests/test_wiring_experiments.py`
- Modify: `run_experiments.py:1-35, 260-320`

- [ ] **Step 1: Write the failing tests**

```python
from run_experiments import build_ablation_configs, resolve_output_dir


def test_ablation_configs_are_full_unique_and_llm_free():
    configs = build_ablation_configs()
    assert configs["Ours"]["use_skeleton"] is True
    assert configs["w/o_Skeleton"]["use_skeleton"] is False
    assert configs["w/o_Sobel"]["use_sobel"] is False
    assert all(cfg["skip_llm"] is True for cfg in configs.values())
    fingerprints = {tuple(sorted(cfg.items())) for cfg in configs.values()}
    assert len(fingerprints) == len(configs)


def test_output_directory_is_explicit_and_created(tmp_path):
    output_dir = resolve_output_dir(tmp_path / "visual-only-run")
    assert output_dir.is_dir()
    assert output_dir.name == "visual-only-run"
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_wiring_experiments.py -v`

Expected: FAIL because `build_ablation_configs` and `resolve_output_dir` do not exist.

- [ ] **Step 3: Implement the minimal experiment API**

```python
def build_ablation_configs():
    full = {**DEFAULT_CONFIG, "skip_llm": True}
    configs = {
        "Ours": full,
        "Baseline": {**full, **{key: False for key in DEFAULT_CONFIG}, "skip_llm": True},
        "w/o_Skeleton": {**full, "use_skeleton": False},
        "w/o_Sobel": {**full, "use_sobel": False},
        "w/o_NN_Filter": {**full, "use_nn_filter": False},
        "w/o_Close_Port": {**full, "use_close_port": False},
        "CCL": {**full, "use_ccl": True},
    }
    return configs
```

Add `resolve_output_dir(output_dir)` using `Path.mkdir(parents=True, exist_ok=True)`. Change `main` to accept `output_dir`, save CSV/plots and a `run_metadata.json` with the resolved configs there, and add an argparse `--output-dir` option. Never overwrite the root-level historical experiment artifacts.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `python -m pytest tests/test_wiring_experiments.py -v`

Expected: PASS with two tests.

- [ ] **Step 5: Commit**

```bash
git add run_experiments.py tests/test_wiring_experiments.py
git commit -m "feat: make visual ablations reproducible"
```

### Task 2: Component-masked wire evidence shared by CCL and skeleton steps

**Files:**
- Create: `tests/test_wire_mask.py`
- Modify: `src/vision/unified_pipeline.py:40-50, 800-850, 1594-1620, 1705-1715`

- [ ] **Step 1: Write the failing tests**

```python
import cv2
import numpy as np

from src.vision.unified_pipeline import _build_wire_mask, _extract_skeleton_from_mask


def test_component_mask_removes_symbol_interior_but_keeps_wire_outside():
    gray = np.full((80, 100), 255, dtype=np.uint8)
    cv2.line(gray, (5, 40), (95, 40), 0, 2)
    cv2.rectangle(gray, (38, 25), (62, 55), 0, 2)
    components = [{"xyxy": [35, 22, 65, 58]}]

    mask = _build_wire_mask(gray, components, im_scale=1.0)

    assert mask[40, 15] == 255
    assert mask[40, 50] == 0


def test_skeleton_from_binary_mask_preserves_a_thin_wire():
    mask = np.zeros((40, 80), dtype=np.uint8)
    cv2.line(mask, (5, 20), (74, 20), 255, 3)

    skeleton = _extract_skeleton_from_mask(mask)

    assert skeleton[20, 10] == 255
    assert skeleton[20, 70] == 255
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_wire_mask.py -v`

Expected: FAIL because the component-mask helpers do not exist.

- [ ] **Step 3: Implement the minimal visual-only preprocessing**

```python
def _build_wire_mask(gray_img, components, im_scale=1.0):
    wire_mask = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6,
    )
    inset = max(3, int(5 * im_scale))
    for component in components:
        x1, y1, x2, y2 = component["xyxy"]
        wire_mask[y1 + inset:y2 - inset, x1 + inset:x2 - inset] = 0
    return wire_mask
```

Add `_extract_skeleton_from_mask` by moving the existing morphology, thinning, and resize logic behind a binary-mask entry point. Keep `_extract_skeleton(gray_img)` as a compatibility wrapper. Add `use_component_mask` to `DEFAULT_CONFIG`; use the shared mask in both CCL and normal skeleton execution when that flag is enabled. Do not change OCR, model weights, input images, benchmark labels, or LLM integration.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `python -m pytest tests/test_wire_mask.py -v`

Expected: PASS with two tests.

- [ ] **Step 5: Commit**

```bash
git add src/vision/unified_pipeline.py tests/test_wire_mask.py
git commit -m "feat: mask component interiors from wire evidence"
```

### Task 3: Visual-only verification without overwriting history

**Files:**
- Modify: none

- [ ] **Step 1: Run all non-LLM tests**

Run: `python -m pytest tests/test_wiring_experiments.py tests/test_wire_mask.py tests/test_phase2.py -v`

Expected: PASS; phase-1 LLM tests are intentionally excluded because no LLM endpoint is configured or used.

- [ ] **Step 2: Run the benchmark into a new directory**

Run: `python run_experiments.py --output-dir results/visual_wiring_20260731`

Expected: CSV, charts, and `run_metadata.json` are created below the new directory while root-level historical artifacts remain unchanged.

- [ ] **Step 3: Compare the recorded Ours metrics**

Read `results/visual_wiring_20260731/experiment_results.csv` and compare `Ours` against the backed-up historical 30.1% port-correct rate, 14.0% group accuracy, and 57.9% FP rate. Report a metric only if the benchmark completes successfully.

- [ ] **Step 4: Commit only source and tests**

```bash
git add run_experiments.py src/vision/unified_pipeline.py tests/
git commit -m "test: cover visual wiring accuracy pipeline"
```
