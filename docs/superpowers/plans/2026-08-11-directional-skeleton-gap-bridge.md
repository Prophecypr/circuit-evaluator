# Directional Skeleton Gap Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover port-to-junction paths interrupted by one short skeleton gap while preventing sideways jumps, backward jumps, repeated gap hopping, and global distance relaxation.

**Architecture:** Extend the pure `trace_port_to_anchor()` BFS with one optional, direction-constrained transition that is considered only at a forward dead end. Keep the feature behind an independent configuration flag, expose bridge evidence in `wiring_trace`, and compare only `strict_jj` versus `directional_gap_bridge` on the worst 10 images before scheduling the full 50-image run.

**Tech Stack:** Python 3, NumPy/OpenCV skeleton arrays, pytest, existing wiring-reliability experiment runner.

---

### Task 1: Add focused experiment configuration selection

**Files:**
- Modify: `run_experiments.py`
- Test: `tests/test_wiring_experiments.py`

- [ ] **Step 1: Write failing tests**

Add tests proving that `run_wiring_reliability_experiment(..., selected_configs=["strict_jj", "directional_gap_bridge"])` executes exactly those two configurations and rejects unknown configuration names before creating results.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_wiring_experiments.py -k "selected_configs or unknown_config" -v
```

Expected: failure because `selected_configs` and the new configuration do not exist.

- [ ] **Step 3: Implement minimal selection support**

Add an optional `selected_configs` parameter, validate it against `build_wiring_reliability_configs()`, preserve declared order, record only selected configurations in metadata, and add CLI `--configs` as a comma-separated list.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused tests and then all of `tests/test_wiring_experiments.py`.

- [ ] **Step 5: Commit**

```powershell
git add run_experiments.py tests/test_wiring_experiments.py
git commit -m "test: support focused wiring comparisons"
```

### Task 2: Define directional gap-bridge behavior with RED tests

**Files:**
- Modify: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Add a collinear-gap acceptance test**

Create a horizontal skeleton split by a three-pixel blank gap. Assert that `trace_port_to_anchor(..., gap_bridge=4)` reaches the forward anchor, reports `directional_gap_bridge`, and reports exactly one bridge.

- [ ] **Step 2: Add rejection tests**

Cover: a gap longer than the configured limit, a side/parallel segment whose local tangent is inconsistent, and a trace that would require two separate gap bridges. All must return `None`.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/vision/test_wiring_graph.py -k "gap_bridge" -v
```

Expected: failure because `trace_port_to_anchor()` does not accept or implement gap bridging.

### Task 3: Implement one direction-constrained local bridge

**Files:**
- Modify: `src/vision/wiring_graph.py`
- Test: `tests/vision/test_wiring_graph.py`

- [ ] **Step 1: Extend the trace contract**

Add optional parameters `gap_bridge=0`, `min_bridge_cosine=0.85`, and `max_gap_bridges=1`. Existing callers remain continuous-only when `gap_bridge` is zero.

- [ ] **Step 2: Add dead-end candidate selection**

At a BFS dead end, rank disconnected skeleton pixels within `gap_bridge`. A candidate is eligible only when it advances along the port-outward projection, aligns with the recent path tangent by at least `min_bridge_cosine`, has no branch degree above two, and has a forward skeleton neighbor aligned with the bridge direction. Do not modify the skeleton array.

- [ ] **Step 3: Limit state to one bridge**

Track bridge count per queued path and never bridge again after `max_gap_bridges` is reached. Return `reason="directional_gap_bridge"`, `gap_bridges`, and `max_gap` when the accepted path used a bridge; preserve `continuous_skeleton_path` for untouched paths.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused gap tests followed by the complete wiring graph test file.

- [ ] **Step 5: Commit**

```powershell
git add src/vision/wiring_graph.py tests/vision/test_wiring_graph.py
git commit -m "feat: bridge one directional skeleton gap"
```

### Task 4: Wire the feature into the pipeline and trace metadata

**Files:**
- Modify: `src/vision/unified_pipeline.py`
- Modify: `run_experiments.py`
- Test: `tests/vision/test_wiring_graph.py`
- Test: `tests/test_wiring_experiments.py`

- [ ] **Step 1: Write failing integration assertions**

Assert that `DEFAULT_CONFIG["use_directional_gap_bridge"]` is false until publication validation, and that the cumulative experiment suite contains `directional_gap_bridge` immediately after `strict_jj` with only that flag changed.

- [ ] **Step 2: Run tests and verify RED**

Run the two affected test files; expect missing-key/config failures.

- [ ] **Step 3: Implement minimal integration**

Pass `gap_bridge=skel_gap` only when the flag is enabled. Record returned bridge count and maximum gap in the accepted `p2j_trace` event. Add the cumulative experiment configuration; make `crossing_semantics` inherit from it so later ablations remain cumulative.

- [ ] **Step 4: Run regression suite**

```powershell
python -m pytest tests/vision/test_wiring_graph.py tests/vision/test_wiring_error_analysis.py tests/vision/test_wiring_error_cli.py tests/test_wiring_experiments.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add src/vision/unified_pipeline.py run_experiments.py tests/vision/test_wiring_graph.py tests/test_wiring_experiments.py
git commit -m "feat: ablate directional gap bridging"
```

### Task 5: Run the worst-10 acceptance gate

**Files:**
- Create locally only: `results/wiring_gap_bridge_worst10_20260811/`

- [ ] **Step 1: Run exactly two configurations**

```powershell
python run_experiments.py --suite wiring-reliability --configs strict_jj,directional_gap_bridge --images C11_D1_P1,C11_D1_P3,C158_D2_P1,C20_D1_P1,C274_D2_P1,C19_D1_P2,C79_D1_P2,C103_D1_P1,C195_D1_P3,C171_D1_P1 --expected-count 10 --output-dir results/wiring_gap_bridge_worst10_20260811
```

- [ ] **Step 2: Apply the gate**

Baseline reference is TP=20, FP=257, FN=248, F1=0.0733945. Proceed only if the new configuration has F1 above the concurrently rerun `strict_jj` row and FP does not exceed that row. If it fails, retain the diagnostic output but do not enable the feature or run all 50 images.

### Task 6: Run full-50 comparison only after the worst-10 gate passes

**Files:**
- Create locally only: `results/wiring_gap_bridge_full50_20260811/`
- Modify: `docs/wiring_reliability_experiment_2026-08-10.md`

- [ ] **Step 1: Run the full comparison**

```powershell
python run_experiments.py --suite wiring-reliability --configs strict_jj,directional_gap_bridge --expected-count 50 --output-dir results/wiring_gap_bridge_full50_20260811
```

- [ ] **Step 2: Decide retention**

Retain and enable the feature for the publication configuration only if full-50 micro edge F1 exceeds the concurrently rerun `strict_jj` result. Report TP, FP, FN, precision, recall, F1, failed-image count, and bridge-stage counts even if rejected.

- [ ] **Step 3: Protect the final test**

Confirm `final_42_image_test_used` remains false and do not inspect any final-42 image.

- [ ] **Step 4: Update conclusions and commit**

Document whether the bridge was retained or rejected, link the local result directories, and commit only code, tests, and documentation—not `results/`.
