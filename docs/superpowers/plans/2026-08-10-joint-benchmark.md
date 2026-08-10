# Joint Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, LLM-free 50-image benchmark report covering components, junctions, ports, OCR values, and wiring.

**Architecture:** Add pure metric and GT-loading helpers, minimally expose raw detector outputs from the existing pipeline, then add a runner that executes the pipeline and writes an isolated result directory. Select and hash the seed-3407 OCR checkpoint only after validating all three seed records.

**Tech Stack:** Python 3, PyTorch/Ultralytics, OpenCV, pytest, JSON/CSV, Pascal VOC XML.

---

### Task 1: Freeze three-seed evidence

**Files:**
- Create: `runs/ocr_crnn_hand_v2/three_seed_summary.json`
- Create: `runs/ocr_crnn_hand_v2/three_seed_summary.csv`

- [ ] Verify both returned configurations differ from seed 42 only in `data.seed`.
- [ ] Verify independent metrics equal the training-run best metrics.
- [ ] Calculate mean and sample standard deviation for the six OCR metrics.
- [ ] Record ZIP and checkpoint SHA-256 hashes.

### Task 2: Add raw detector outputs

**Files:**
- Modify: `src/vision/unified_pipeline.py`
- Test: `tests/vision/test_joint_benchmark.py`

- [ ] Write a failing test requiring raw junction and text detections in the returned result.
- [ ] Add the minimal return fields without changing existing detection or graph behavior.
- [ ] Run the focused test and existing pipeline integration tests.

### Task 3: Add publication metrics and GT loaders

**Files:**
- Create: `src/vision/joint_benchmark.py`
- Test: `tests/vision/test_joint_benchmark.py`

- [ ] Test and implement box IoU matching, precision/recall/F1, and AP50.
- [ ] Test and implement point matching for processed junctions and ports.
- [ ] Test and implement XML text/junction parsing and electrical-value normalization.
- [ ] Test and implement end-to-end OCR scoring with missed detections represented as empty predictions.

### Task 4: Add the isolated runner

**Files:**
- Create: `run_joint_benchmark.py`
- Test: `tests/vision/test_joint_benchmark.py`

- [ ] Test that every process call forces `skip_llm=True` and `save_artifacts=False`.
- [ ] Write predictions, metrics, errors, images, and provenance only below the requested output directory.
- [ ] Refuse a non-empty output directory and refuse a benchmark/XML mismatch.

### Task 5: Promote and execute

**Files:**
- Create: `runs/ocr_crnn_hand_v2/backups/pre_switch_seed3407_*/`
- Modify: `runs/ocr_crnn_hand_v2/best.pt`
- Create: `results/joint_benchmark_*/`

- [ ] Back up the seed-42 canonical checkpoint and copy the verified seed-3407 checkpoint into place.
- [ ] Confirm runtime loading and a fixed-crop prediction after the switch.
- [ ] Run all 50 images with the LLM disabled.
- [ ] Verify image counts, non-null denominators, hashes, and output completeness.

