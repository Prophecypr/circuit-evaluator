# CRNN-v2 Kaggle Training Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Kaggle GPU package that trains and independently evaluates a handwritten circuit-value CRNN without using the frozen 42-image final test set or any LLM.

**Architecture:** A versioned `ocr_v2` package owns normalization, provenance-aware dataset construction, preprocessing, augmentation, CRNN definition, metrics, training, evaluation, and safe export. A thin builder creates manifests and a Kaggle-ready ZIP from CGHD plus historical Digitize-HCD crops. Kaggle runs one command from a Run All notebook and exports a self-describing checkpoint. Pipeline integration is a later gated step after the returned checkpoint passes the held-out-writer validation thresholds.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV, NumPy, standard-library CSV/JSON/XML/hash/ZIP modules, pytest, Kaggle GPU.

---

## Fixed experiment contract

- Final test set: `E:\ClaudeCode\电路图智能评价系统\测试集` and `E:\ClaudeCode\电路图智能评价系统\测试集实验_v1`; these paths, images, spreadsheets, hashes, and derivatives are forbidden in all training-package outputs.
- Training sources: CGHD v16 at `E:\circuit_image\cghd-zenodo-16` plus value-like crops from `data/ocr_training`.
- Split unit: CGHD `drafter_id`; validation drafters never appear in training. Digitize-HCD is training-only.
- Canonical characters: `0123456789.kmMΩμunpFV AHz-+/`; aliases are normalized before charset validation.
- Image contract: grayscale, aspect-preserving resize, fixed `32 x 160`, black padding before inversion, normalized to `[-1, 1]`.
- Selection metric: normalized exact match; tie-breaker normalized CER.
- Training limit: 40 epochs, AdamW, seed 42, early-stop patience 7.
- Promotion gate: normalized exact match at least 80%, normalized CER at most 10%, and better than the active machine checkpoint on the identical held-out-writer validation manifest.
- Output checkpoint: `runs/ocr_crnn_hand_v2/best.pt`; never overwrite existing weights.

## Task 1: Establish the normalization and value-label contract

**Files:**
- Create: `src/vision/ocr_v2/__init__.py`
- Create: `src/vision/ocr_v2/config.py`
- Create: `src/vision/ocr_v2/normalize.py`
- Create: `configs/ocr_crnn_hand_v2.json`
- Test: `tests/vision/ocr_v2/test_normalize.py`

- [ ] **Step 1: Write failing normalization tests**

```python
from src.vision.ocr_v2.normalize import is_value_label, normalize_label


def test_normalize_unit_aliases_without_collapsing_mega_and_milli():
    assert normalize_label(" 10KΩ ") == "10kΩ"
    assert normalize_label("4.7uF") == "4.7μF"
    assert normalize_label("4.7µF") == "4.7μF"
    assert normalize_label("1MΩ") == "1MΩ"
    assert normalize_label("1mA") == "1mA"


def test_value_filter_rejects_ic_identifiers():
    assert is_value_label("10kΩ")
    assert is_value_label("3V3")
    assert not is_value_label("LM358")
    assert not is_value_label("NE555")
```

- [ ] **Step 2: Run the focused test and confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_normalize.py -q`

Expected: import failure because `src.vision.ocr_v2.normalize` does not exist.

- [ ] **Step 3: Implement the canonical charset, normalization, and conservative value grammar**

```python
NORMALIZATION_VERSION = "circuit-value-v1"
CANONICAL_CHARS = "0123456789.kmMΩμunpFV AHz-+/"


def normalize_label(text: str) -> str:
    value = text.strip().replace("µ", "μ").replace("u", "μ")
    value = value.replace("Ω", "Ω").replace("ω", "Ω")
    return re.sub(r"(?<=\d)K(?=Ω|$)", "k", value)
```

Implement `is_value_label()` with a full-match grammar that requires a numeric form and permits circuit units or the `3V3` notation. Keep raw and normalized labels separate.

- [ ] **Step 4: Add JSON config and validate it on import**

The config records data roots, fixed validation drafter IDs, charset, image size, seed, training hyperparameters, augmentation probabilities, output path, and promotion thresholds. `load_config(path)` must reject a charset mismatch with `CANONICAL_CHARS`.

- [ ] **Step 5: Re-run the focused test**

Run: `python -m pytest tests/vision/ocr_v2/test_normalize.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit only Task 1 files**

```bash
git add src/vision/ocr_v2/__init__.py src/vision/ocr_v2/config.py src/vision/ocr_v2/normalize.py configs/ocr_crnn_hand_v2.json tests/vision/ocr_v2/test_normalize.py
git commit -m "feat: define OCR v2 label contract"
```

## Task 2: Build provenance-aware manifests and writer-isolated splits

**Files:**
- Create: `src/vision/ocr_v2/dataset_builder.py`
- Test: `tests/vision/ocr_v2/test_dataset_builder.py`

- [ ] **Step 1: Write failing tests for provenance, split isolation, and hash rejection**

```python
from src.vision.ocr_v2.dataset_builder import assert_no_leakage, split_cghd_rows


def test_writer_split_has_no_drafter_overlap(sample_rows):
    train, val = split_cghd_rows(sample_rows, validation_drafters={"drafter_2"})
    assert {r["drafter_id"] for r in train}.isdisjoint(
        {r["drafter_id"] for r in val}
    )


def test_hash_guard_rejects_final_test_image(sample_rows):
    sample_rows[0]["source_sha256"] = "blocked"
    with pytest.raises(ValueError, match="final-test hash"):
        assert_no_leakage(sample_rows, forbidden_hashes={"blocked"})
```

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_dataset_builder.py -q`

Expected: import failure for `dataset_builder`.

- [ ] **Step 3: Implement XML extraction with stable provenance**

Each manifest row must contain:

```python
MANIFEST_FIELDS = (
    "sample_id", "drafter_id", "source_kind", "source_image",
    "source_xml", "bbox", "raw_label", "normalized_label",
    "source_sha256", "crop_sha256", "crop_path", "split",
)
```

Use `xml.etree.ElementTree`, clamp boxes to image bounds, write grayscale PNG crops, hash source bytes and encoded crop bytes, and create deterministic sample IDs from provenance rather than a global counter.

- [ ] **Step 4: Add writer split and final-test guards**

`split_cghd_rows()` assigns configured drafter IDs to validation and all other valid drafters to training. `assert_no_leakage()` rejects shared drafter IDs, source hashes, crop hashes, missing files, forbidden path fragments, duplicate sample IDs, and unsupported normalized characters. Historical Digitize-HCD rows are always marked `source_kind=digitize_hcd` and `split=train`.

- [ ] **Step 5: Re-run focused tests**

Run: `python -m pytest tests/vision/ocr_v2/test_dataset_builder.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2 files**

```bash
git add src/vision/ocr_v2/dataset_builder.py tests/vision/ocr_v2/test_dataset_builder.py
git commit -m "feat: build leakage-safe OCR manifests"
```

## Task 3: Share preprocessing and photo-domain augmentation

**Files:**
- Create: `src/vision/ocr_v2/image_ops.py`
- Create: `src/vision/ocr_v2/dataset.py`
- Test: `tests/vision/ocr_v2/test_image_ops.py`

- [ ] **Step 1: Write failing image-contract tests**

```python
def test_preprocess_has_fixed_shape_range_and_dtype():
    out = preprocess_gray(np.full((20, 50), 255, np.uint8), 32, 160)
    assert out.shape == (1, 32, 160)
    assert out.dtype == np.float32
    assert -1.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_validation_preprocess_is_deterministic():
    assert np.array_equal(preprocess_gray(IMAGE, 32, 160), preprocess_gray(IMAGE, 32, 160))
```

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_image_ops.py -q`

Expected: import failure for `image_ops`.

- [ ] **Step 3: Implement shared preprocessing**

`preprocess_gray(image, img_h, img_w)` must be the only resize/pad/invert/normalize implementation used by dataset loading and inference. Reject empty crops instead of replacing them with silent black images.

- [ ] **Step 4: Implement seeded training-only augmentation**

`augment_phone_photo(image, rng, config)` applies bounded affine/perspective geometry, crop jitter, blur, JPEG, sensor noise, brightness/contrast, local shadow, and morphology. It must return `uint8`, preserve at least one foreground component, and never run for validation.

- [ ] **Step 5: Implement `CircuitValueDataset` and CTC collate**

Dataset input is a manifest CSV, not an unversioned tab-delimited file. It returns the processed tensor, encoded label, label length, sample ID, raw label, and normalized label. Unsupported characters raise an exception with the sample ID.

- [ ] **Step 6: Re-run focused tests and commit**

Run: `python -m pytest tests/vision/ocr_v2/test_image_ops.py -q`

```bash
git add src/vision/ocr_v2/image_ops.py src/vision/ocr_v2/dataset.py tests/vision/ocr_v2/test_image_ops.py
git commit -m "feat: add OCR v2 image pipeline"
```

## Task 4: Version the CRNN and checkpoint contract

**Files:**
- Create: `src/vision/ocr_v2/model.py`
- Create: `src/vision/ocr_v2/checkpoint.py`
- Test: `tests/vision/ocr_v2/test_checkpoint.py`

- [ ] **Step 1: Write failing round-trip and incompatibility tests**

```python
def test_checkpoint_round_trip_preserves_contract(tmp_path):
    model = CRNNv2(num_classes=len(CANONICAL_CHARS) + 1)
    save_checkpoint(tmp_path / "best.pt", model, metadata=VALID_METADATA)
    loaded, meta = load_checkpoint(tmp_path / "best.pt", device="cpu")
    assert meta["model_version"] == "crnn-v2-handwritten-1"
    assert meta["chars"] == CANONICAL_CHARS


def test_checkpoint_rejects_wrong_image_width(tmp_path):
    with pytest.raises(CheckpointContractError, match="img_w"):
        validate_metadata({**VALID_METADATA, "img_w": 128})
```

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_checkpoint.py -q`

- [ ] **Step 3: Implement `CRNNv2` and greedy CTC decoding**

Use one CNN + two-layer bidirectional LSTM + linear CTC head implementation. `forward()` returns `[batch, time, classes]`. `greedy_decode()` removes blanks and consecutive duplicate indices.

- [ ] **Step 4: Implement atomic, self-describing checkpoints**

Required metadata: model version, architecture, chars, `img_h`, `img_w`, normalization version, split ID, seed, epoch, validation metrics, Git commit, PyTorch version, and training platform. Write to a temporary sibling path and replace only after serialization succeeds; reject any pre-existing destination unless `allow_replace=True` is explicitly passed inside the run output directory.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/vision/ocr_v2/test_checkpoint.py -q`

```bash
git add src/vision/ocr_v2/model.py src/vision/ocr_v2/checkpoint.py tests/vision/ocr_v2/test_checkpoint.py
git commit -m "feat: version OCR v2 checkpoints"
```

## Task 5: Implement publication-grade OCR metrics

**Files:**
- Create: `src/vision/ocr_v2/metrics.py`
- Test: `tests/vision/ocr_v2/test_metrics.py`

- [ ] **Step 1: Write failing CER, exact, numeric, and unit tests**

```python
def test_cer_can_exceed_one():
    assert cer("1", "111") == 2.0


def test_metric_report_separates_numeric_and_unit_accuracy():
    report = evaluate_pairs([("10kΩ", "10MΩ"), ("4.7μF", "4.7μF")])
    assert report["numeric_exact"] == 1.0
    assert report["unit_exact"] == 0.5
    assert report["normalized_exact"] == 0.5
```

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_metrics.py -q`

- [ ] **Step 3: Implement edit distance and structured value parsing**

Return raw exact, normalized exact, raw CER, normalized CER, numeric exact, unit exact, sample count, and per-unit `{count, exact}` for `Ω`, `kΩ`, `MΩ`, `μF`, `nF`, `V`, `H`, `A`, and `Hz`. Emit an error-case row for every non-exact prediction.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/vision/ocr_v2/test_metrics.py -q`

```bash
git add src/vision/ocr_v2/metrics.py tests/vision/ocr_v2/test_metrics.py
git commit -m "feat: add OCR publication metrics"
```

## Task 6: Add reproducible training and independent evaluation CLIs

**Files:**
- Create: `src/vision/ocr_v2/train.py`
- Create: `src/vision/ocr_v2/evaluate.py`
- Test: `tests/vision/ocr_v2/test_train_smoke.py`

- [ ] **Step 1: Write a one-batch CPU smoke test**

The fixture creates four tiny labeled images and train/validation manifests. The test calls `train_from_config(..., max_epochs=1, max_batches=1, device="cpu")`, then asserts that `last.pt`, `best.pt`, `history.csv`, `metrics.json`, `errors.csv`, `run_config.json`, and `environment.txt` exist and that the independent evaluator loads `best.pt`.

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_train_smoke.py -q`

- [ ] **Step 3: Implement deterministic training**

```python
def train_from_config(
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    max_epochs: int | None = None,
    max_batches: int | None = None,
) -> dict[str, object]:
    ...
```

Set Python, NumPy, and PyTorch seeds. Use AdamW, CTC loss, gradient clipping, a learning-rate scheduler, normalized exact selection, normalized CER tie-break, and patience 7. Do not inspect any final-test GT. Log every epoch to CSV and save both best and last checkpoints.

- [ ] **Step 4: Implement independent evaluation**

CLI:

```bash
python -m src.vision.ocr_v2.evaluate --checkpoint runs/ocr_crnn_hand_v2/best.pt --manifest data/ocr_v2/val_manifest.csv --output runs/ocr_crnn_hand_v2/independent_eval
```

The evaluator loads preprocessing, charset, and image size from the checkpoint contract, writes `metrics.json`, `predictions.csv`, and `errors.csv`, and exits nonzero on a contract mismatch.

- [ ] **Step 5: Run smoke test and commit**

Run: `python -m pytest tests/vision/ocr_v2/test_train_smoke.py -q`

```bash
git add src/vision/ocr_v2/train.py src/vision/ocr_v2/evaluate.py tests/vision/ocr_v2/test_train_smoke.py
git commit -m "feat: train and evaluate OCR v2"
```

## Task 7: Build a guarded Kaggle data and code package

**Files:**
- Create: `src/vision/ocr_v2/package.py`
- Create: `scripts/build_ocr_kaggle_package.py`
- Create: `notebooks/crnn_v2_kaggle.ipynb`
- Create: `docs/ocr_crnn_v2_kaggle_guide.md`
- Test: `tests/vision/ocr_v2/test_package_guard.py`

- [ ] **Step 1: Write failing package-safety tests**

```python
def test_package_rejects_secret_names(tmp_path):
    (tmp_path / "kaggle.json").write_text("secret")
    with pytest.raises(PackageSafetyError, match="secret"):
        inspect_package_tree(tmp_path, forbidden_hashes=set())


def test_package_rejects_final_test_hash(tmp_path):
    image = tmp_path / "copied.jpg"
    image.write_bytes(b"final-test-image")
    with pytest.raises(PackageSafetyError, match="final-test hash"):
        inspect_package_tree(tmp_path, forbidden_hashes={sha256_file(image)})
```

- [ ] **Step 2: Confirm red**

Run: `python -m pytest tests/vision/ocr_v2/test_package_guard.py -q`

- [ ] **Step 3: Implement builder and package inventory**

CLI:

```bash
python scripts/build_ocr_kaggle_package.py --config configs/ocr_crnn_hand_v2.json --cghd-root E:\circuit_image\cghd-zenodo-16 --digitize-root data/ocr_training --final-test-root E:\ClaudeCode\电路图智能评价系统\测试集 --final-gt-root E:\ClaudeCode\电路图智能评价系统\测试集实验_v1 --output outputs/ocr_crnn_hand_v2_kaggle
```

The builder extracts normalized crops, writes `train_manifest.csv`, `val_manifest.csv`, `dataset_report.json`, `split.json`, `file_inventory.csv`, `SHA256SUMS`, and a ZIP. It copies only allowlisted `src/vision/ocr_v2` modules, config, notebook, and guide. It scans member names and bytes for API-key patterns and rejects any member whose SHA-256 matches a file under either final-test root.

- [ ] **Step 4: Create a Run All notebook with no embedded secrets or data paths**

Notebook cells must: locate the attached package under `/kaggle/input`, install nothing unless an import check fails, display environment/GPU, validate manifests and hashes, run the trainer, run the independent evaluator, show metrics/error tables, and create `/kaggle/working/ocr_crnn_hand_v2_results.zip`.

- [ ] **Step 5: Write the Chinese upload/run/download guide**

The guide gives exact Kaggle UI actions and commands, expected output files, failure diagnostics, and the instruction to return `ocr_crnn_hand_v2_results.zip` without opening or editing the 42-image GT during model selection.

- [ ] **Step 6: Run package tests and commit**

Run: `python -m pytest tests/vision/ocr_v2/test_package_guard.py -q`

```bash
git add src/vision/ocr_v2/package.py scripts/build_ocr_kaggle_package.py notebooks/crnn_v2_kaggle.ipynb docs/ocr_crnn_v2_kaggle_guide.md tests/vision/ocr_v2/test_package_guard.py
git commit -m "feat: package OCR v2 for Kaggle"
```

## Task 8: Generate and verify the deliverable locally

**Files:**
- Generate: `outputs/ocr_crnn_hand_v2_kaggle/ocr_crnn_hand_v2_kaggle.zip`
- Generate: `outputs/ocr_crnn_hand_v2_kaggle/SHA256SUMS`
- Modify if needed: `.gitignore`

- [ ] **Step 1: Build the real package from the configured sources**

Run the Task 7 builder command from the repository root.

Expected: nonzero training and validation sample counts; zero drafter/hash/path overlap; one ZIP and one checksum inventory.

- [ ] **Step 2: Run a CPU one-batch smoke train against the generated manifests**

```bash
python -m src.vision.ocr_v2.train --config outputs/ocr_crnn_hand_v2_kaggle/package/configs/ocr_crnn_hand_v2.json --data-root outputs/ocr_crnn_hand_v2_kaggle/package/data --output outputs/ocr_crnn_hand_v2_kaggle/smoke --device cpu --max-epochs 1 --max-batches 1
```

Expected: all required artifacts exist and the checkpoint reloads.

- [ ] **Step 3: Verify the ZIP independently**

```bash
python -m src.vision.ocr_v2.package verify --zip outputs/ocr_crnn_hand_v2_kaggle/ocr_crnn_hand_v2_kaggle.zip --final-test-root E:\ClaudeCode\电路图智能评价系统\测试集 --final-gt-root E:\ClaudeCode\电路图智能评价系统\测试集实验_v1
python -m pytest tests/vision/ocr_v2 -q
```

Expected: package verification exits 0; focused test suite reports zero failures.

- [ ] **Step 4: Add generated outputs to `.gitignore`, not Git**

Do not commit training crops, ZIPs, checkpoints, logs, API keys, or final-test files.

## Task 9: Evaluate the checkpoint returned from Kaggle

**Files:**
- User-provided: downloaded `ocr_crnn_hand_v2_results.zip`
- Generate: `runs/ocr_crnn_hand_v2/independent_eval/`

- [ ] **Step 1: Verify returned hashes and checkpoint metadata**

Reject the result if the checkpoint charset, split ID, normalization version, image size, or Git commit differs from the training package contract.

- [ ] **Step 2: Run the independent evaluator on the frozen held-out-writer manifest**

Run the evaluation CLI from Task 6 and compare the new checkpoint with `runs/ocr_crnn_machine/best.pt` on the exact same manifest.

- [ ] **Step 3: Apply the promotion gate**

Promote only if normalized exact is at least 80%, normalized CER is at most 10%, and the new model beats the active model. Otherwise retain the existing pipeline model and use the error CSV to decide whether to rebalance units or compare SVTR/PARSeq; do not use the 42-image GT for this decision.

## Task 10: Back up, integrate, freeze, then run the final test once

**Files:**
- Modify: `src/vision/unified_pipeline.py`
- Create: `tests/vision/ocr_v2/test_pipeline_integration.py`
- Backup: timestamped copies under `backups/ocr_switch_<timestamp>/`

- [ ] **Step 1: Add a failing integration test for configurable OCR paths and rollback**

The test proves the pipeline loads a configured v2 checkpoint, validates its contract, and can explicitly select the legacy machine checkpoint.

- [ ] **Step 2: Back up old/new weights and configuration before editing the default**

Record SHA-256 hashes and copy, never move or overwrite, both weight files and the pipeline/config files.

- [ ] **Step 3: Implement the configurable loader and pass the integration test**

Do not change the default until the Task 9 gate passes. Preserve a one-command legacy rollback.

- [ ] **Step 4: Freeze the experiment**

Write a freeze manifest containing checkpoint hash, config hash, code commit, preprocessing version, normalization version, and all validation metrics. After this point, no OCR changes are allowed before final testing.

- [ ] **Step 5: Complete the 42-image GT independently**

The three annotators finish values and wiring GT from image content only. Resolve disagreements without showing predictions. Lock the workbook and record its hash.

- [ ] **Step 6: Run the 42-image final test exactly once**

Report component/port/value metrics, raw and normalized OCR metrics, circuit-level end-to-end accuracy, per-writer results, confidence intervals, and failure categories. These results are for the paper, not for another tuning cycle.

## Final verification checklist

- [ ] `python -m pytest tests/vision/ocr_v2 -q` reports zero failures.
- [ ] Real package build reports train/validation counts and zero leakage.
- [ ] CPU smoke training exports and reloads a checkpoint.
- [ ] ZIP verification reports no secret names/patterns, no final-test hashes, and no forbidden paths.
- [ ] `git status --short` shows only intentional source/docs/test changes; unrelated reference-pack files remain untouched.
- [ ] Placeholder scan is empty: `Select-String -Path src/vision/ocr_v2/*.py,scripts/build_ocr_kaggle_package.py,docs/ocr_crnn_v2_kaggle_guide.md -Pattern 'TODO|TBD|PLACEHOLDER|pass\s*(#.*)?$'`.
- [ ] Dataset and checkpoint reports retain raw labels/predictions as well as normalized values.
- [ ] No completion claim is made until the exact verification commands above have fresh exit-code-zero evidence.
