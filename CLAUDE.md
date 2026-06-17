# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Benchmark experiment (56-image wiring evaluation with ablations)
python run_experiments.py

# Annotation tool server (port 8765)
cd benchmark && python server.py

# Generate/regenerate detection JSONs for benchmark images
python gen_annotation_data.py

# Single image pipeline test (skip LLM for speed)
python -c "from src.vision.unified_pipeline import process_image; r = process_image('benchmark/img_000.jpg', config={'skip_llm': True})"

# CRNN OCR training (local CPU — slow, use Kaggle instead)
python -m src.vision.train_ocr --train

# Extract CGHD text crops for OCR training data
python -m src.vision.extract_cghd_text

# Tests
pytest tests/ -v
```

## Architecture

**Main pipeline** (`src/vision/unified_pipeline.py`): Image → CGHD YOLO 61-class detection → CRNN OCR on text regions → component port assignment (hardcoded + Sobel orientation) → port grid snapping → skeleton extraction (Zhang-Suen) → **skeleton port snapping (改进1)** → **P2J with skeleton-priority lock (改进2)** → JJ with NN filter → LOS → close-port → force-connect → Union-Find connection graph → LLM evaluation → annotated image + TXT.

**Config-based ablation**: `process_image(img_path, config=None)` accepts a dict toggling 8 wiring steps. Default config lives in `DEFAULT_CONFIG` dict. Every wiring step is gated by its config flag. `skip_llm=True` avoids API calls during experiments.

**Text pipelines** (`src/phase1_verify/`, `phase2_pipeline/`, `phase4_netlist/`): Text circuit description → LLM evaluation or SPICE simulation.

**LLM** (`src/llm.py`): OpenAI-compatible SDK. Priority: `OLLAMA_BASE_URL` env var → Anthropic key → DeepSeek key.

## Active Models

| Model | Path | Classes | Role |
|-------|------|:-------:|------|
| CGHD 61cls | `runs/detect/cghd_61cls/weights/best.pt` | 61 | Component + junction + text detection |
| CRNN OCR | `runs/ocr_crnn_machine/best.pt` | 28 chars | Text recognition (CGHD hand-drawn trained) |

## Benchmark Dataset

`benchmark/` — 41 CGHD hand-drawn circuit images with manual wiring ground truth.

| Directory/File | Purpose |
|------|---------|
| `benchmark/img_*.jpg` | Source images (img_000 to img_040) |
| `benchmark/manifest.txt` | Image list (one filename per line) |
| `benchmark/detections/img_*.json` | Pre-computed YOLO detections (component boxes + ports) |
| `benchmark/result/img_*_gt.txt` | Manual wiring ground truth (41 files, 476 groups total) |
| `benchmark/annotation_tool.html` | Web annotation interface |
| `benchmark/server.py` | HTTP server for annotation tool |
| `benchmark/missed_detections.txt` | Known YOLO detection errors (img_014/017/018) |

**GT format**: `G1: R1.1, R2.2, C1.+` — one group per line, `designator.port_label` pairs.

**Designator matching**: Pipeline and detection JSON generate designators independently (same algorithm, different YOLO run order → different numbering). Evaluation bridges this via **IoU-based component matching** (threshold 0.3), then maps ports by **index** (not position) since both use same PORT_POSITIONS.

## Experiment Pipeline (`run_experiments.py`)

4 configs run against all GT images:

| Config | Skeleton | Sobel | NN filter | Close/LOS/Force |
|--------|:--:|:--:|:--:|:--:|
| Ours | ✓ | ✓ | ✓ | ✓ |
| Baseline | ✗ | ✗ | ✗ | ✗ |
| w/o_Sobel | ✓ | ✗ | ✓ | ✓ |
| w/_Skeleton | ✓ | ✓ | ✓ | ✓ |

4 evaluation metrics:
- **CNA** (Component Neighbor Accuracy): % of components whose neighbor set exactly matches GT
- **GA** (Group Accuracy): % of connectivity groups perfectly matching GT
- **PC** (Port Correct Rate): % of port-pair edges matching GT
- **FP/FN**: false positive/negative rates

Outputs: `experiment_results.csv`, `experiment_summary.png`, `ablation_impact.png`.

## Wiring Algorithm

Steps execute in order (all gated by `config`):

| Step | What | Config key |
|------|------|-----------|
| Skeleton extraction | Zhang-Suen on adaptive-thresholded gray | `use_skeleton` |
| Port snap to skeleton (改进1) | Snap each port to nearest skeleton endpoint | `use_skeleton` |
| P2J (Step 6) | Port → nearest junction; skeleton-verified pairs lock immediately (改进2) | — |
| Direct P2P | Port → port midpoint | — |
| JJ (Step 7) | Aligned junction pairs, NN-filtered per direction (L/R/U/D) | `use_nn_filter` |
| JJ non-aligned | Diagonal junctions with skeleton verification, NMS-dedup | `use_skel_jj` |
| LOS | Line-of-sight between aligned ports, no bbox blocking | `use_los` |
| Close-port | Different-component ports within 50px | `use_close_port` |
| Force-connect | Isolated port → nearest valid junction | `use_force_connect` |
| CCL mode | Alternative: mask components → dilate wires → connected components | `use_ccl` |

**Key additions**:
- **改进1** (`_snap_ports_to_skeleton`): Replaces hardcoded port positions with skeleton endpoint coordinates. Reduces ~10-20px offset.
- **改进2** (P2J skeleton priority): If skeleton path exists between port and junction (min_ratio=0.50), locks connection immediately instead of competing on distance.

**Skeleton**: Zhang-Suen thinning of adaptive-thresholded grayscale. Used for:
- `_verify_skeleton_path()`: aligned wire coverage check (sampling, min_ratio=0.15)
- `_verify_skeleton_any()`: diagonal wire coverage check
- `_count_skeleton_branches()`: junction degree constraint
- `_trace_wire_connections()`: BFS from port to nearest junction

**Image-size normalization**: All distance thresholds scaled by `img_diag / 2000`.

## Key Rules

1. **LLM evaluation**: Always runs unless `config['skip_llm']=True`. Experiments should always skip it.
2. **Designator mismatch**: Pipeline and detection JSON have different designators. Always use IoU matching then index-based port labeling for evaluation — never assume designator names match.
3. **Port orientation**: GND never rotates. Capacitors/LEDs use Sobel edge detection. Other 2-pin comps use aspect ratio (bh/bw > 1.3 → rotate).
4. **OCR corrections**: `_ocr_variants()` handles H→kΩ, V→μF/Ω. `_clean_ocr()` handles 3V3→3.3V.
5. **CGHD annotations**: XML has component boxes + text values (ground truth for detection/OCR). Instances/ has polygon masks. SPICE files only exist for drafter_1 (12 circuits). **No wiring GT in CGHD** — only the manually annotated benchmark provides this.
6. **Do NOT delete/rename user files without explicit permission**. The user has lost benchmark data multiple times. The benchmark and picture directories are manually curated.

## External Paths

| Resource | Path |
|----------|------|
| CGHD v16 dataset | `E:\circuit_image\cghd-zenodo-16\` |
| CGHD YOLO format | `E:\circuit_data\cghd_yolo_v16\` |
| Tesseract | `E:\Tesseract-OCR\tesseract.exe` |
| Ngspice | `E:\ngspice\bin\ngspice.exe` |

## Publication

`发表路线图.md` — 3-stage publication plan (EI conference → EI journal → SCI). `docs/node_wire_tech_review.md` — 12-paper survey with 5 proposed improvements to wiring algorithm.
