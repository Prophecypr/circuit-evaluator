# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Image evaluation (main pipeline)
python -m src.vision.unified_pipeline text1.jpg
python -m src.vision.unified_pipeline text1.jpg text2.jpg text3.jpg

# Text evaluation
python eval.py data/samples/circuit_01_good.txt

# Training
python src/vision/train_ocr.py --train        # CRNN OCR
python src/vision/train_yolo.py                # YOLO detection
python src/vision/convert_cghd.py              # CGHD→YOLO format (need to update path)

# OCR data cropping
python -m src.vision.crop_machine_text

# Tests
pytest tests/ -v
```

## Architecture

**Main pipeline** (`src/vision/unified_pipeline.py`): Image → YOLO 61-class detection → HCD text detection → CRNN OCR → value matching → junction wiring → LLM evaluation → annotated image + TXT.

**Text pipelines** (`src/phase1_verify/`, `phase2_pipeline/`, `phase4_netlist/`): Text circuit description → LLM evaluation (Phase 1/2) or SPICE simulation (Phase 3/4).

**LLM** (`src/llm.py`): OpenAI-compatible SDK. Priority: `OLLAMA_BASE_URL` env var → Anthropic key → DeepSeek key. Ollama model default: `qwen2.5:14b`.

## Active Models

| Model | Path | Classes | Metric | Role |
|-------|------|:-------:|--------|------|
| CGHD 61cls | `runs/detect/cghd_61cls/weights/best.pt` | 61 | mAP50=74.0% | Component + junction detection |
| circuit_text | `runs/detect/circuit_text/weights/best.pt` | 1 | mAP50=99.4% | Text region detection |
| CRNN OCR | `runs/ocr_crnn_machine/best.pt` | 28 chars | val_loss=0.164 | Text recognition (machine-print trained) |
| Jumper wire | `runs/segment/jumper_wire/weights/best.pt` | 3 | mAP50=90.1% | Wire detection (CI2N, domain-limited) |

Legacy: `circuit_real`(17-class), `circuit_port`(4-class), `cghd_56cls`(old), `ocr_crnn`(hand-drawn) - kept for reference.

## Key Rules (DO NOT BREAK)

1. **LLM**: When `OLLAMA_BASE_URL` is not set, falls back to DeepSeek. Remove retry/short-prompt logic for Ollama (no content filtering).
2. **Drawing colors**: Legend shows only component types present in the current image.
3. **Junction numbering**: J1, J2... sorted by (y, x). Rebuild after synthetic junctions added.
4. **OCR corrections**: `_ocr_variants()` handles H→kΩ, V→μF/Ω, z→kΩ. `_clean_ocr()` handles 3V3→3.3V, double-dot fix. Context-aware: bare "33" near V-DC→3.3V.
5. **IC reading**: CRNN can't read chip names (only 28 numeric chars). Tesseract scans IC bbox ±20px for alphanumeric strings.
6. **Port orientation**: Sobel edge detection for 2-pin components. GND never rotates. Power sources can rotate.
7. **LOS connections**: Direct straight lines between aligned (<25px) ports with no bbox between them. GND-GND pairs excluded. Closest pair per component pair only.
8. **Crosses check**: 6px bbox expansion margin. Aligned junctions skip the check. Port-to-port target coordinates skip the check.

## External Paths

| Resource | Path |
|----------|------|
| CGHD v16 dataset | `E:\circuit_image\cghd-zenodo-16\` |
| CGHD v16 YOLO | `E:\circuit_data\cghd_yolo_v16\` |
| CGHD v13 (old) | `E:\circuit_image\cghd-zenodo-13\` |
| CI2N dataset | `E:\circuit_image\ci2n_datasets-main\` |
| Tesseract | `E:\Tesseract-OCR\tesseract.exe` |
| Ngspice | `E:\ngspice\bin\ngspice.exe` |

## Dataset Sources

- **CGHD v16**: 3,269 hand-drawn circuit images, 61 classes, 32 drafters. XML→YOLO via `convert_cghd_v16.py`.
- **Digitize-HCD**: 11,642 hand-drawn text crops for CRNN training. In `data/ocr_training/`.
- **OCR Machine**: 344 machine-printed text crops (Arial+TNR). In `data/ocr_machine/`.
- **CI2N jumper_identification**: 855 machine-drawn images, 3-class wire segmentation. In `E:\circuit_image\ci2n_datasets-main\jumper_identification\`.
