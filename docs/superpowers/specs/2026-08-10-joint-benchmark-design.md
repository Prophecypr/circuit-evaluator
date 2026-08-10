# CRNN Three-Seed and Joint Benchmark Design

## Goal

Freeze the three CRNN-v2 seed results, select the deployment candidate using only the held-out 4,801-crop validation set, and evaluate the complete visual pipeline on all 50 GT-backed `benchmark/` images with the LLM disabled.

## Inputs and separation

- OCR seeds: 42, 2026, and 3407. Training data, validation manifests, architecture, and hyperparameters must be identical except for `data.seed`.
- Component and port GT: `benchmark/fixed/*.json`.
- Wiring GT: `benchmark/result/*_gt.txt`.
- Junction and text GT: the matching CGHD XML files under `E:\circuit_image\cghd-zenodo-16`.
- The separate 42-image hand-drawn final test set is not read by this experiment.

## Pipeline

The runner calls `process_image` once per benchmark image with `skip_llm=True`, `save_artifacts=False`, and an explicit OCR checkpoint. It stores predictions under a new run directory and never writes generated files into `benchmark/`.

Raw YOLO junction and text detections are exposed in the pipeline result in addition to the existing post-processed components, OCR text, junctions, and wiring groups. This makes detector evaluation independent of later graph processing.

## Metrics

- Components: class-aware box precision, recall, F1, AP50, mAP50, and per-class counts.
- Junction detection: raw class-aware box metrics against XML `junction` objects; post-processed point precision, recall, and F1 against GT junction centers.
- Ports: localization precision/recall/F1 after IoU component matching, plus index-label correctness.
- Text/value: raw text-box detection metrics, all-text normalized exact/CER, and electrical-value-only normalized exact/CER. Unmatched GT text counts as an empty OCR prediction.
- Wiring: existing PC, GA, CNA, FP, and FN plus explicitly named edge precision, recall, and F1 derived from the same port-edge counts.

## Outputs

Each run writes `summary.json`, per-image CSV/JSON records, module metric JSON files, OCR error CSV files, rendered prediction images, model/data hashes, and a failure log. Partial failures remain visible and are included in the denominator; they are never silently dropped.

