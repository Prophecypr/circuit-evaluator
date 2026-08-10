# Wiring Graph Reliability Design

## 1. Goal and scope

Improve the LLM-free visual pipeline's recovery of component-port connectivity while preserving the already strong component, junction, and text detection results. The primary publication metric is micro wiring-edge F1 on the 50-image GT-backed CGHD benchmark. All six requested changes are included:

1. Model `terminal` as a one-port component instead of merging it into `junction`.
2. Render predicted P2J/JJ links and color every final predicted network independently.
3. Record connection candidates and accepted/rejected counts at every wiring stage.
4. Remove or constrain unverified fallback port-to-junction connections.
5. Replace nearest-node port assignment with outward skeleton tracing where evidence is available.
6. Require continuous skeleton evidence and explicit crossing semantics for junction-to-junction links.

The existing 50 images are the development/ablation benchmark. The separate 42 hand-drawn final-test images remain sealed. LLM calls remain disabled for all vision experiments.

## 2. Evidence and current failure mode

- Raw YOLO junction detection is already strong: center-distance F1 95.33% on 50 images.
- Naive skeleton-branch junction discovery is unsuitable: F1 7.39% with component masking and 8.04% after also masking text, due to paper texture, residual symbol strokes, handwriting, and wire artifacts.
- Current 50-image wiring-edge F1 is 24.91% (TP 327, FP 991, FN 980).
- On the C170/C171/C274 diagnostic subset, disabling skeleton increased edge FP from 314 to 449 and reduced edge F1 from 14.14% to 12.86%. Skeleton evidence helps, but graph construction remains the bottleneck.
- The current annotated image shows detections but hides the actual port-to-node and node-to-node assignments, making graph errors hard to diagnose visually.

Therefore the pipeline remains detector-first: YOLO supplies junction candidates, then skeleton evidence validates topology. Skeleton branch points may be diagnostic candidates near detected junctions, but never replace the junction detector globally.

## 3. Approaches considered

### A. Threshold-only patching

Keep the monolithic pipeline and tune P2J/JJ radii and coverage thresholds. This is fast but risks overfitting the 50 development images, does not expose which stage creates an error, and is weak publication evidence.

### B. Traceable graph-construction layer (selected)

Retain the current detector and OCR, but introduce small pure functions for terminal conversion, skeleton-path candidate generation, stage decisions, graph events, and rendering. Every change is independently configurable and ablatable. This provides reproducible evidence and limits regression risk without requiring a new dataset.

### C. End-to-end wire segmentation and keypoint models

Train wire masks and semantic port keypoints. This may eventually outperform heuristics, but requires new pixel/keypoint training annotations and would contaminate the present schedule if attempted before the current graph logic is measured and corrected.

Approach B is selected now. Approach C remains a later experiment if graph F1 plateaus.

## 4. Architecture and data flow

The current detector and OCR front half remain unchanged. The wiring half becomes:

```text
YOLO components/junctions/terminals/text
  -> terminal-to-one-port-component conversion
  -> port template/orientation assignment
  -> component-masked skeleton
  -> outward port trace candidates
  -> strict P2J decision
  -> strict JJ decision with crossing semantics
  -> optional low-risk P2P/LOS candidates
  -> Union-Find network construction
  -> stage trace JSON + colored graph rendering
```

### 4.1 Terminal semantics

- A `terminal` detection becomes a component named `Terminal` with one port at the detected box center and semantic label `T`.
- It receives a stable designator (`T1`, `T2`, ...).
- It remains available in `raw_junction_detections` for audit, but its center is not added to the junction list.
- The GT evaluator continues to match components by class and IoU, then ports by distance.

### 4.2 Wiring trace contract

The pipeline result gains `wiring_trace`, containing one record for each considered connection:

```json
{
  "stage": "p2j_trace",
  "kind": "p2j",
  "source": {"component_index": 3, "port_index": 1},
  "target": {"junction": [120, 240]},
  "accepted": true,
  "reason": "continuous_skeleton_path",
  "path_coverage": 0.91,
  "distance": 38.2
}
```

Every stage reports candidate, accepted, and rejected counts. Rejections use stable reason codes such as `no_skeleton_path`, `crosses_component`, `would_short_component`, `ambiguous_crossing`, and `outside_direction_cone`.

### 4.3 Colored graph rendering

- Each final Union-Find network receives a deterministic color based on its sorted port identities.
- Component ports and accepted P2J/JJ segments use that network color.
- Rejected candidates are not drawn in the publication image; an optional debug rendering draws them as thin dashed gray/red lines.
- A sidecar `<stem>_wiring_trace.json` stores the complete evidence so visualization does not replace numeric evaluation.

### 4.4 Strict P2J logic

The main assignment starts at the predicted port and searches outward from the component center through a direction cone. It attaches to the first reachable detected junction or terminal/port anchor on the same skeleton path.

Rules:

- Continuous skeleton support is required beyond a short local gap allowance.
- A port cannot attach through another component box.
- A two-port component cannot attach both ports to the same network.
- Candidate selection ranks path continuity first, then path length, then Euclidean distance.
- The existing unverified second-pass nearest-junction fallback is disabled in the strict configuration. A compatibility switch preserves it only for ablation.

### 4.5 Skeleton tracing

The trace begins at the closest skeleton pixel within a class-scaled boundary radius and follows the skeleton by BFS. Component interiors and detected text regions are masked. Small gaps may be bridged only when direction continuity is preserved. The trace stops at the first detected junction, terminal, or different-component port anchor; it does not jump to an arbitrary nearby node after dead-ending.

### 4.6 Strict JJ and crossing semantics

- JJ edges require a continuous skeleton path, not only alignment plus low sampled coverage.
- At a line crossing, networks merge only if a YOLO junction is present within the crossing tolerance or the skeleton topology provides an unambiguous branch anchored by that detection.
- A detected terminal never acts as a multiway junction.
- Nearest-neighbor filtering remains, but operates only on verified path candidates.
- Non-aligned JJ candidates use the same continuous trace evidence as aligned candidates.

## 5. Configuration and ablation

Each behavior has an independent defaulted key:

- `use_terminal_components`
- `use_wiring_trace`
- `use_strict_p2j`
- `use_outward_skeleton_trace`
- `use_strict_jj`
- `use_crossing_semantics`

The legacy path remains selectable for controlled ablations. The final experiment sequence is cumulative so each row measures the marginal effect of one change:

1. Frozen current baseline.
2. Baseline + observability only (metric-equivalence check).
3. + Terminal component semantics.
4. + Strict fallback policy.
5. + Outward skeleton P2J trace.
6. + Strict JJ continuous paths.
7. + Crossing semantics (full proposed system).

## 6. Metrics and acceptance criteria

Primary:

- Micro wiring-edge precision, recall, and F1 across all 50 images.

Secondary:

- Macro group accuracy, component-neighbor accuracy, and port correct rate.
- Port localization F1 and semantic-label accuracy.
- Predicted network-size distribution and count of abnormally large networks.
- Per-stage accepted/rejected candidate counts.

Guardrails:

- 50/50 images complete with zero pipeline and render failures.
- Component, raw junction, text detection, and OCR predictions do not change except for the deliberate Terminal representation.
- Observability-only configuration is metric-equivalent to the frozen baseline.
- A cumulative algorithm change is retained in the proposed default only if full-50 wiring-edge F1 improves; precision/recall trade-offs remain reported even when the change is rejected.
- No final-42 image is opened or used.

## 7. Testing strategy

All production behavior is introduced by red-green TDD.

Unit tests cover:

- Terminal conversion and exclusion from junction centers.
- Deterministic network colors and trace serialization.
- Stage counters and rejection reason codes.
- Unverified fallback rejection.
- Outward tracing choosing the first reachable anchor and rejecting backward/cross-component candidates.
- JJ accepting continuous paths and rejecting blank gaps.
- Crossings without junction dots remaining separate.
- Crossings with detected junction dots merging.

Integration tests use synthetic binary skeletons and minimal component graphs before any 50-image run. Existing wiring and joint-benchmark test suites remain mandatory. Full-50 experiments write into a new isolated results directory and record code revision, configuration, model hashes, GT hashes, and per-image metrics.

## 8. Non-goals

- No LLM scoring changes.
- No OCR retraining or component-detector retraining.
- No pixel-level wire-segmentation training in this phase.
- No tuning or inspection against the sealed 42-image final test.
- No deletion or renaming of curated benchmark files.
