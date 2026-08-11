# Directional Morphological Wire Closing Plan

**Goal:** Recover short horizontal or vertical breaks in the binary wire mask before thinning, while avoiding diagonal and unrestricted proximity merges.

## Constraints

- Keep the accepted `outward_port_anchors` configuration as the comparison baseline.
- Add one independent flag, disabled by default until the gate passes.
- Use separate 5x1 and 1x5 closing kernels and union their outputs with the original mask.
- Do not change P2J, P2P, JJ, LOS, or distance thresholds.
- Do not use OCR, LLM, or the sealed final 42-image test set.

## Verification

1. Unit test: close one short horizontal gap.
2. Unit test: close one short vertical gap.
3. Unit test: do not connect diagonally offset segments.
4. Wiring and experiment regression tests.
5. Fixed worst-10 A/B: `outward_port_anchors` versus `directional_morph_close`.
6. Run the open 50-image comparison only if worst-10 edge F1 rises and FP does not increase.
