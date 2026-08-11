# Outward Port-Anchor Trace Implementation Plan

**Goal:** Recover wiring edges whose continuous skeleton reaches another component port before any detected junction, without enabling distance-only fallback or directional gap bridging.

**Publication baseline:** `strict_jj` remains unchanged. The candidate is isolated behind `use_outward_port_anchors=False` and is promoted only if the fixed worst-10 gate improves edge F1 without increasing FP.

## Tasks

1. Add typed trace-anchor helpers that include detected junctions and, only when enabled, ports belonging to other components.
2. Add unit tests proving source-component ports are excluded and a continuous trace can terminate at another component port when no junction exists.
3. Convert a traced port anchor into a deterministic synthetic midpoint shared by the two ports, and deduplicate reverse traces.
4. Add an independent `outward_port_anchors` experiment configuration based directly on `strict_jj`; do not inherit the rejected gap bridge.
5. Run wiring/unit regression tests.
6. Run the fixed worst-10 comparison: `strict_jj` versus `outward_port_anchors`.
7. Promote to the open 50-image benchmark only if worst-10 F1 increases and FP does not increase. Keep the final 42-image set sealed.

## Evidence and rollback

- Save per-image predictions, edge TP/FP/FN, and wiring traces under a new local `results/` directory.
- Keep `use_outward_port_anchors` disabled by default until the gate passes.
- If the gate fails, document the result and do not run the 50-image promotion experiment.
