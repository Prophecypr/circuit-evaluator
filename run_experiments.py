"""Run wiring ablation experiments and evaluate against GT annotations.

Usage: python run_experiments.py
Output: experiment_results.csv, experiment_summary.png, ablation_impact.png
"""
import json, math, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vision.unified_pipeline import process_image, DEFAULT_CONFIG

BENCHMARK = Path("benchmark")
RESULT = BENCHMARK / "result"
DETECTIONS = BENCHMARK / "detections"
FIXED = BENCHMARK / "fixed"  # manual corrections override detections/
PORT_MATCH_RADIUS = 60  # px tolerance for port position matching

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "Ours": {"skip_llm": True},
    "Baseline": {k: False for k in DEFAULT_CONFIG},
    "w/o_Sobel": {"use_sobel": False, "skip_llm": True},
    "w/_Skeleton": {"use_skeleton": True, "skip_llm": True},
    "CCL": {"use_ccl": True, "skip_llm": True},
}
ABLATION_CONFIGS["Baseline"]["skip_llm"] = True


# ---------------------------------------------------------------------------
# GT parsing
# ---------------------------------------------------------------------------
def parse_gt(gt_path):
    """Parse a GT file into list of groups, each group = list of (designator, port_label)."""
    groups = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":", 1)[1].strip()
            if not parts:
                continue
            entries = [e.strip() for e in parts.split(",")]
            port_pairs = []
            for e in entries:
                if "." in e:
                    desig, label = e.rsplit(".", 1)
                    port_pairs.append((desig, label))
            if len(port_pairs) >= 2:
                groups.append(port_pairs)
    return groups


# ---------------------------------------------------------------------------
# Component matching: pipeline components ↔ detection JSON components
# ---------------------------------------------------------------------------
def match_components(pipeline_comps, det_comps):
    """Match pipeline components to detection JSON components by bounding-box IoU.

    Returns dict: pipeline_comp_idx -> detection_comp_idx
    """
    matches = {}
    used = set()
    for pi, pc in enumerate(pipeline_comps):
        px1, py1, px2, py2 = pc.get("xyxy", [0, 0, 0, 0])
        best_iou, best_di = 0.0, None
        for di, dc in enumerate(det_comps):
            if di in used:
                continue
            dx1, dy1, dx2, dy2 = dc["xyxy"]
            ix1, iy1 = max(px1, dx1), max(py1, dy1)
            ix2, iy2 = min(px2, dx2), min(py2, dy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_p = (px2 - px1) * (py2 - py1)
            area_d = (dx2 - dx1) * (dy2 - dy1)
            iou = inter / (area_p + area_d - inter) if (area_p + area_d - inter) > 0 else 0
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_di = di
        if best_di is not None:
            matches[pi] = best_di
            used.add(best_di)
    return matches


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(pipeline_result, gt_groups, det_comps):
    """Compute 4 metrics by matching pipeline components to detection JSON components.

    Uses port-position matching to establish (designator, port_label) identity for
    each pipeline port, then compares predicted groups against GT groups.
    """
    pipeline_comps = pipeline_result["components"]
    raw_groups = pipeline_result["raw_groups"]  # list of sets of (ci, pi)

    # 1. Match pipeline components → detection JSON components (by IoU)
    comp_match = match_components(pipeline_comps, det_comps)

    # 2. Map pipeline ports to detection JSON ports by INDEX (matching order)
    #    Pipeline and detection JSON use the same PORT_POSITIONS, so ports are
    #    in the same order. Fall back to position match if counts differ.
    pipeline_port_id = {}  # (ci, pi) -> "designator.label"
    for ci, c in enumerate(pipeline_comps):
        det_idx = comp_match.get(ci)
        if det_idx is None:
            continue
        dc = det_comps[det_idx]
        n_pipe_ports = len(c["ports"])
        n_det_ports = len(dc["ports"])
        if n_pipe_ports == n_det_ports:
            # Happy path: same number of ports, match by index
            for pi in range(n_pipe_ports):
                label = dc["labels"][pi] if pi < len(dc["labels"]) else "?"
                pipeline_port_id[(ci, pi)] = f"{dc['designator']}.{label}"
        else:
            # Fallback: port counts differ, use position match
            for pi, (px, py) in enumerate(c["ports"]):
                best_label = None
                best_dist = PORT_MATCH_RADIUS
                for dp_idx, (dpx, dpy) in enumerate(dc["ports"]):
                    d = math.hypot(px - dpx, py - dpy)
                    if d < best_dist:
                        best_dist = d
                        best_label = dc["labels"][dp_idx] if dp_idx < len(dc["labels"]) else "?"
                if best_label is not None:
                    pipeline_port_id[(ci, pi)] = f"{dc['designator']}.{best_label}"

    # 3. Build predicted groups as sets of "designator.label" strings
    pred_groups = []
    for port_set in raw_groups:
        group_entries = set()
        for ci, pi in port_set:
            pid = pipeline_port_id.get((ci, pi))
            if pid:
                group_entries.add(pid)
        if len(group_entries) >= 2:
            pred_groups.append(group_entries)

    # 4. Build GT groups as sets
    gt_sets = []
    all_gt_ports = set()
    for g in gt_groups:
        entries = set(f"{d}.{l}" for d, l in g)
        gt_sets.append(entries)
        all_gt_ports.update(entries)

    gt_n_ports = len(all_gt_ports) if all_gt_ports else 1

    # 5. Match predicted groups to GT groups
    matched_gt = set()
    matched_pred = set()
    gt_to_pred = {}  # gt_idx -> pred_idx

    for pi, ps in enumerate(pred_groups):
        for gi, gs in enumerate(gt_sets):
            if gi in matched_gt:
                continue
            if ps == gs:
                matched_pred.add(pi)
                matched_gt.add(gi)
                gt_to_pred[gi] = pi
                break

    # Collect all predicted ports
    all_pred_ports = set()
    for pg in pred_groups:
        all_pred_ports.update(pg)

    # Metrics
    # Group accuracy
    group_accuracy = len(matched_gt) / len(gt_sets) if gt_sets else 0.0

    # Designator-level groups (for component-neighbor accuracy)
    gt_desig_groups = [set(d for d, l in g) for g in gt_groups]
    pred_desig_groups = []
    for pg in pred_groups:
        desigs = set(e.split(".")[0] for e in pg)
        if len(desigs) >= 2:
            pred_desig_groups.append(desigs)

    # Component neighbor accuracy: for each component, compare its neighbor set
    # (other components in the same group) between GT and prediction
    def get_neighbors(groups):
        neighbors = {}
        for g in groups:
            comps = sorted(g)
            for c in comps:
                neighbors[c] = g - {c}
        return neighbors

    gt_neighbors = get_neighbors(gt_desig_groups)
    pred_neighbors = get_neighbors(pred_desig_groups)

    all_comps = set(gt_neighbors.keys()) | set(pred_neighbors.keys())
    neighbor_correct = 0
    neighbor_total = len(all_comps)
    for comp in all_comps:
        gt_n = gt_neighbors.get(comp, set())
        pred_n = pred_neighbors.get(comp, set())
        if gt_n == pred_n:
            neighbor_correct += 1

    comp_neighbor_accuracy = neighbor_correct / neighbor_total if neighbor_total else 0.0

    # Port-level metrics via edge comparison
    def group_to_edges(groups):
        edges = set()
        for g in groups:
            ports = sorted(g)
            for i in range(len(ports)):
                for j in range(i + 1, len(ports)):
                    edges.add((ports[i], ports[j]))
        return edges

    gt_edges = group_to_edges(gt_sets)
    pred_edges = group_to_edges(pred_groups)

    tp_edges = len(gt_edges & pred_edges)
    fp_edges = len(pred_edges - gt_edges)
    fn_edges = len(gt_edges - pred_edges)

    port_correct_rate = tp_edges / len(gt_edges) if gt_edges else 0.0
    fp_rate = fp_edges / len(pred_edges) if pred_edges else 0.0
    fn_rate = fn_edges / len(gt_edges) if gt_edges else 0.0

    return {
        "port_correct_rate": port_correct_rate,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "group_accuracy": group_accuracy,
        "comp_neighbor_accuracy": comp_neighbor_accuracy,
        "n_gt_groups": len(gt_sets),
        "n_pred_groups": len(pred_groups),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def get_image_list():
    """Return list of (img_name, img_path, gt_path, det_path) for images with GT."""
    images = []
    for gt_file in sorted(RESULT.glob("*_gt.txt")):
        stem = gt_file.stem.replace("_gt", "")
        img_name = stem + ".jpg"
        img_path = BENCHMARK / img_name
        det_path = FIXED / (stem + ".json")
        if not det_path.exists():
            det_path = DETECTIONS / (stem + ".json")
        if img_path.exists() and det_path.exists():
            images.append((stem, str(img_path), str(gt_file), str(det_path)))
    return images


def main():
    images = get_image_list()
    configs = list(ABLATION_CONFIGS.keys())
    print(f"Running {len(images)} images x {len(configs)} configs...")
    print(f"Configs: {', '.join(configs)}")
    print()

    results = []
    for img_idx, (stem, img_path, gt_path, det_path) in enumerate(images):
        # Load GT and detection JSON once per image
        gt_groups = parse_gt(gt_path)
        with open(det_path, encoding="utf-8") as f:
            det_comps = json.load(f)["components"]
        n_comps = len(det_comps)

        print(f"[{img_idx+1}/{len(images)}] {stem} ({n_comps} components, {len(gt_groups)} GT groups)")

        for cfg_name in configs:
            cfg = ABLATION_CONFIGS[cfg_name]
            try:
                result = process_image(img_path, config=dict(cfg))
                if result is None:
                    raise RuntimeError("process_image returned None")
                metrics = evaluate(result, gt_groups, det_comps)
            except Exception as e:
                print(f"  {cfg_name}: ERROR - {e}")
                metrics = {"port_correct_rate": None, "fp_rate": None,
                           "fn_rate": None, "group_accuracy": None,
                           "comp_neighbor_accuracy": None,
                           "n_gt_groups": len(gt_groups), "n_pred_groups": 0}

            results.append({
                "image": stem, "config": cfg_name, "n_components": n_comps,
                **metrics,
            })
            if metrics["port_correct_rate"] is not None:
                print(f"  {cfg_name}: PC={metrics['port_correct_rate']:.3f} "
                      f"FP={metrics['fp_rate']:.3f} FN={metrics['fn_rate']:.3f} "
                      f"GA={metrics['group_accuracy']:.3f}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = "experiment_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["image", "config", "n_components", "n_gt_groups", "n_pred_groups",
                "port_correct_rate", "fp_rate", "fn_rate", "group_accuracy",
                "comp_neighbor_accuracy"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nSaved {csv_path} ({len(results)} rows)")

    # -----------------------------------------------------------------------
    # Compute summary statistics (mean per config)
    # -----------------------------------------------------------------------
    summary = {}
    for cfg_name in configs:
        cfg_results = [r for r in results if r["config"] == cfg_name and r["port_correct_rate"] is not None]
        if cfg_results:
            summary[cfg_name] = {
                "port_correct_rate": np.mean([r["port_correct_rate"] for r in cfg_results]),
                "fp_rate": np.mean([r["fp_rate"] for r in cfg_results]),
                "fn_rate": np.mean([r["fn_rate"] for r in cfg_results]),
                "group_accuracy": np.mean([r["group_accuracy"] for r in cfg_results]),
                "comp_neighbor_accuracy": np.mean([r["comp_neighbor_accuracy"] for r in cfg_results]),
                "n": len(cfg_results),
            }

    # -----------------------------------------------------------------------
    # Plot 1: experiment_summary.png — grouped bar chart of all configs
    # -----------------------------------------------------------------------
    metrics = ["comp_neighbor_accuracy", "group_accuracy", "port_correct_rate"]
    metric_labels = ["Comp Neighbor Acc", "Group Accuracy", "Port Correct Rate"]
    x = np.arange(len(metrics))
    n_configs = len(configs)
    width = 0.8 / n_configs
    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, cfg_name in enumerate(configs):
        if cfg_name in summary:
            values = [summary[cfg_name][m] for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=cfg_name, color=colors[i])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_ylabel("Score")
    ax.set_title("Wiring Algorithm: Full vs Baselines vs Ablations")
    ax.set_xticks(x + width * (n_configs - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig("experiment_summary.png", dpi=150)
    print("Saved experiment_summary.png")

    # -----------------------------------------------------------------------
    # Plot 2: ablation_impact.png — delta from full
    # -----------------------------------------------------------------------
    if "Ours" in summary:
        full_vals = summary["Ours"]
        ablation_names = [c for c in configs if c not in ("Ours", "Baseline", "CCL")]
        x2 = np.arange(len(ablation_names))
        width2 = 0.8 / len(metrics)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for j, (m, ml) in enumerate(zip(metrics, metric_labels)):
            deltas = []
            for ab_name in ablation_names:
                if ab_name in summary:
                    deltas.append(summary[ab_name][m] - full_vals[m])
                else:
                    deltas.append(0)
            bars = ax2.bar(x2 + j * width2, deltas, width2, label=ml)
            for bar, val in zip(bars, deltas):
                y_pos = bar.get_height() if val >= 0 else bar.get_height() - 0.03
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                         f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)

        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Delta from Full")
        ax2.set_title("Ablation Impact (Delta from Full Algorithm)")
        ax2.set_xticks(x2 + width2 * (len(metrics) - 1) / 2)
        ax2.set_xticklabels(ablation_names)
        ax2.legend(loc="lower left", fontsize=7)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig2.savefig("ablation_impact.png", dpi=150)
        print("Saved ablation_impact.png")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    header = f"{'Config':<16} {'CNA':>8} {'GA':>8} {'PC':>8} {'FP':>8} {'FN':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for cfg_name in configs:
        if cfg_name in summary:
            s = summary[cfg_name]
            print(f"{cfg_name:<16} {s['comp_neighbor_accuracy']:8.3f} {s['group_accuracy']:8.3f} "
                  f"{s['port_correct_rate']:8.3f} {s['fp_rate']:8.3f} "
                  f"{s['fn_rate']:8.3f} {s['n']:6d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
