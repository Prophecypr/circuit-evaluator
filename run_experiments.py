"""Run wiring ablation experiments and evaluate against GT annotations.

Usage: python run_experiments.py [--output-dir results/visual_wiring_run]
Output: CSV, plots, and metadata in a dedicated run directory.
"""
import argparse
import csv
import hashlib
import json, math, os, sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from uuid import uuid4
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PORT_MATCH_RADIUS = 60  # px tolerance for port position matching

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
def build_ablation_configs():
    """Return complete, independent visual-only configurations for each ablation."""
    full = {**DEFAULT_CONFIG, "skip_llm": True, "save_artifacts": False}
    return {
        "Ours": dict(full),
        "Baseline": {
            **{key: False for key in DEFAULT_CONFIG},
            "skip_llm": True,
        },
        "w/o_Skeleton": {**full, "use_skeleton": False},
        "w/o_Sobel": {**full, "use_sobel": False},
        "w/o_NN_Filter": {**full, "use_nn_filter": False},
        "w/o_Close_Port": {**full, "use_close_port": False},
        "w/o_Component_Mask": {**full, "use_component_mask": False},
        "CCL": {**full, "use_ccl": True},
    }


def build_wiring_reliability_configs():
    """Return cumulative, visual-only configs for the wiring reliability study."""
    baseline = {
        **DEFAULT_CONFIG,
        "skip_llm": True,
        "skip_ocr": True,
        "save_artifacts": False,
        "use_wiring_trace": False,
        "use_terminal_components": False,
        "use_strict_p2j": False,
        "use_outward_skeleton_trace": False,
        "use_strict_jj": False,
        "use_crossing_semantics": False,
    }
    observability = {**baseline, "use_wiring_trace": True}
    terminal = {**observability, "use_terminal_components": True}
    strict_fallback = {**terminal, "use_strict_p2j": True}
    outward_trace = {**strict_fallback, "use_outward_skeleton_trace": True}
    strict_jj = {**outward_trace, "use_strict_jj": True}
    crossing_semantics = {**strict_jj, "use_crossing_semantics": True}
    return {
        "frozen_baseline": baseline,
        "observability": observability,
        "terminal": terminal,
        "strict_fallback": strict_fallback,
        "outward_trace": outward_trace,
        "strict_jj": strict_jj,
        "crossing_semantics": crossing_semantics,
    }


def resolve_output_dir(output_dir=None):
    """Create and return a dedicated output directory for one experiment run."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path("results") / f"visual_wiring_{timestamp}_{uuid4().hex}"
        output_path.mkdir(parents=True)
        return output_path

    output_path = Path(output_dir)
    if output_path.exists():
        if not output_path.is_dir() or any(output_path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty experiment output directory: {output_path}"
            )
    else:
        output_path.mkdir(parents=True)
    return output_path


def get_git_revision():
    """Return the current Git revision, or None when Git metadata is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


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
    edge_precision = tp_edges / (tp_edges + fp_edges) if tp_edges + fp_edges else 0.0
    edge_recall = tp_edges / (tp_edges + fn_edges) if tp_edges + fn_edges else 0.0
    edge_f1 = (
        2 * edge_precision * edge_recall / (edge_precision + edge_recall)
        if edge_precision + edge_recall else 0.0
    )

    return {
        "port_correct_rate": port_correct_rate,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "group_accuracy": group_accuracy,
        "comp_neighbor_accuracy": comp_neighbor_accuracy,
        "edge_tp": tp_edges,
        "edge_fp": fp_edges,
        "edge_fn": fn_edges,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "n_gt_groups": len(gt_sets),
        "n_pred_groups": len(pred_groups),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def get_image_list(benchmark_dir=BENCHMARK):
    """Return every GT-backed image, refusing silent omissions or duplicate stems."""
    benchmark_dir = Path(benchmark_dir)
    result_dir = benchmark_dir / "result"
    detections_dir = benchmark_dir / "detections"
    fixed_dir = benchmark_dir / "fixed"
    images = []

    for gt_file in sorted(result_dir.glob("*_gt.txt")):
        stem = gt_file.stem.removesuffix("_gt")
        candidates = sorted(
            path for path in benchmark_dir.iterdir()
            if path.is_file()
            and path.stem == stem
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not candidates:
            raise FileNotFoundError(f"{stem}: missing benchmark image")
        if len(candidates) > 1:
            joined = ", ".join(path.name for path in candidates)
            raise RuntimeError(f"{stem}: multiple images share the GT stem: {joined}")

        fixed_path = fixed_dir / f"{stem}.json"
        detected_path = detections_dir / f"{stem}.json"
        if fixed_path.is_file():
            det_path = fixed_path
        elif detected_path.is_file():
            det_path = detected_path
        else:
            raise FileNotFoundError(f"{stem}: missing detection JSON")

        images.append((stem, str(candidates[0]), str(gt_file), str(det_path)))
    return images


def _file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _aggregate_stage_summaries(rows):
    totals = defaultdict(lambda: {
        "candidates": 0, "accepted": 0, "rejected": 0, "reasons": defaultdict(int),
    })
    for row in rows:
        for stage, values in row.get("stage_summary_raw", {}).items():
            target = totals[stage]
            for key in ("candidates", "accepted", "rejected"):
                target[key] += int(values.get(key, 0))
            for reason, count in values.get("reasons", {}).items():
                target["reasons"][reason] += int(count)
    return {
        stage: {
            "candidates": values["candidates"],
            "accepted": values["accepted"],
            "rejected": values["rejected"],
            "reasons": dict(sorted(values["reasons"].items())),
        }
        for stage, values in sorted(totals.items())
    }


def run_wiring_reliability_experiment(
    *,
    output_dir,
    benchmark_dir=BENCHMARK,
    selected_images=None,
    expected_count=None,
    process_fn=None,
):
    """Run the cumulative wiring suite without LLM calls or final-42 inputs."""
    output = resolve_output_dir(output_dir)
    images = get_image_list(benchmark_dir)
    if selected_images:
        requested = {str(stem) for stem in selected_images}
        found = {stem for stem, *_ in images}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"unknown requested benchmark images: {', '.join(missing)}")
        images = [item for item in images if item[0] in requested]
    if expected_count is not None and len(images) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} benchmark cases, found {len(images)}"
        )
    if not images:
        raise RuntimeError("wiring reliability experiment has no images")

    configs = build_wiring_reliability_configs()
    process_fn = process_fn or process_image
    rows = []
    failures = []
    print(f"Output directory: {output}")
    print(f"Wiring reliability: {len(images)} images x {len(configs)} configs")

    for image_index, (stem, image_path, gt_path, det_path) in enumerate(images, start=1):
        gt_groups = parse_gt(gt_path)
        with open(det_path, encoding="utf-8") as handle:
            det_comps = json.load(handle)["components"]
        print(f"[{image_index}/{len(images)}] {stem}", flush=True)
        for config_name, config in configs.items():
            stage_summary = {}
            error = ""
            try:
                result = process_fn(image_path, config=dict(config))
                if result is None:
                    raise RuntimeError("process_image returned None")
                if result.get("evaluation") not in (None, "(LLM skipped for experiment)"):
                    raise RuntimeError("pipeline did not confirm LLM skip")
                metrics = evaluate(result, gt_groups, det_comps)
                stage_summary = result.get("wiring_trace", {}).get("summary", {})
                _write_json(output / "predictions" / config_name / f"{stem}.json", result)
                print(
                    f"  {config_name}: TP={metrics['edge_tp']} FP={metrics['edge_fp']} "
                    f"FN={metrics['edge_fn']} F1={metrics['edge_f1']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"image": stem, "config": config_name, "error": error})
                gt_edges = sum(len(group) * (len(group) - 1) // 2 for group in gt_groups)
                metrics = {
                    "port_correct_rate": 0.0, "fp_rate": 0.0, "fn_rate": 1.0,
                    "group_accuracy": 0.0, "comp_neighbor_accuracy": 0.0,
                    "edge_tp": 0, "edge_fp": 0, "edge_fn": gt_edges,
                    "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0,
                    "n_gt_groups": len(gt_groups), "n_pred_groups": 0,
                }
                print(f"  {config_name}: ERROR - {error}", flush=True)
            rows.append({
                "image": stem,
                "config": config_name,
                "n_components": len(det_comps),
                **metrics,
                "stage_summary": json.dumps(stage_summary, ensure_ascii=False, sort_keys=True),
                "stage_summary_raw": stage_summary,
                "error": error,
            })

    csv_columns = [
        "image", "config", "n_components", "n_gt_groups", "n_pred_groups",
        "edge_tp", "edge_fp", "edge_fn", "edge_precision", "edge_recall", "edge_f1",
        "port_correct_rate", "fp_rate", "fn_rate", "group_accuracy",
        "comp_neighbor_accuracy", "stage_summary", "error",
    ]
    with (output / "experiment_results.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for config_name in configs:
        config_rows = [row for row in rows if row["config"] == config_name]
        tp = sum(int(row["edge_tp"]) for row in config_rows)
        fp = sum(int(row["edge_fp"]) for row in config_rows)
        fn = sum(int(row["edge_fn"]) for row in config_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[config_name] = {
            "edge": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            },
            "macro_group_accuracy": float(np.mean([row["group_accuracy"] for row in config_rows])),
            "macro_component_neighbor_accuracy": float(np.mean([row["comp_neighbor_accuracy"] for row in config_rows])),
            "stage_summary": _aggregate_stage_summaries(config_rows),
            "failed_images": sum(bool(row["error"]) for row in config_rows),
        }
    _write_json(output / "wiring_reliability_summary.json", summary)
    _write_json(output / "failures.json", failures)
    _write_json(output / "run_metadata.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": "wiring-reliability",
        "git_revision": get_git_revision(),
        "benchmark_dir": str(Path(benchmark_dir).resolve()),
        "image_count": len(images),
        "expected_case_count": expected_count,
        "selected_images": [stem for stem, *_ in images],
        "config_count": len(configs),
        "configs": configs,
        "ocr_model_sha256": _file_sha256(DEFAULT_CONFIG["ocr_model_path"]),
        "detector_model_sha256": _file_sha256("runs/detect/cghd_61cls/weights/best.pt"),
        "failure_count": len(failures),
        "final_42_image_test_used": False,
    })
    print(f"Saved {len(rows)} rows; failures={len(failures)}")
    return output


def merge_wiring_reliability_shards(shard_dirs, output_dir, expected_count):
    """Merge disjoint shard outputs after exact provenance and coverage checks."""
    shard_dirs = [Path(path) for path in shard_dirs]
    if not shard_dirs:
        raise RuntimeError("no wiring reliability shards supplied")
    expected_configs = build_wiring_reliability_configs()
    reference = None
    rows = []
    failures = []
    seen_pairs = set()
    image_configs = defaultdict(set)

    for shard in shard_dirs:
        metadata = json.loads((shard / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("suite") != "wiring-reliability":
            raise RuntimeError(f"invalid wiring reliability shard: {shard}")
        if metadata.get("final_42_image_test_used") is not False:
            raise RuntimeError(f"shard used or failed to declare sealed final-42 status: {shard}")
        provenance = {
            "git_revision": metadata.get("git_revision"),
            "configs": metadata.get("configs"),
            "ocr_model_sha256": metadata.get("ocr_model_sha256"),
            "detector_model_sha256": metadata.get("detector_model_sha256"),
        }
        if reference is None:
            reference = provenance
        elif provenance != reference:
            raise RuntimeError(f"shard provenance mismatch: {shard}")
        if metadata.get("configs") != expected_configs:
            raise RuntimeError(f"shard configs differ from current suite: {shard}")

        with (shard / "experiment_results.csv").open(
            encoding="utf-8-sig", newline="",
        ) as handle:
            shard_rows = list(csv.DictReader(handle))
        for row in shard_rows:
            key = (row["image"], row["config"])
            if key in seen_pairs:
                raise RuntimeError(
                    f"duplicate image/config across shards: {row['image']} {row['config']}"
                )
            seen_pairs.add(key)
            image_configs[row["image"]].add(row["config"])
            rows.append(row)
        failure_path = shard / "failures.json"
        if failure_path.is_file():
            failures.extend(json.loads(failure_path.read_text(encoding="utf-8")))

    expected_config_names = set(expected_configs)
    if len(image_configs) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} unique images across shards, found {len(image_configs)}"
        )
    incomplete = {
        image: sorted(expected_config_names - config_names)
        for image, config_names in image_configs.items()
        if config_names != expected_config_names
    }
    if incomplete:
        raise RuntimeError(f"incomplete shard config coverage: {incomplete}")

    output = resolve_output_dir(output_dir)
    for shard in shard_dirs:
        for source in (shard / "predictions").glob("*/*.json"):
            destination = output / "predictions" / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    rows.sort(key=lambda row: (row["image"], list(expected_configs).index(row["config"])))
    columns = list(rows[0])
    with (output / "experiment_results.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    normalized_rows = []
    for row in rows:
        normalized_rows.append({
            **row,
            "edge_tp": int(row["edge_tp"]),
            "edge_fp": int(row["edge_fp"]),
            "edge_fn": int(row["edge_fn"]),
            "group_accuracy": float(row["group_accuracy"]),
            "comp_neighbor_accuracy": float(row["comp_neighbor_accuracy"]),
            "stage_summary_raw": json.loads(row.get("stage_summary") or "{}"),
        })
    summary = {}
    for config_name in expected_configs:
        config_rows = [row for row in normalized_rows if row["config"] == config_name]
        tp = sum(row["edge_tp"] for row in config_rows)
        fp = sum(row["edge_fp"] for row in config_rows)
        fn = sum(row["edge_fn"] for row in config_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[config_name] = {
            "edge": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            },
            "macro_group_accuracy": float(np.mean([row["group_accuracy"] for row in config_rows])),
            "macro_component_neighbor_accuracy": float(np.mean([row["comp_neighbor_accuracy"] for row in config_rows])),
            "stage_summary": _aggregate_stage_summaries(config_rows),
            "failed_images": sum(bool(row.get("error")) for row in config_rows),
        }
    _write_json(output / "wiring_reliability_summary.json", summary)
    _write_json(output / "failures.json", failures)
    _write_json(output / "run_metadata.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": "wiring-reliability-merged",
        **reference,
        "image_count": len(image_configs),
        "expected_case_count": expected_count,
        "config_count": len(expected_configs),
        "failure_count": len(failures),
        "final_42_image_test_used": False,
        "shard_count": len(shard_dirs),
        "source_shards": [str(path.resolve()) for path in shard_dirs],
    })
    return output


def recover_complete_wiring_predictions(partial_dir, output_dir, benchmark_dir=BENCHMARK):
    """Rebuild a valid shard from images whose seven prediction files are complete."""
    partial_dir = Path(partial_dir)
    configs = build_wiring_reliability_configs()
    stems_by_config = {
        config_name: {
            path.stem for path in (partial_dir / "predictions" / config_name).glob("*.json")
        }
        for config_name in configs
    }
    complete_stems = set.intersection(*stems_by_config.values()) if stems_by_config else set()
    if not complete_stems:
        raise RuntimeError("partial directory contains no fully completed images")
    cases = {stem: (image, gt, det) for stem, image, gt, det in get_image_list(benchmark_dir)}
    unknown = sorted(complete_stems - set(cases))
    if unknown:
        raise RuntimeError(f"partial predictions are not benchmark cases: {unknown}")

    output = resolve_output_dir(output_dir)
    rows = []
    for stem in sorted(complete_stems):
        _, gt_path, det_path = cases[stem]
        gt_groups = parse_gt(gt_path)
        with open(det_path, encoding="utf-8") as handle:
            det_comps = json.load(handle)["components"]
        for config_name in configs:
            source = partial_dir / "predictions" / config_name / f"{stem}.json"
            result = json.loads(source.read_text(encoding="utf-8"))
            if result.get("evaluation") != "(LLM skipped for experiment)":
                raise RuntimeError(f"{stem}/{config_name}: prediction did not skip LLM")
            metrics = evaluate(result, gt_groups, det_comps)
            stage_summary = result.get("wiring_trace", {}).get("summary", {})
            rows.append({
                "image": stem, "config": config_name,
                "n_components": len(det_comps), **metrics,
                "stage_summary": json.dumps(stage_summary, ensure_ascii=False, sort_keys=True),
                "stage_summary_raw": stage_summary, "error": "",
            })
            destination = output / "predictions" / config_name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    columns = [
        "image", "config", "n_components", "n_gt_groups", "n_pred_groups",
        "edge_tp", "edge_fp", "edge_fn", "edge_precision", "edge_recall", "edge_f1",
        "port_correct_rate", "fp_rate", "fn_rate", "group_accuracy",
        "comp_neighbor_accuracy", "stage_summary", "error",
    ]
    with (output / "experiment_results.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for config_name in configs:
        config_rows = [row for row in rows if row["config"] == config_name]
        tp = sum(int(row["edge_tp"]) for row in config_rows)
        fp = sum(int(row["edge_fp"]) for row in config_rows)
        fn = sum(int(row["edge_fn"]) for row in config_rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[config_name] = {
            "edge": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            },
            "macro_group_accuracy": float(np.mean([row["group_accuracy"] for row in config_rows])),
            "macro_component_neighbor_accuracy": float(np.mean([row["comp_neighbor_accuracy"] for row in config_rows])),
            "stage_summary": _aggregate_stage_summaries(config_rows),
            "failed_images": 0,
        }
    _write_json(output / "wiring_reliability_summary.json", summary)
    _write_json(output / "failures.json", [])
    _write_json(output / "run_metadata.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": "wiring-reliability",
        "git_revision": get_git_revision(),
        "benchmark_dir": str(Path(benchmark_dir).resolve()),
        "image_count": len(complete_stems),
        "expected_case_count": len(complete_stems),
        "selected_images": sorted(complete_stems),
        "config_count": len(configs),
        "configs": configs,
        "ocr_model_sha256": _file_sha256(DEFAULT_CONFIG["ocr_model_path"]),
        "detector_model_sha256": _file_sha256("runs/detect/cghd_61cls/weights/best.pt"),
        "failure_count": 0,
        "final_42_image_test_used": False,
        "recovered_from_partial_predictions": True,
        "partial_source": str(partial_dir.resolve()),
    })
    return output


def main(output_dir=None):
    output_dir = resolve_output_dir(output_dir)
    images = get_image_list()
    ablation_configs = build_ablation_configs()
    configs = list(ablation_configs.keys())
    print(f"Output directory: {output_dir}")
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
            cfg = ablation_configs[cfg_name]
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
    csv_path = output_dir / "experiment_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["image", "config", "n_components", "n_gt_groups", "n_pred_groups",
                "port_correct_rate", "fp_rate", "fn_rate", "group_accuracy",
                "comp_neighbor_accuracy"]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nSaved {csv_path} ({len(results)} rows)")

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_revision": get_git_revision(),
            "image_count": len(images),
            "configs": ablation_configs,
        }, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved {metadata_path}")

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
    summary_plot_path = output_dir / "experiment_summary.png"
    fig.savefig(summary_plot_path, dpi=150)
    print(f"Saved {summary_plot_path}")

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
        impact_plot_path = output_dir / "ablation_impact.png"
        fig2.savefig(impact_plot_path, dpi=150)
        print(f"Saved {impact_plot_path}")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("general", "wiring-reliability"),
        default="general",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for this run's CSV, plots, and metadata.",
    )
    parser.add_argument(
        "--images",
        help="Comma-separated benchmark stems (wiring-reliability only).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Refuse the run unless this many GT-backed cases are scheduled.",
    )
    args = parser.parse_args()
    if args.suite == "wiring-reliability":
        selected = [item.strip() for item in args.images.split(",") if item.strip()] if args.images else None
        run_wiring_reliability_experiment(
            output_dir=args.output_dir,
            selected_images=selected,
            expected_count=args.expected_count,
        )
    else:
        if args.images or args.expected_count is not None:
            parser.error("--images and --expected-count require --suite wiring-reliability")
        main(args.output_dir)
