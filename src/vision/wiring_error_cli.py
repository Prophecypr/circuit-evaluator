"""Validated local runner for wiring-edge error attribution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

from .wiring_error_analysis import (
    FnEvidence,
    attribute_fn_edges,
    attribute_fp_edges,
    build_physical_edges,
    index_trace_events,
    skeleton_near_port,
)
from .wiring_topology import build_edge_inventory


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class AnalysisCase:
    stem: str
    image_path: Path
    gt_path: Path
    detection_path: Path
    prediction_path: Path
    trace_path: Path


def prepare_output(output_dir: str | Path, resume: bool = False) -> Path:
    """Create a dedicated output directory without overwriting user data."""
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Refusing to overwrite non-directory output: {output}")
        if any(output.iterdir()) and not resume:
            raise FileExistsError(
                f"Refusing to overwrite non-empty attribution output directory: {output}"
            )
    else:
        output.mkdir(parents=True)
    return output


def _stems(paths, suffix_to_remove="") -> set[str]:
    stems = set()
    for path in paths:
        stem = path.stem
        if suffix_to_remove and stem.endswith(suffix_to_remove):
            stem = stem[: -len(suffix_to_remove)]
        stems.add(stem)
    return stems


def _one_image(benchmark_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in benchmark_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem == stem
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{stem}: expected one image, found {len(matches)}")
    return matches[0]


def _require_file(path: Path, label: str, stem: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{stem}: missing {label}: {path}")
    return path


def validate_inputs(
    run_dir: str | Path,
    benchmark_dir: str | Path = "benchmark",
    expected_count: int = 50,
    config_name: str = "strict_jj",
) -> tuple[dict, tuple[AnalysisCase, ...]]:
    """Validate one complete cached config and join every file by stem."""
    run_dir = Path(run_dir)
    benchmark_dir = Path(benchmark_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    image_count = metadata.get("image_count")
    if image_count != expected_count:
        raise RuntimeError(f"expected {expected_count} images, found {image_count}")
    if metadata.get("failure_count") != 0:
        raise RuntimeError("input run contains failures")
    selected_config = metadata.get("configs", {}).get(config_name)
    if not isinstance(selected_config, dict):
        raise RuntimeError(f"input run has no {config_name} configuration")
    if selected_config.get("skip_llm") is not True:
        raise RuntimeError(f"{config_name} must set skip_llm=true")
    if selected_config.get("skip_ocr") is not True:
        raise RuntimeError(f"{config_name} must set skip_ocr=true")
    if selected_config.get("use_strict_jj") is not True:
        raise RuntimeError(f"{config_name} must set use_strict_jj=true")
    if metadata.get("final_42_image_test_used") is not False:
        raise RuntimeError("sealed final 42-image test set was used")

    prediction_dir = run_dir / "predictions" / config_name
    trace_dir = run_dir / f"wiring_traces_{config_name}"
    gt_dir = benchmark_dir / "result"
    prediction_stems = _stems(prediction_dir.glob("*.json"))
    trace_stems = _stems(trace_dir.glob("*.json"))
    gt_stems = _stems(gt_dir.glob("*_gt.txt"), "_gt")
    if len(prediction_stems) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} {config_name} predictions, found {len(prediction_stems)}"
        )
    extra_traces = sorted(trace_stems - prediction_stems)
    if extra_traces:
        raise RuntimeError(f"trace stems do not match predictions: {extra_traces}")
    if gt_stems != prediction_stems:
        missing_gt = sorted(prediction_stems - gt_stems)
        extra_gt = sorted(gt_stems - prediction_stems)
        raise RuntimeError(f"GT stems do not match predictions: missing={missing_gt}, extra={extra_gt}")

    cases = []
    for stem in sorted(prediction_stems):
        prediction_path = _require_file(
            prediction_dir / f"{stem}.json", "prediction", stem
        )
        separate_trace_path = trace_dir / f"{stem}.json"
        if separate_trace_path.is_file():
            trace_path = separate_trace_path
        else:
            prediction_payload = _read_json(prediction_path)
            if not isinstance(prediction_payload.get("wiring_trace"), dict):
                raise FileNotFoundError(f"missing trace for {stem}")
            trace_path = prediction_path
        fixed_path = benchmark_dir / "fixed" / f"{stem}.json"
        detection_path = (
            fixed_path
            if fixed_path.is_file()
            else benchmark_dir / "detections" / f"{stem}.json"
        )
        cases.append(
            AnalysisCase(
                stem=stem,
                image_path=_one_image(benchmark_dir, stem),
                gt_path=_require_file(gt_dir / f"{stem}_gt.txt", "GT", stem),
                detection_path=_require_file(detection_path, "detection", stem),
                prediction_path=prediction_path,
                trace_path=trace_path,
            )
        )
    return metadata, tuple(cases)


def _parse_gt(path: Path) -> list[list[tuple[str, str]]]:
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        entries = []
        for token in line.split(":", 1)[1].split(","):
            token = token.strip()
            if "." in token:
                entries.append(tuple(token.rsplit(".", 1)))
        if len(entries) >= 2:
            groups.append(entries)
    return groups


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def reconcile_counts(image_rows, error_rows, expected_counts):
    actual = {
        "tp": sum(int(row["edge_tp"]) for row in image_rows),
        "fp": sum(row["error_type"] == "FP" for row in error_rows),
        "fn": sum(row["error_type"] == "FN" for row in error_rows),
    }
    if actual != expected_counts:
        raise RuntimeError(
            f"edge reconciliation failed: expected {expected_counts}, got {actual}"
        )
    return actual


def _decode_image(path: Path):
    import cv2
    import numpy as np

    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    return image


def _analysis_skeleton(image, components):
    import cv2

    from .unified_pipeline import _extract_component_masked_skeleton

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    im_scale = math.hypot(width, height) / 2000.0
    return _extract_component_masked_skeleton(
        gray,
        components,
        im_scale=im_scale,
    )


def _port_event_evidence(events) -> tuple[bool, bool, str | None]:
    accepted = any(event.get("accepted") is True for event in events)
    reasons = [str(event.get("reason", "")) for event in events]
    has_no_path = "no_skeleton_path" in reasons
    actionable_rejections = {
        "ambiguous_crossing",
        "crosses_component",
        "would_short_component",
    }
    rejection = next(
        (reason for reason in reasons if reason in actionable_rejections),
        None,
    )
    trace_reached = accepted or not has_no_path
    return bool(events), trace_reached, rejection


def _build_fn_evidence(inventory, detections, skeleton, trace_payload):
    detection_index_by_designator = {
        str(component.get("designator", "")): index
        for index, component in enumerate(detections)
    }
    matched_detection_indices = set(inventory.component_matches.values())
    pipeline_key_by_port = {
        port_id: key for key, port_id in inventory.pipeline_port_ids.items()
    }
    trace_index = index_trace_events(trace_payload)
    all_gt_ports = set().union(*inventory.gt_groups) if inventory.gt_groups else set()
    evidence_by_port = {}
    for port_id in sorted(all_gt_ports):
        designator = port_id.rsplit(".", 1)[0]
        detection_index = detection_index_by_designator.get(designator)
        component_matched = (
            detection_index is not None
            and detection_index in matched_detection_indices
        )
        port_mapped = port_id not in inventory.unmapped_gt_ports
        if not component_matched:
            evidence_by_port[port_id] = FnEvidence(component_matched=False)
            continue
        if not port_mapped:
            evidence_by_port[port_id] = FnEvidence(
                component_matched=True,
                port_mapped=False,
            )
            continue
        pipeline_key = pipeline_key_by_port.get(port_id)
        events = trace_index.get(pipeline_key, []) if pipeline_key else []
        candidate_generated, trace_reached, rejection = _port_event_evidence(events)
        evidence_by_port[port_id] = FnEvidence(
            component_matched=True,
            port_mapped=True,
            skeleton_near_port=skeleton_near_port(
                skeleton,
                inventory.port_points[port_id],
            ),
            trace_reached_network=trace_reached,
            candidate_generated=candidate_generated,
            rejection_reason=rejection,
        )
    return evidence_by_port


def _network_id(groups, port_id):
    return next(
        (f"N{index + 1}" for index, group in enumerate(groups) if port_id in group),
        "",
    )


def _error_row(stem, attribution, inventory):
    first, second = attribution.edge
    return {
        "image": stem,
        "error_type": attribution.error_type,
        "port_a": first,
        "port_b": second,
        "category": attribution.category,
        "secondary_reason": attribution.secondary_reason,
        "is_root": attribution.is_root,
        "root_event_id": attribution.root_event_id,
        "pred_network": _network_id(inventory.pred_groups, first),
        "gt_network_a": _network_id(inventory.gt_groups, first),
        "gt_network_b": _network_id(inventory.gt_groups, second),
    }


def _detection_port_points(detections):
    points = {}
    for component in detections:
        labels = component.get("labels", [])
        for port_index, point in enumerate(component.get("ports", [])):
            label = labels[port_index] if port_index < len(labels) else "?"
            points[f"{component.get('designator', '')}.{label}"] = tuple(map(int, point))
    return points


def _analyze_case(case: AnalysisCase):
    prediction = _read_json(case.prediction_path)
    trace = _read_json(case.trace_path)
    if case.trace_path == case.prediction_path:
        trace = trace["wiring_trace"]
    detections = _read_json(case.detection_path).get("components", [])
    gt_groups = _parse_gt(case.gt_path)
    inventory = build_edge_inventory(prediction, gt_groups, detections)
    image = _decode_image(case.image_path)
    skeleton = _analysis_skeleton(image, prediction.get("components", []))
    fn_evidence = _build_fn_evidence(inventory, detections, skeleton, trace)
    fn_rows = attribute_fn_edges(inventory.fn_edges, fn_evidence)
    physical_edges = build_physical_edges(trace, inventory.pipeline_port_ids)
    fp_rows = attribute_fp_edges(
        inventory.gt_groups,
        inventory.pred_groups,
        physical_edges,
    )
    error_rows = [
        _error_row(case.stem, attribution, inventory)
        for attribution in (*fp_rows, *fn_rows)
    ]
    tp, fp, fn = len(inventory.tp_edges), len(inventory.fp_edges), len(inventory.fn_edges)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    image_row = {
        "image": case.stem,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "root_errors": sum(row["is_root"] for row in error_rows),
        "dominant_category": Counter(row["category"] for row in error_rows).most_common(1)[0][0]
        if error_rows
        else "",
    }
    port_points = _detection_port_points(detections)
    port_points.update(inventory.port_points)
    return {
        "case": case,
        "image": image,
        "inventory": inventory,
        "error_rows": error_rows,
        "image_row": image_row,
        "port_points": port_points,
    }


def _category_summary(error_rows):
    by_category = defaultdict(list)
    for row in error_rows:
        by_category[row["category"]].append(row)
    total = len(error_rows)
    return {
        category: {
            "edge_count": len(rows),
            "root_count": sum(bool(row["is_root"]) for row in rows),
            "image_count": len({row["image"] for row in rows}),
            "percentage": len(rows) / total if total else 0.0,
        }
        for category, rows in sorted(by_category.items())
    }


def _atomic_text(path: Path, text: str, encoding="utf-8"):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def _write_csv(path: Path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue(), encoding="utf-8-sig")


def _select_render_error_rows(error_rows):
    """Keep root FP and one representative FN per GT network/category."""
    selected = []
    seen_fn = set()
    for row in sorted(
        error_rows,
        key=lambda item: (
            item["error_type"],
            item["category"],
            item.get("gt_network_a", ""),
            item["port_a"],
            item["port_b"],
        ),
    ):
        if row["error_type"] == "FP":
            if row["is_root"]:
                selected.append(row)
            continue
        key = (row.get("gt_network_a", ""), row["category"])
        if key not in seen_fn:
            seen_fn.add(key)
            selected.append(row)
    return tuple(selected)


def _render_diagnostic(analysis, output_path: Path):
    import cv2

    image = analysis["image"].copy()
    inventory = analysis["inventory"]
    points = analysis["port_points"]
    colors = {
        "TP": (0, 180, 0),
        "FP": (0, 165, 255),
        "FN": (0, 0, 255),
    }

    def draw_edge(edge, color, thickness=2):
        first, second = points.get(edge[0]), points.get(edge[1])
        if first is not None and second is not None:
            cv2.line(image, first, second, color, thickness, cv2.LINE_AA)
            return first, second
        return None, None

    for edge in sorted(inventory.tp_edges):
        draw_edge(edge, colors["TP"], 2)
    for row in _select_render_error_rows(analysis["error_rows"]):
        edge = (row["port_a"], row["port_b"])
        first, second = draw_edge(edge, colors[row["error_type"]], 2)
        if row["is_root"] and first is not None and second is not None:
            midpoint = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
            cv2.circle(image, midpoint, 8, (180, 0, 180), 3, cv2.LINE_AA)
            cv2.putText(
                image,
                row["category"],
                (midpoint[0] + 10, midpoint[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 0, 180),
                1,
                cv2.LINE_AA,
            )
    legend = [("TP", colors["TP"]), ("FP", colors["FP"]), ("FN", colors["FN"]), ("ROOT", (180, 0, 180))]
    for index, (label, color) in enumerate(legend):
        y = 24 + index * 24
        cv2.line(image, (12, y), (42, y), color, 3)
        cv2.putText(image, label, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    error_counts = Counter(row["error_type"] for row in analysis["error_rows"])
    cascade_count = sum(
        row["category"] == "cascade_fp" for row in analysis["error_rows"]
    )
    summary = (
        f"metric errors: FP={error_counts['FP']} FN={error_counts['FN']} "
        f"cascade_FP={cascade_count}"
    )
    cv2.putText(
        image,
        summary,
        (12, 24 + len(legend) * 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix or ".jpg", image)
    if not success:
        raise RuntimeError(f"cannot encode diagnostic image: {output_path}")
    encoded.tofile(str(output_path))


def _report_markdown(image_rows, category_summary, actual_counts):
    actionable = [
        (category, values)
        for category, values in category_summary.items()
        if category != "cascade_fp"
    ]
    actionable.sort(
        key=lambda item: (-item[1]["root_count"], -item[1]["edge_count"], item[0])
    )
    dominant = actionable[0][0] if actionable else "none"
    lines = [
        "# Wiring Edge 错误归因报告",
        "",
        f"- 图像数：{len(image_rows)}",
        f"- TP/FP/FN：{actual_counts['tp']}/{actual_counts['fp']}/{actual_counts['fn']}",
        "- LLM：未使用",
        "- OCR：未使用",
        "- 最终42张测试图：未使用",
        f"- 首要根因类别：`{dominant}`",
        "",
        "## 类别汇总",
        "",
        "| 类别 | 边数 | 根因数 | 涉及图像 | 占全部错误 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in category_summary.items():
        lines.append(
            f"| {category} | {values['edge_count']} | {values['root_count']} | "
            f"{values['image_count']} | {values['percentage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            f"下一轮只针对 `{dominant}` 设计最小算法改动；完整50张F1提升且Precision下降不超过1个百分点后，才允许启封最终42张。",
        ]
    )
    return "\n".join(lines) + "\n"


def _provenance(run_dir: Path, metadata_path: Path, cases):
    return {
        "source_run": str(run_dir.resolve()),
        "source_metadata_sha256": sha256_file(metadata_path),
        "cases": [
            {
                "stem": case.stem,
                "image_sha256": sha256_file(case.image_path),
                "gt_sha256": sha256_file(case.gt_path),
                "detection_sha256": sha256_file(case.detection_path),
                "prediction_sha256": sha256_file(case.prediction_path),
                "trace_sha256": sha256_file(case.trace_path),
            }
            for case in cases
        ],
    }


def run_analysis(
    run_dir,
    benchmark_dir,
    output_dir,
    *,
    expected_count=50,
    expected_counts=None,
    worst_count=10,
    resume=False,
    config_name="strict_jj",
):
    """Analyze cached wiring errors and write a fully reconciled report."""
    expected_counts = expected_counts or {"tp": 305, "fp": 537, "fn": 1002}
    run_dir = Path(run_dir)
    _, cases = validate_inputs(
        run_dir,
        benchmark_dir,
        expected_count,
        config_name=config_name,
    )
    provenance = _provenance(run_dir, run_dir / "run_metadata.json", cases)
    output = Path(output_dir)
    if resume and output.is_dir() and any(output.iterdir()):
        metadata_path = output / "run_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError("resume output has no run_metadata.json")
        existing = _read_json(metadata_path)
        if existing.get("provenance") != provenance:
            raise RuntimeError("resume provenance does not match current inputs")
        required_files = {
            "edge_errors.csv",
            "image_summary.csv",
            "category_summary.json",
            "wiring_error_report.md",
            "run_metadata.json",
        }
        missing_files = sorted(
            name for name in required_files if not (output / name).is_file()
        )
        rendered_count = len(list((output / "annotated_worst10").glob("*.jpg")))
        expected_rendered = min(worst_count, expected_count)
        if missing_files or rendered_count != expected_rendered:
            raise RuntimeError(
                "resume output is incomplete: "
                f"missing={missing_files}, rendered={rendered_count}/{expected_rendered}"
            )
        return output
    output = prepare_output(output, resume=False)

    analyses = [_analyze_case(case) for case in cases]
    image_rows = [analysis["image_row"] for analysis in analyses]
    error_rows = [row for analysis in analyses for row in analysis["error_rows"]]
    actual_counts = reconcile_counts(image_rows, error_rows, expected_counts)
    category_summary = _category_summary(error_rows)

    _write_csv(
        output / "edge_errors.csv",
        error_rows,
        [
            "image", "error_type", "port_a", "port_b", "category",
            "secondary_reason", "is_root", "root_event_id", "pred_network",
            "gt_network_a", "gt_network_b",
        ],
    )
    _write_csv(
        output / "image_summary.csv",
        image_rows,
        [
            "image", "edge_tp", "edge_fp", "edge_fn", "edge_precision",
            "edge_recall", "edge_f1", "root_errors", "dominant_category",
        ],
    )
    _atomic_text(
        output / "category_summary.json",
        json.dumps(category_summary, ensure_ascii=False, indent=2),
    )
    _atomic_text(
        output / "wiring_error_report.md",
        _report_markdown(image_rows, category_summary, actual_counts),
    )

    worst_stems = {
        row["image"]
        for row in sorted(
            image_rows,
            key=lambda row: (
                float(row["edge_f1"]),
                -(int(row["edge_fp"]) + int(row["edge_fn"])),
                row["image"],
            ),
        )[: min(worst_count, len(image_rows))]
    }
    for analysis in analyses:
        if analysis["case"].stem in worst_stems:
            _render_diagnostic(
                analysis,
                output / "annotated_worst10" / f"{analysis['case'].stem}.jpg",
            )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "config_name": config_name,
        "image_count": len(cases),
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "category_summary": category_summary,
        "provenance": provenance,
        "llm_used": False,
        "ocr_used": False,
        "final_42_image_test_used": False,
    }
    _atomic_text(
        output / "run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute cached wiring FP/FN root causes"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/wiring_reliability_full_20260810_merged"),
    )
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--config", default="strict_jj")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/wiring_error_attribution_20260811"),
    )
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--expected-tp", type=int, default=305)
    parser.add_argument("--expected-fp", type=int, default=537)
    parser.add_argument("--expected-fn", type=int, default=1002)
    parser.add_argument("--worst-count", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    expected_counts = {
        "tp": args.expected_tp,
        "fp": args.expected_fp,
        "fn": args.expected_fn,
    }
    output = run_analysis(
        args.run_dir,
        args.benchmark_dir,
        args.output_dir,
        expected_count=args.expected_count,
        expected_counts=expected_counts,
        worst_count=args.worst_count,
        resume=args.resume,
        config_name=args.config,
    )
    metadata = _read_json(output / "run_metadata.json")
    actual = metadata["actual_counts"]
    print(f"images={metadata['image_count']} failures=0")
    print(
        f"TP={actual['tp']} FP={actual['fp']} FN={actual['fn']} reconciled=true"
    )
    print("LLM=false OCR=false final42=false")
    print(f"output={output.resolve()}")
    return output


if __name__ == "__main__":
    main()
