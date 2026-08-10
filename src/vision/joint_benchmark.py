"""Pure evaluation helpers for the LLM-free joint circuit benchmark."""

from __future__ import annotations

import math
import re
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ocr_v2.metrics import edit_distance


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_VALUE_RE = re.compile(
    rf"^(?:{_NUMBER})(?:kΩ|MΩ|mΩ|Ω|μF|mF|nF|pF|F|μA|mA|A|μH|mH|nH|H|mV|kV|V|Hz)?$"
)
_PARSE_VALUE_RE = re.compile(rf"^(?P<number>{_NUMBER})(?P<unit>.*)$")


def _canonical_class(label: str) -> str:
    token = re.sub(r"[\s_-]+", " ", str(label).strip()).casefold()
    aliases = {
        "phototransistor": "photo transistor",
        "fet": "mosfet p",
        "variable capacitor": "capacitor",
    }
    return aliases.get(token, token)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _greedy_box_matches(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_threshold: float,
    *,
    class_aware: bool = True,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, truth in enumerate(ground_truth):
            if prediction.get("image") != truth.get("image"):
                continue
            if class_aware and _canonical_class(prediction["label"]) != _canonical_class(truth["label"]):
                continue
            overlap = bbox_iou(prediction["bbox"], truth["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_index, gt_index))

    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in used_predictions or gt_index in used_truth:
            continue
        used_predictions.add(pred_index)
        used_truth.add(gt_index)
        matches.append((pred_index, gt_index, overlap))

    unmatched_predictions = [index for index in range(len(predictions)) if index not in used_predictions]
    unmatched_truth = [index for index in range(len(ground_truth)) if index not in used_truth]
    return matches, unmatched_predictions, unmatched_truth


def _safe_prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def average_precision_at_iou(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> float:
    """Return ranked, class-aware all-point interpolated AP at one IoU."""
    if not ground_truth:
        return 0.0
    ranked = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    truth_by_image: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, truth in enumerate(ground_truth):
        truth_by_image[str(truth.get("image", ""))].append((index, truth))
    matched_truth: set[int] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for prediction in ranked:
        best_iou = 0.0
        best_index: int | None = None
        for gt_index, truth in truth_by_image[str(prediction.get("image", ""))]:
            if gt_index in matched_truth:
                continue
            if _canonical_class(prediction["label"]) != _canonical_class(truth["label"]):
                continue
            overlap = bbox_iou(prediction["bbox"], truth["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        is_true = best_index is not None and best_iou >= iou_threshold
        tp_flags.append(int(is_true))
        fp_flags.append(int(not is_true))
        if is_true:
            matched_truth.add(best_index)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        cumulative_tp += tp_flag
        cumulative_fp += fp_flag
        recalls.append(cumulative_tp / len(ground_truth))
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))

    recall_curve = [0.0, *recalls, 1.0]
    precision_curve = [0.0, *precisions, 0.0]
    for index in range(len(precision_curve) - 2, -1, -1):
        precision_curve[index] = max(precision_curve[index], precision_curve[index + 1])
    return sum(
        (recall_curve[index] - recall_curve[index - 1]) * precision_curve[index]
        for index in range(1, len(recall_curve))
        if recall_curve[index] != recall_curve[index - 1]
    )


def detection_metrics(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    matches, unmatched_predictions, unmatched_truth = _greedy_box_matches(
        predictions, ground_truth, iou_threshold, class_aware=True
    )
    result: dict[str, Any] = _safe_prf(
        len(matches), len(unmatched_predictions), len(unmatched_truth)
    )
    labels = sorted({_canonical_class(item["label"]) for item in ground_truth})
    per_class: dict[str, dict[str, Any]] = {}
    ap_values: list[float] = []
    for label in labels:
        class_gt = [item for item in ground_truth if _canonical_class(item["label"]) == label]
        class_pred = [item for item in predictions if _canonical_class(item["label"]) == label]
        class_matches, class_fp, class_fn = _greedy_box_matches(
            class_pred, class_gt, iou_threshold, class_aware=True
        )
        class_result = _safe_prf(len(class_matches), len(class_fp), len(class_fn))
        class_result["ap50"] = average_precision_at_iou(class_pred, class_gt, iou_threshold)
        class_result["gt_count"] = len(class_gt)
        class_result["prediction_count"] = len(class_pred)
        per_class[label] = class_result
        ap_values.append(float(class_result["ap50"]))
    result["iou_threshold"] = iou_threshold
    result["ap50"] = average_precision_at_iou(predictions, ground_truth, iou_threshold)
    result["map50"] = sum(ap_values) / len(ap_values) if ap_values else 0.0
    result["per_class"] = per_class
    return result


def point_metrics(
    predictions: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
    radius: float,
) -> dict[str, float | int]:
    matches, unmatched_predictions, unmatched_truth = _point_matches(
        predictions, ground_truth, radius
    )
    return _safe_prf(len(matches), len(unmatched_predictions), len(unmatched_truth))


def _point_matches(
    predictions: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
    radius: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, (px, py) in enumerate(predictions):
        for gt_index, (gx, gy) in enumerate(ground_truth):
            distance = math.hypot(float(px) - float(gx), float(py) - float(gy))
            if distance <= radius:
                candidates.append((distance, pred_index, gt_index))
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_predictions or gt_index in used_truth:
            continue
        used_predictions.add(pred_index)
        used_truth.add(gt_index)
        matches.append((pred_index, gt_index, distance))
    return (
        matches,
        [index for index in range(len(predictions)) if index not in used_predictions],
        [index for index in range(len(ground_truth)) if index not in used_truth],
    )


def normalize_benchmark_value(text: str) -> str:
    value = "".join(str(text).strip().split())
    value = value.replace("µ", "μ").replace("u", "μ")
    value = re.sub(r"(?i)ohms?", "Ω", value)
    value = re.sub(r"(?i)meg\.?$", "MΩ", value)
    value = re.sub(r"(?<=\d)K(?=Ω|$)", "k", value)
    if re.fullmatch(rf"{_NUMBER}[kM]", value):
        value += "Ω"
    return value


def is_electrical_value(text: str) -> bool:
    return _VALUE_RE.fullmatch(normalize_benchmark_value(text)) is not None


def parse_cghd_xml(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    parsed: dict[str, list[dict[str, Any]]] = {
        "junctions": [], "terminals": [], "texts": []
    }
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        bbox = [int(float(box.findtext(key, "0"))) for key in ("xmin", "ymin", "xmax", "ymax")]
        if name == "junction":
            parsed["junctions"].append({"label": "junction", "bbox": bbox})
        elif name == "terminal":
            parsed["terminals"].append({"label": "terminal", "bbox": bbox})
        elif name == "text":
            parsed["texts"].append(
                {"label": "text", "bbox": bbox, "text": obj.findtext("text") or ""}
            )
    return parsed


def _ocr_pair_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {
            "sample_count": 0,
            "raw_exact": 0.0,
            "normalized_exact": 0.0,
            "raw_cer": 0.0,
            "normalized_cer": 0.0,
            "numeric_exact": 0.0,
            "unit_labeled_count": 0,
            "unit_exact": 0.0,
        }
    raw_exact = sum(gt == prediction for gt, prediction in pairs)
    normalized_pairs = [
        (normalize_benchmark_value(gt), normalize_benchmark_value(prediction))
        for gt, prediction in pairs
    ]
    raw_edits = sum(edit_distance(gt, prediction) for gt, prediction in pairs)
    normalized_edits = sum(edit_distance(gt, prediction) for gt, prediction in normalized_pairs)
    numeric_correct = 0
    unit_correct = 0
    unit_count = 0
    for ground_truth, prediction in normalized_pairs:
        gt_match = _PARSE_VALUE_RE.fullmatch(ground_truth)
        pred_match = _PARSE_VALUE_RE.fullmatch(prediction)
        gt_number = gt_match.group("number") if gt_match else ground_truth
        gt_unit = gt_match.group("unit") if gt_match else ""
        pred_number = pred_match.group("number") if pred_match else prediction
        pred_unit = pred_match.group("unit") if pred_match else ""
        numeric_correct += int(gt_number == pred_number)
        if gt_unit:
            unit_count += 1
            unit_correct += int(gt_unit == pred_unit)
    count = len(pairs)
    return {
        "sample_count": count,
        "raw_exact": raw_exact / count,
        "normalized_exact": sum(gt == prediction for gt, prediction in normalized_pairs) / count,
        "raw_cer": raw_edits / max(1, sum(len(gt) for gt, _ in pairs)),
        "normalized_cer": normalized_edits / max(1, sum(len(gt) for gt, _ in normalized_pairs)),
        "numeric_exact": numeric_correct / count,
        "unit_labeled_count": unit_count,
        "unit_exact": unit_correct / unit_count if unit_count else 0.0,
    }


def ocr_metrics_from_detections(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches, _, _ = _greedy_box_matches(
        predictions, ground_truth, iou_threshold, class_aware=False
    )
    prediction_for_truth = {gt_index: predictions[pred_index] for pred_index, gt_index, _ in matches}
    rows: list[dict[str, Any]] = []
    all_pairs: list[tuple[str, str]] = []
    value_pairs: list[tuple[str, str]] = []
    for truth_index, truth in enumerate(ground_truth):
        prediction = prediction_for_truth.get(truth_index, {})
        gt_text = str(truth.get("text", ""))
        predicted_text = str(prediction.get("text", ""))
        is_value = is_electrical_value(gt_text)
        all_pairs.append((gt_text, predicted_text))
        if is_value:
            value_pairs.append((gt_text, predicted_text))
        rows.append(
            {
                "image": truth.get("image", ""),
                "ground_truth": gt_text,
                "prediction": predicted_text,
                "normalized_ground_truth": normalize_benchmark_value(gt_text),
                "normalized_prediction": normalize_benchmark_value(predicted_text),
                "is_electrical_value": is_value,
                "matched_detection": bool(prediction),
            }
        )
    return {
        "all_text": _ocr_pair_metrics(all_pairs),
        "value_text": _ocr_pair_metrics(value_pairs),
    }, rows


def port_metrics(
    predicted_components: Sequence[dict[str, Any]],
    gt_components: Sequence[dict[str, Any]],
    radius: float,
    component_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    predicted_boxes = [
        {"image": "image", "label": component["name"], "bbox": component["xyxy"]}
        for component in predicted_components
    ]
    gt_boxes = [
        {"image": "image", "label": component["name"], "bbox": component["xyxy"]}
        for component in gt_components
    ]
    component_matches, unmatched_pred_components, unmatched_gt_components = _greedy_box_matches(
        predicted_boxes, gt_boxes, component_iou_threshold, class_aware=True
    )
    tp = 0
    fp = sum(
        len(predicted_components[index].get("ports", []))
        for index in unmatched_pred_components
    )
    fn = sum(len(gt_components[index].get("ports", [])) for index in unmatched_gt_components)
    label_correct = 0
    label_compared = 0
    for pred_index, gt_index, _ in component_matches:
        predicted = predicted_components[pred_index]
        truth = gt_components[gt_index]
        predicted_ports = list(predicted.get("ports", []))
        gt_ports = list(truth.get("ports", []))
        port_matches, unmatched_pred_ports, unmatched_gt_ports = _point_matches(
            predicted_ports, gt_ports, radius
        )
        tp += len(port_matches)
        fp += len(unmatched_pred_ports)
        fn += len(unmatched_gt_ports)
        try:
            from .unified_pipeline import PORT_LABELS

            predicted_labels = predicted.get("labels") or PORT_LABELS.get(predicted["name"], [])
        except ImportError:
            predicted_labels = predicted.get("labels", [])
        gt_labels = truth.get("labels", [])
        for pred_port_index, gt_port_index, _ in port_matches:
            if gt_port_index >= len(gt_labels):
                continue
            label_compared += 1
            predicted_label = (
                predicted_labels[pred_port_index]
                if pred_port_index < len(predicted_labels)
                else None
            )
            label_correct += int(str(predicted_label) == str(gt_labels[gt_port_index]))
    result: dict[str, Any] = _safe_prf(tp, fp, fn)
    result["label_correct"] = label_correct
    result["label_compared"] = label_compared
    result["label_accuracy"] = label_correct / label_compared if label_compared else 0.0
    return result


def attach_image(records: Iterable[dict[str, Any]], image: str) -> list[dict[str, Any]]:
    return [{**record, "image": image} for record in records]


def _sha256(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_provenance(
    cases: Sequence[dict[str, Path]],
    ocr_model_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "ocr_model_path": str(Path(ocr_model_path).resolve()),
        "ocr_model_sha256": _sha256(ocr_model_path),
        "config": config,
        "cases": [
            {
                "stem": str(case["stem"]),
                "image_sha256": _sha256(case["image"]),
                "fixed_sha256": _sha256(case["fixed"]),
                "wiring_sha256": _sha256(case["wiring"]),
                "xml_sha256": _sha256(case["xml"]),
            }
            for case in cases
        ],
    }


def _find_cases(benchmark_dir: Path, cghd_root: Path) -> list[dict[str, Path]]:
    xml_index = {path.stem: path for path in cghd_root.rglob("*.xml")}
    cases: list[dict[str, Path]] = []
    suffixes = {".jpg", ".jpeg", ".png"}
    for gt_path in sorted((benchmark_dir / "result").glob("*_gt.txt")):
        stem = gt_path.stem.removesuffix("_gt")
        images = [
            path
            for path in benchmark_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in suffixes
            and path.stem == stem
        ]
        if len(images) != 1:
            raise FileNotFoundError(f"{stem}: expected one benchmark image, found {len(images)}")
        fixed_path = benchmark_dir / "fixed" / f"{stem}.json"
        if not fixed_path.is_file():
            raise FileNotFoundError(f"{stem}: missing fixed component/port JSON")
        xml_path = xml_index.get(stem)
        if xml_path is None:
            raise FileNotFoundError(f"{stem}: missing CGHD XML ground truth")
        cases.append(
            {
                "stem": Path(stem),
                "image": images[0],
                "fixed": fixed_path,
                "wiring": gt_path,
                "xml": xml_path,
            }
        )
    if not cases:
        raise FileNotFoundError(f"no GT-backed images found in {benchmark_dir}")
    return cases


def _sum_prf(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row.get("tp", 0)) for row in rows)
    fp = sum(int(row.get("fp", 0)) for row in rows)
    fn = sum(int(row.get("fn", 0)) for row in rows)
    return _safe_prf(tp, fp, fn)


def _render_prediction(image_path: Path, result: dict[str, Any], output_path: Path) -> None:
    import cv2
    import numpy as np

    image_bytes = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot render unreadable image: {image_path}")
    for component in result.get("components", []):
        x1, y1, x2, y2 = map(int, component["xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        label = str(component.get("name", "component"))
        if component.get("value"):
            label += f"={component['value']}"
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 130, 0), 1)
        for px, py in component.get("ports", []):
            cv2.circle(image, (int(px), int(py)), 5, (0, 0, 255), -1)
    for jx, jy in result.get("junctions", []):
        cv2.circle(image, (int(jx), int(jy)), 5, (0, 255, 255), -1)
    for detection in result.get("raw_junction_detections", []):
        x1, y1, x2, y2 = map(int, detection["bbox"])
        color = (0, 165, 255) if detection.get("label") == "junction" else (180, 0, 180)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    for text in result.get("raw_text_detections", []):
        x1, y1, x2, y2 = map(int, text["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 120, 0), 1)
        cv2.putText(
            image,
            str(text.get("text", "")),
            (x1, min(image.shape[0] - 4, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 80, 0),
            1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"failed to write rendered image: {output_path}")
    output_path.write_bytes(encoded.tobytes())


def run_joint_benchmark(
    *,
    benchmark_dir: str | Path,
    cghd_root: str | Path,
    output_dir: str | Path,
    ocr_model_path: str | Path,
    process_fn=None,
    render: bool = True,
    resume: bool = False,
    expected_case_count: int | None = None,
    prediction_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete visual pipeline and write a self-contained benchmark record."""
    benchmark = Path(benchmark_dir)
    cghd = Path(cghd_root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    cases = _find_cases(benchmark, cghd)
    if expected_case_count is not None and len(cases) != expected_case_count:
        raise RuntimeError(
            f"expected {expected_case_count} benchmark cases, found {len(cases)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=resume)
    if render:
        (output / "annotated_images").mkdir(exist_ok=resume)

    if process_fn is None:
        from .unified_pipeline import process_image as process_fn
    from run_experiments import evaluate as evaluate_wiring
    from run_experiments import parse_gt

    config = {
        "skip_llm": True,
        "save_artifacts": False,
        "ocr_model_path": str(ocr_model_path),
    }
    imported_predictions: dict[str, Path] = {}
    cache_record: dict[str, Any] | None = None
    if prediction_cache_dir is not None:
        cache_root = Path(prediction_cache_dir)
        cache_metadata_path = cache_root / "run_metadata.json"
        if not cache_metadata_path.is_file():
            raise RuntimeError("prediction cache provenance missing run_metadata.json")
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if cache_metadata.get("ocr_model_sha256") != _sha256(ocr_model_path):
            raise RuntimeError("prediction cache provenance has a different OCR model")
        cache_config = cache_metadata.get("config", {})
        if cache_config.get("skip_llm") is not True or cache_config.get("save_artifacts") is not False:
            raise RuntimeError("prediction cache provenance has incompatible pipeline config")
        if int(cache_metadata.get("image_count", -1)) != len(cases):
            raise RuntimeError("prediction cache provenance has a different image count")
        expected_stems = {str(case["stem"]) for case in cases}
        cache_files = {
            path.stem: path for path in (cache_root / "predictions").glob("*.json")
        }
        if set(cache_files) != expected_stems:
            raise RuntimeError("prediction cache files do not exactly match benchmark cases")
        imported_predictions = cache_files
        cache_record = {
            "source": str(cache_root.resolve()),
            "source_metadata_sha256": _sha256(cache_metadata_path),
            "prediction_sha256": {
                stem: _sha256(path) for stem, path in sorted(cache_files.items())
            },
        }
    provenance = _build_provenance(cases, ocr_model_path, config)
    if cache_record is not None:
        provenance["prediction_cache"] = cache_record
    metadata_path = output / "run_metadata.json"
    if resume:
        if not metadata_path.is_file():
            raise RuntimeError(
                "resume provenance missing: output is not a verified joint benchmark run"
            )
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior_metadata.get("provenance") != provenance:
            raise RuntimeError("resume provenance mismatch; cached predictions cannot be reused")
        metadata = prior_metadata
        metadata["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "running"
    else:
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "benchmark_dir": str(benchmark.resolve()),
            "cghd_root": str(cghd.resolve()),
            "ocr_model_path": str(Path(ocr_model_path).resolve()),
            "ocr_model_sha256": _sha256(ocr_model_path),
            "config": config,
            "image_count": len(cases),
            "xml_count": len(cases),
            "final_42_image_test_used": False,
            "provenance": provenance,
            "status": "running",
        }
    _write_json(metadata_path, metadata)
    all_component_predictions: list[dict[str, Any]] = []
    all_component_gt: list[dict[str, Any]] = []
    all_junction_predictions: list[dict[str, Any]] = []
    all_junction_gt: list[dict[str, Any]] = []
    all_text_predictions: list[dict[str, Any]] = []
    all_text_gt: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    wiring_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    render_failures: list[dict[str, str]] = []
    strict_successes = 0

    for case_index, case in enumerate(cases, start=1):
        stem = str(case["stem"])
        fixed = json.loads(case["fixed"].read_text(encoding="utf-8"))
        gt_components = fixed.get("components", [])
        xml_gt = parse_cghd_xml(case["xml"])
        component_gt = attach_image(
            [
                {"label": item["name"], "bbox": item["xyxy"]}
                for item in gt_components
            ],
            stem,
        )
        junction_gt = attach_image(xml_gt["junctions"], stem)
        terminal_gt = attach_image(xml_gt["terminals"], stem)
        text_gt = attach_image(xml_gt["texts"], stem)
        all_component_gt.extend(component_gt)
        all_junction_gt.extend(junction_gt)
        all_text_gt.extend(text_gt)
        gt_wiring = parse_gt(case["wiring"])

        prediction_path = output / "predictions" / f"{stem}.json"
        is_cached = resume and prediction_path.is_file()
        imported_path = imported_predictions.get(stem) if not is_cached else None
        cache_label = " (cached)" if is_cached else " (verified import)" if imported_path else ""
        print(f"[{case_index}/{len(cases)}] {stem}{cache_label}", flush=True)
        try:
            if is_cached:
                result = json.loads(prediction_path.read_text(encoding="utf-8"))
            elif imported_path is not None:
                result = json.loads(imported_path.read_text(encoding="utf-8"))
            else:
                result = process_fn(str(case["image"]), config=dict(config))
            if result is None:
                raise RuntimeError("process_image returned None")
            if result.get("evaluation") != "(LLM skipped for experiment)":
                raise RuntimeError("pipeline did not confirm LLM skip")

            component_predictions = attach_image(
                [
                    {
                        "label": item["name"],
                        "bbox": list(item["xyxy"]),
                        "score": float(item.get("conf", 0.0)),
                    }
                    for item in result.get("components", [])
                ],
                stem,
            )
            junction_predictions = attach_image(
                [
                    item
                    for item in result.get("raw_junction_detections", [])
                    if item.get("label") == "junction"
                ],
                stem,
            )
            text_predictions = attach_image(result.get("raw_text_detections", []), stem)
            component_result = detection_metrics(component_predictions, component_gt, 0.5)
            junction_box_result = detection_metrics(junction_predictions, junction_gt, 0.5)
            gt_connection_points = [
                ((item["bbox"][0] + item["bbox"][2]) / 2, (item["bbox"][1] + item["bbox"][3]) / 2)
                for item in [*junction_gt, *terminal_gt]
            ]
            diagonal = math.hypot(float(fixed.get("width", 0)), float(fixed.get("height", 0)))
            junction_point_result = point_metrics(
                result.get("junctions", []), gt_connection_points, max(10.0, diagonal * 0.01)
            )
            port_result = port_metrics(
                result.get("components", []), gt_components, max(10.0, diagonal * 0.03)
            )
            image_ocr_metrics, image_ocr_rows = ocr_metrics_from_detections(
                text_predictions, text_gt, 0.3
            )
            wiring_result = evaluate_wiring(
                result, gt_wiring, gt_components
            )
            value_metrics = image_ocr_metrics["value_text"]
            value_exact_or_na = (
                value_metrics["sample_count"] == 0
                or value_metrics["normalized_exact"] == 1.0
            )
            strict_ok = (
                component_result["fp"] == component_result["fn"] == 0
                and junction_box_result["fp"] == junction_box_result["fn"] == 0
                and port_result["fp"] == port_result["fn"] == 0
                and port_result["label_accuracy"] == 1.0
                and value_exact_or_na
                and wiring_result["edge_fp"] == wiring_result["edge_fn"] == 0
            )
            all_component_predictions.extend(component_predictions)
            all_junction_predictions.extend(junction_predictions)
            all_text_predictions.extend(text_predictions)
            point_rows.append(junction_point_result)
            port_rows.append(port_result)
            wiring_rows.append(wiring_result)
            strict_successes += int(strict_ok)
            per_image_rows.append(
                {
                    "image": stem,
                    "status": "ok",
                    "component_precision": component_result["precision"],
                    "component_recall": component_result["recall"],
                    "junction_precision": junction_box_result["precision"],
                    "junction_recall": junction_box_result["recall"],
                    "port_precision": port_result["precision"],
                    "port_recall": port_result["recall"],
                    "value_exact": image_ocr_metrics["value_text"]["normalized_exact"],
                    "wiring_edge_precision": wiring_result["edge_precision"],
                    "wiring_edge_recall": wiring_result["edge_recall"],
                    "group_accuracy": wiring_result["group_accuracy"],
                    "strict_success": strict_ok,
                }
            )
            if not is_cached:
                _write_json(prediction_path, result)
            rendered_path = output / "annotated_images" / f"{stem}.jpg"
            if render and (not is_cached or not rendered_path.is_file()):
                try:
                    _render_prediction(case["image"], result, rendered_path)
                except Exception as render_error:
                    render_failures.append(
                        {"image": stem, "error": f"{type(render_error).__name__}: {render_error}"}
                    )
                    per_image_rows[-1]["status"] = "render_failed"
        except Exception as error:
            failures.append({"image": stem, "error": f"{type(error).__name__}: {error}"})
            point_rows.append(_safe_prf(0, 0, len(junction_gt) + len(terminal_gt)))
            port_rows.append(_safe_prf(0, 0, sum(len(item.get("ports", [])) for item in gt_components)))
            wiring_rows.append(evaluate_wiring(
                {"components": [], "raw_groups": []}, gt_wiring, gt_components
            ))
            per_image_rows.append({"image": stem, "status": "failed", "strict_success": False})

    component_metrics = detection_metrics(all_component_predictions, all_component_gt, 0.5)
    junction_detection_metrics = detection_metrics(all_junction_predictions, all_junction_gt, 0.5)
    connection_point_metrics = _sum_prf(point_rows)
    port_summary = _sum_prf(port_rows)
    label_correct = sum(int(row.get("label_correct", 0)) for row in port_rows)
    label_compared = sum(int(row.get("label_compared", 0)) for row in port_rows)
    port_summary.update(
        {
            "label_correct": label_correct,
            "label_compared": label_compared,
            "label_accuracy": label_correct / label_compared if label_compared else 0.0,
        }
    )
    text_detection_metrics = detection_metrics(all_text_predictions, all_text_gt, 0.5)
    ocr_metrics, ocr_rows = ocr_metrics_from_detections(all_text_predictions, all_text_gt, 0.3)
    edge_summary = _safe_prf(
        sum(int(row.get("edge_tp", 0)) for row in wiring_rows),
        sum(int(row.get("edge_fp", 0)) for row in wiring_rows),
        sum(int(row.get("edge_fn", 0)) for row in wiring_rows),
    )
    wiring_metrics = {
        "edge": edge_summary,
        "macro_port_correct_rate": sum(float(row.get("port_correct_rate", 0.0)) for row in wiring_rows) / len(cases),
        "macro_group_accuracy": sum(float(row.get("group_accuracy", 0.0)) for row in wiring_rows) / len(cases),
        "macro_component_neighbor_accuracy": sum(float(row.get("comp_neighbor_accuracy", 0.0)) for row in wiring_rows) / len(cases),
        "macro_fp_rate": sum(float(row.get("fp_rate", 0.0)) for row in wiring_rows) / len(cases),
        "macro_fn_rate": sum(float(row.get("fn_rate", 0.0)) for row in wiring_rows) / len(cases),
    }
    summary = {
        "image_count": len(cases),
        "failed_images": len(failures),
        "render_failed_images": len(render_failures),
        "strict_end_to_end_success_count": strict_successes,
        "strict_end_to_end_success_rate": strict_successes / len(cases),
        "component_detection": component_metrics,
        "junction_detection": junction_detection_metrics,
        "processed_connection_points": connection_point_metrics,
        "ports": port_summary,
        "text_detection": text_detection_metrics,
        "ocr": ocr_metrics,
        "wiring": wiring_metrics,
    }
    metric_files = {
        "component_metrics.json": component_metrics,
        "junction_metrics.json": {
            "raw_detection": junction_detection_metrics,
            "processed_connection_points_including_terminals": connection_point_metrics,
        },
        "port_metrics.json": port_summary,
        "value_metrics.json": {
            "text_detection": text_detection_metrics,
            "ocr": ocr_metrics,
        },
        "wiring_metrics.json": wiring_metrics,
        "summary.json": summary,
    }
    for filename, payload in metric_files.items():
        _write_json(output / filename, payload)
    per_image_fields = [
        "image", "status", "component_precision", "component_recall",
        "junction_precision", "junction_recall", "port_precision", "port_recall",
        "value_exact", "wiring_edge_precision", "wiring_edge_recall",
        "group_accuracy", "strict_success",
    ]
    _write_csv(output / "per_image_results.csv", per_image_rows, per_image_fields)
    _write_csv(
        output / "ocr_errors.csv",
        [row for row in ocr_rows if row["normalized_ground_truth"] != row["normalized_prediction"]],
        [
            "image", "ground_truth", "prediction", "normalized_ground_truth",
            "normalized_prediction", "is_electrical_value", "matched_detection",
        ],
    )
    _write_csv(output / "failures.csv", failures, ["image", "error"])
    _write_csv(output / "render_failures.csv", render_failures, ["image", "error"])
    metadata["status"] = "complete"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["failed_images"] = len(failures)
    metadata["render_failed_images"] = len(render_failures)
    _write_json(metadata_path, metadata)
    return summary
