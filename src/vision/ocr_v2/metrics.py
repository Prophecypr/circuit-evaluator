"""Publication-grade metrics for circuit-value OCR."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .normalize import normalize_label


UNIT_GROUPS = ("Ω", "kΩ", "MΩ", "μF", "nF", "V", "H", "A", "Hz")
_PARSE_RE = re.compile(
    r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>(?:[kMmunpμ]?(?:Ω|F|V|A|H)|Hz)?)$"
)
_VOLTAGE_EMBEDDED_RE = re.compile(r"^(?P<major>[+-]?\d+)V(?P<minor>\d+)$")


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row_index, ref_char in enumerate(reference, start=1):
        current = [row_index]
        for col_index, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col_index] + 1,
                    previous[col_index - 1] + (ref_char != pred_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    """Character error rate; insertion-heavy output may exceed 1.0."""
    if not reference:
        return 0.0 if not prediction else float(len(prediction))
    return edit_distance(reference, prediction) / len(reference)


def parse_value(text: str) -> tuple[str, str]:
    normalized = normalize_label(text)
    embedded = _VOLTAGE_EMBEDDED_RE.fullmatch(normalized)
    if embedded:
        return f"{embedded.group('major')}.{embedded.group('minor')}", "V"
    match = _PARSE_RE.fullmatch(normalized)
    if not match:
        return normalized, ""
    return match.group("number"), match.group("unit")


def _aggregate_cer(references: Sequence[str], predictions: Sequence[str]) -> float:
    edits = sum(edit_distance(reference, prediction) for reference, prediction in zip(references, predictions))
    characters = sum(len(reference) for reference in references)
    return edits / max(1, characters)


def evaluate_pairs(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Evaluate ``(ground_truth, prediction)`` pairs in raw and normalized form."""
    if not pairs:
        raise ValueError("cannot evaluate an empty OCR prediction set")
    raw_gt = [ground_truth for ground_truth, _ in pairs]
    raw_pred = [prediction for _, prediction in pairs]
    normalized_gt = [normalize_label(value) for value in raw_gt]
    normalized_pred = [normalize_label(value) for value in raw_pred]
    sample_count = len(pairs)

    numeric_matches = 0
    unit_matches = 0
    unit_labeled_count = 0
    unit_stats = {unit: {"count": 0, "exact": 0.0} for unit in UNIT_GROUPS}
    unit_correct = {unit: 0 for unit in UNIT_GROUPS}
    errors: list[dict[str, str | int]] = []

    for index, (ground_truth, prediction) in enumerate(
        zip(normalized_gt, normalized_pred)
    ):
        gt_number, gt_unit = parse_value(ground_truth)
        pred_number, pred_unit = parse_value(prediction)
        numeric_matches += int(gt_number == pred_number)
        if gt_unit:
            unit_labeled_count += 1
            unit_matches += int(gt_unit == pred_unit)
        if gt_unit in unit_stats:
            unit_stats[gt_unit]["count"] += 1
            unit_correct[gt_unit] += int(ground_truth == prediction)
        if ground_truth != prediction:
            errors.append(
                {
                    "index": index,
                    "raw_ground_truth": raw_gt[index],
                    "raw_prediction": raw_pred[index],
                    "normalized_ground_truth": ground_truth,
                    "normalized_prediction": prediction,
                    "edit_distance": edit_distance(ground_truth, prediction),
                }
            )

    for unit, stats in unit_stats.items():
        count = int(stats["count"])
        stats["exact"] = unit_correct[unit] / count if count else 0.0

    return {
        "sample_count": sample_count,
        "raw_exact": sum(gt == pred for gt, pred in zip(raw_gt, raw_pred)) / sample_count,
        "normalized_exact": sum(
            gt == pred for gt, pred in zip(normalized_gt, normalized_pred)
        )
        / sample_count,
        "raw_cer": _aggregate_cer(raw_gt, raw_pred),
        "normalized_cer": _aggregate_cer(normalized_gt, normalized_pred),
        "numeric_exact": numeric_matches / sample_count,
        "unit_labeled_count": unit_labeled_count,
        "unit_exact": unit_matches / unit_labeled_count if unit_labeled_count else 0.0,
        "per_unit": unit_stats,
        "errors": errors,
    }
