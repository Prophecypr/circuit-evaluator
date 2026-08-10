"""Independent evaluator for self-describing CRNN-v2 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .dataset import CircuitValueDataset, ctc_collate
from .metrics import evaluate_pairs
from .model import decode_logits


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, *, stream: TextIO | None = None) -> None:
    """Print JSON without depending on the host console's Unicode codec."""
    target = stream or sys.stdout
    target.write(json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return console-sized metrics while keeping full errors in artifact files."""
    return {
        key: value for key, value in metrics.items() if key != "errors"
    } | {"error_count": len(metrics.get("errors", []))}


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint(checkpoint_path, device=selected_device)
    dataset = CircuitValueDataset(
        manifest_path,
        chars=metadata["chars"],
        img_h=int(metadata["img_h"]),
        img_w=int(metadata["img_w"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate,
    )
    prediction_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for images, _, _, batch_metadata in loader:
            predictions = decode_logits(model(images.to(selected_device)), metadata["chars"])
            for sample, prediction in zip(batch_metadata, predictions):
                prediction_rows.append(
                    {
                        **sample,
                        "prediction": prediction,
                    }
                )

    metrics = evaluate_pairs(
        [(row["raw_label"], row["prediction"]) for row in prediction_rows]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        output / "predictions.csv",
        prediction_rows,
        [
            "sample_id",
            "drafter_id",
            "source_kind",
            "raw_label",
            "normalized_label",
            "prediction",
        ],
    )
    error_rows = [prediction_rows[int(error["index"])] | error for error in metrics["errors"]]
    _write_csv(
        output / "errors.csv",
        error_rows,
        [
            "sample_id",
            "drafter_id",
            "source_kind",
            "raw_ground_truth",
            "raw_prediction",
            "normalized_ground_truth",
            "normalized_prediction",
            "edit_distance",
        ],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    write_json(metrics_summary(metrics))


if __name__ == "__main__":
    main()
