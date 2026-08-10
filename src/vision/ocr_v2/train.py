"""Reproducible CRNN-v2 training CLI for Kaggle GPU or local smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config
from .dataset import CircuitValueDataset, ctc_collate
from .metrics import evaluate_pairs
from .model import CRNNv2, decode_logits


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _split_id(config: dict[str, Any]) -> str:
    drafters = ",".join(config["data"]["validation_drafters"])
    import hashlib

    return f"writers-v1-{hashlib.sha256(drafters.encode('utf-8')).hexdigest()[:12]}"


def _checkpoint_metadata(
    config: dict[str, Any], epoch: int, metrics: dict[str, Any], device: str
) -> dict[str, Any]:
    model = config["model"]
    return {
        "model_version": model["version"],
        "architecture": model["architecture"],
        "chars": model["chars"],
        "img_h": model["img_h"],
        "img_w": model["img_w"],
        "normalization_version": model["normalization_version"],
        "split_id": _split_id(config),
        "seed": config["data"]["seed"],
        "epoch": epoch,
        "validation_metrics": {
            key: value for key, value in metrics.items() if key != "errors"
        },
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "training_platform": f"{platform.platform()}|device={device}",
    }


def _ctc_loss(
    criterion: nn.CTCLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_lengths: torch.Tensor,
) -> torch.Tensor:
    time_major = logits.permute(1, 0, 2).log_softmax(2)
    input_lengths = torch.full(
        (logits.shape[0],),
        time_major.shape[0],
        dtype=torch.long,
        device=logits.device,
    )
    return criterion(time_major, labels, input_lengths, label_lengths)


def _run_validation(
    model: CRNNv2,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: str,
    chars: str,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    batches = 0
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, labels, lengths, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(images)
            total_loss += float(_ctc_loss(criterion, logits, labels, lengths).item())
            batches += 1
            predictions = decode_logits(logits, chars)
            for sample, prediction in zip(metadata, predictions):
                rows.append({**sample, "prediction": prediction})
    metrics = evaluate_pairs([(row["raw_label"], row["prediction"]) for row in rows])
    return total_loss / max(1, batches), metrics, rows


def _write_error_csv(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "drafter_id",
        "source_kind",
        "raw_ground_truth",
        "raw_prediction",
        "normalized_ground_truth",
        "normalized_prediction",
        "edit_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for error in metrics["errors"]:
            writer.writerow(rows[int(error["index"])] | error)


def train_from_config(
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    max_epochs: int | None = None,
    max_batches: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    seed = int(config["data"]["seed"])
    _set_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment_lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"torch={torch.__version__}",
        f"opencv={__import__('cv2').__version__}",
        f"numpy={np.__version__}",
        f"platform={platform.platform()}",
        f"device={selected_device}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
    ]
    (output / "environment.txt").write_text("\n".join(environment_lines) + "\n", encoding="utf-8")

    model_config = config["model"]
    train_dataset = CircuitValueDataset(
        Path(data_root) / config["data"]["train_manifest"],
        chars=model_config["chars"],
        img_h=model_config["img_h"],
        img_w=model_config["img_w"],
        augment=True,
        augmentation_config=config["augmentation"],
        seed=seed,
    )
    val_dataset = CircuitValueDataset(
        Path(data_root) / config["data"]["val_manifest"],
        chars=model_config["chars"],
        img_h=model_config["img_h"],
        img_w=model_config["img_w"],
        augment=False,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    training = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        collate_fn=ctc_collate,
        pin_memory=selected_device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        collate_fn=ctc_collate,
        pin_memory=selected_device.startswith("cuda"),
    )

    model = CRNNv2(len(model_config["chars"]) + 1).to(selected_device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(max_epochs or training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history_path = output / "history.csv"
    history_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "normalized_exact",
        "normalized_cer",
        "learning_rate",
        "seconds",
    ]
    best_exact = -1.0
    best_cer = float("inf")
    stale_epochs = 0
    epochs_completed = 0
    best_metrics: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []

    with history_path.open("w", encoding="utf-8", newline="") as history_handle:
        history_writer = csv.DictWriter(history_handle, fieldnames=history_fields)
        history_writer.writeheader()
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            losses: list[float] = []
            for batch_index, (images, labels, lengths, _) in enumerate(train_loader, start=1):
                images = images.to(selected_device, non_blocking=True)
                labels = labels.to(selected_device, non_blocking=True)
                lengths = lengths.to(selected_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = _ctc_loss(criterion, logits, labels, lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                optimizer.step()
                losses.append(float(loss.item()))
                if max_batches is not None and batch_index >= max_batches:
                    break

            val_loss, metrics, rows = _run_validation(
                model, val_loader, criterion, selected_device, model_config["chars"]
            )
            exact = float(metrics["normalized_exact"])
            validation_cer = float(metrics["normalized_cer"])
            improved = exact > best_exact or (
                exact == best_exact and validation_cer < best_cer
            )
            metadata = _checkpoint_metadata(config, epoch, metrics, selected_device)
            if improved:
                best_exact, best_cer = exact, validation_cer
                best_metrics, best_rows = metrics, rows
                stale_epochs = 0
                save_checkpoint(
                    output / "best.pt",
                    model,
                    metadata=metadata,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    allow_replace=True,
                )
            else:
                stale_epochs += 1
            save_checkpoint(
                output / "last.pt",
                model,
                metadata=metadata,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                allow_replace=True,
            )
            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": sum(losses) / max(1, len(losses)),
                    "val_loss": val_loss,
                    "normalized_exact": exact,
                    "normalized_cer": validation_cer,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - start,
                }
            )
            history_handle.flush()
            epochs_completed = epoch
            scheduler.step()
            if stale_epochs >= int(training["early_stopping_patience"]):
                break

    if best_metrics is None:
        raise RuntimeError("training did not produce a best checkpoint")
    (output / "metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_error_csv(output / "errors.csv", best_metrics, best_rows)
    return {
        "epochs_completed": epochs_completed,
        "best_normalized_exact": best_exact,
        "best_normalized_cer": best_cer,
        "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    result = train_from_config(
        args.config,
        args.data_root,
        args.output,
        device=args.device,
        max_epochs=args.max_epochs,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
