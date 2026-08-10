"""Self-describing, contract-checked CRNN-v2 checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import CRNNv2
from .normalize import CANONICAL_CHARS, NORMALIZATION_VERSION


MODEL_VERSION = "crnn-v2-handwritten-1"
ARCHITECTURE = "cnn-bilstm2-ctc"
IMG_H = 32
IMG_W = 160

REQUIRED_METADATA = {
    "model_version",
    "architecture",
    "chars",
    "img_h",
    "img_w",
    "normalization_version",
    "split_id",
    "seed",
    "epoch",
    "validation_metrics",
    "git_commit",
    "torch_version",
    "training_platform",
}


class CheckpointContractError(ValueError):
    pass


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise CheckpointContractError(f"missing checkpoint metadata: {sorted(missing)}")
    expected = {
        "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE,
        "chars": CANONICAL_CHARS,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "normalization_version": NORMALIZATION_VERSION,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise CheckpointContractError(
                f"{key} mismatch: expected {expected_value!r}, got {metadata.get(key)!r}"
            )


def save_checkpoint(
    path: str | Path,
    model: CRNNv2,
    *,
    metadata: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    allow_replace: bool = False,
) -> None:
    """Atomically save a validated checkpoint, refusing accidental overwrite."""
    validate_metadata(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not allow_replace:
        raise FileExistsError(f"checkpoint already exists: {destination}")
    payload = {
        "model": model.state_dict(),
        "model_hparams": {
            "num_classes": model.num_classes,
            "hidden_size": model.hidden_size,
        },
        "metadata": dict(metadata),
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[CRNNv2, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "metadata" not in payload or "model" not in payload:
        raise CheckpointContractError("not a CRNN-v2 checkpoint")
    metadata = dict(payload["metadata"])
    validate_metadata(metadata)
    hparams = dict(payload.get("model_hparams") or {})
    expected_classes = len(metadata["chars"]) + 1
    if int(hparams.get("num_classes", expected_classes)) != expected_classes:
        raise CheckpointContractError("num_classes does not match checkpoint charset")
    model = CRNNv2(
        num_classes=expected_classes,
        hidden_size=int(hparams.get("hidden_size", 256)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, metadata
