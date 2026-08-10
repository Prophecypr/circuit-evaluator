"""Unified runtime adapter for promoted CRNN-v2 and legacy rollback weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vision.train_ocr import (
    load_trained_model as load_legacy_checkpoint,
    predict as legacy_predict,
)

from .checkpoint import load_checkpoint as load_v2_checkpoint
from .image_ops import preprocess_gray
from .model import decode_logits


@dataclass
class OCRRuntime:
    backend: str
    model: torch.nn.Module
    chars: str
    index_to_char: dict[int, str]
    img_h: int
    img_w: int
    metadata: dict[str, Any]
    device: torch.device

    def predict(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0:
            return ""
        if self.backend == "legacy":
            return legacy_predict(
                self.model,
                crop,
                self.chars,
                self.index_to_char,
                self.img_h,
            )
        processed = preprocess_gray(crop, self.img_h, self.img_w)
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
        return decode_logits(logits, self.chars)[0]


def load_ocr_runtime(
    model_path: str | Path, device: str | torch.device | None = None
) -> OCRRuntime:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OCR model not found: {path}")
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict) and metadata.get("model_version") == "crnn-v2-handwritten-1":
        model, validated_metadata = load_v2_checkpoint(path, device=target_device)
        chars = validated_metadata["chars"]
        return OCRRuntime(
            backend="v2",
            model=model,
            chars=chars,
            index_to_char={index + 1: char for index, char in enumerate(chars)},
            img_h=int(validated_metadata["img_h"]),
            img_w=int(validated_metadata["img_w"]),
            metadata=validated_metadata,
            device=target_device,
        )

    model, chars, _, index_to_char, img_h = load_legacy_checkpoint(str(path))
    model = model.to(target_device)
    model.eval()
    return OCRRuntime(
        backend="legacy",
        model=model,
        chars=chars,
        index_to_char=index_to_char,
        img_h=int(img_h),
        img_w=128,
        metadata={
            "model_version": "legacy-crnn",
            "chars": chars,
            "img_h": int(img_h),
            "img_w": 128,
        },
        device=target_device,
    )
