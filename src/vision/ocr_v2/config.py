"""Configuration loading and validation for CRNN-v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .normalize import CANONICAL_CHARS, NORMALIZATION_VERSION


REQUIRED_TOP_LEVEL = {
    "model",
    "data",
    "training",
    "augmentation",
    "promotion",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config and enforce the versioned OCR contract."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")

    model = config["model"]
    expected = {
        "version": "crnn-v2-handwritten-1",
        "chars": CANONICAL_CHARS,
        "img_h": 32,
        "img_w": 160,
        "normalization_version": NORMALIZATION_VERSION,
    }
    for key, expected_value in expected.items():
        if model.get(key) != expected_value:
            raise ValueError(
                f"model.{key} contract mismatch: "
                f"expected {expected_value!r}, got {model.get(key)!r}"
            )
    return config
