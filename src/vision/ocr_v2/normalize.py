"""Canonical label handling for circuit-value OCR."""

from __future__ import annotations

import re


NORMALIZATION_VERSION = "circuit-value-v1"
CANONICAL_CHARS = "0123456789.kmMΩμunpFV AHz-+/"

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT = r"(?:[kMmunpμ]?(?:Ω|F|V|A|H)|Hz)"
_VALUE_RE = re.compile(rf"(?:{_NUMBER}(?:{_UNIT})?|[+-]?\d+V\d+)")


def normalize_label(text: str) -> str:
    """Normalize visual aliases while preserving engineering semantics."""
    value = "".join(str(text).strip().split())
    value = value.replace("µ", "μ").replace("u", "μ")
    value = value.replace("Ω", "Ω").replace("ω", "Ω")
    value = re.sub(r"(?<=\d)K(?=Ω|$)", "k", value)
    return value


def validate_normalized_label(text: str) -> str:
    """Return *text* if it obeys the canonical character contract."""
    unsupported = sorted(set(text) - set(CANONICAL_CHARS))
    if unsupported:
        rendered = "".join(unsupported)
        raise ValueError(f"unsupported OCR characters: {rendered!r}")
    return text


def is_value_label(text: str) -> bool:
    """Whether *text* is a value-like label rather than an identifier."""
    normalized = normalize_label(text)
    if not normalized:
        return False
    try:
        validate_normalized_label(normalized)
    except ValueError:
        return False
    return _VALUE_RE.fullmatch(normalized) is not None
