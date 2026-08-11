"""Shared OCR preprocessing and phone-photo augmentation."""

from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np


def _as_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("empty OCR image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"expected grayscale or BGR image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def preprocess_gray(image: np.ndarray, img_h: int = 32, img_w: int = 160) -> np.ndarray:
    """Resize, pad, invert, and normalize one OCR crop to ``[1,H,W]``."""
    gray = _as_gray_uint8(image)
    if img_h <= 0 or img_w <= 0:
        raise ValueError("OCR image dimensions must be positive")
    height, width = gray.shape
    scale = min(img_h / height, img_w / width)
    resized_w = max(1, min(img_w, int(round(width * scale))))
    resized_h = max(1, min(img_h, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (resized_w, resized_h), interpolation=interpolation)

    # White image-space padding becomes black background after inversion.
    canvas = np.full((img_h, img_w), 255, dtype=np.uint8)
    y0 = (img_h - resized_h) // 2
    canvas[y0 : y0 + resized_h, :resized_w] = resized
    inverted = 255.0 - canvas.astype(np.float32)
    normalized = inverted / 127.5 - 1.0
    return normalized[np.newaxis, :, :].astype(np.float32, copy=False)


def _probability(config: Mapping[str, float], key: str) -> float:
    return min(1.0, max(0.0, float(config.get(key, 0.0))))


def augment_phone_photo(
    image: np.ndarray,
    rng: np.random.Generator,
    config: Mapping[str, float],
) -> np.ndarray:
    """Apply bounded, seeded distortions representative of phone photographs."""
    original = _as_gray_uint8(image)
    if rng.random() >= _probability(config, "probability"):
        return original.copy()
    result = original.copy()
    height, width = result.shape

    rotation = float(config.get("max_rotation_degrees", 0.0))
    shear = float(config.get("max_shear_degrees", 0.0))
    jitter = float(config.get("crop_jitter_fraction", 0.0))
    angle = rng.uniform(-rotation, rotation)
    shear_value = np.tan(np.deg2rad(rng.uniform(-shear, shear)))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[0, 1] += shear_value
    matrix[0, 2] += rng.uniform(-jitter, jitter) * width
    matrix[1, 2] += rng.uniform(-jitter, jitter) * height
    result = cv2.warpAffine(
        result,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    perspective = float(config.get("max_perspective_fraction", 0.0))
    if perspective > 0:
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        scale = np.float32([width, height]) * perspective
        offsets = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32) * scale
        transform = cv2.getPerspectiveTransform(src, src + offsets)
        result = cv2.warpPerspective(
            result,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    if rng.random() < _probability(config, "lighting_probability"):
        contrast = rng.uniform(0.78, 1.22)
        brightness = rng.uniform(-22.0, 22.0)
        result = np.clip(result.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    if rng.random() < _probability(config, "shadow_probability"):
        axis = np.linspace(rng.uniform(0.65, 1.0), rng.uniform(0.65, 1.0), width)
        if rng.random() < 0.5:
            axis = axis[::-1]
        shadow = np.tile(axis, (height, 1))
        result = np.clip(result.astype(np.float32) * shadow, 0, 255).astype(np.uint8)

    if rng.random() < _probability(config, "blur_probability"):
        if rng.random() < 0.5:
            result = cv2.GaussianBlur(result, (3, 3), rng.uniform(0.2, 1.0))
        else:
            kernel = np.zeros((3, 3), np.float32)
            kernel[1, :] = 1.0 / 3.0
            result = cv2.filter2D(result, -1, kernel)

    if rng.random() < _probability(config, "noise_probability"):
        noise = rng.normal(0.0, rng.uniform(2.0, 10.0), result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < _probability(config, "morphology_probability"):
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.erode(result, kernel, iterations=1) if rng.random() < 0.5 else cv2.dilate(result, kernel, iterations=1)

    if rng.random() < _probability(config, "jpeg_probability"):
        quality = int(rng.integers(45, 91))
        ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                result = decoded

    if np.count_nonzero(result < 245) == 0:
        return original.copy()
    return result.astype(np.uint8, copy=False)
