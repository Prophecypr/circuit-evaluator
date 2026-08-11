import numpy as np

from src.vision.ocr_v2.image_ops import augment_phone_photo, preprocess_gray


def test_preprocess_has_fixed_shape_range_and_dtype():
    image = np.full((20, 50), 255, np.uint8)
    image[6:14, 10:40] = 0
    out = preprocess_gray(image, 32, 160)
    assert out.shape == (1, 32, 160)
    assert out.dtype == np.float32
    assert -1.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_validation_preprocess_is_deterministic():
    image = np.arange(800, dtype=np.uint8).reshape(20, 40)
    assert np.array_equal(
        preprocess_gray(image, 32, 160), preprocess_gray(image, 32, 160)
    )


def test_preprocess_rejects_empty_images():
    try:
        preprocess_gray(np.empty((0, 0), np.uint8), 32, 160)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty image was accepted")


def test_seeded_augmentation_is_repeatable_and_keeps_foreground():
    image = np.full((50, 100), 255, np.uint8)
    image[20:30, 15:85] = 0
    config = {
        "probability": 1.0,
        "max_rotation_degrees": 3.0,
        "max_shear_degrees": 2.0,
        "max_perspective_fraction": 0.02,
        "crop_jitter_fraction": 0.05,
        "blur_probability": 1.0,
        "jpeg_probability": 1.0,
        "noise_probability": 1.0,
        "lighting_probability": 1.0,
        "shadow_probability": 1.0,
        "morphology_probability": 1.0,
    }
    first = augment_phone_photo(image, np.random.default_rng(42), config)
    second = augment_phone_photo(image, np.random.default_rng(42), config)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert np.count_nonzero(first < 245) > 0
