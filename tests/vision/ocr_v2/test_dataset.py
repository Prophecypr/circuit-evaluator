from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.ocr_v2.dataset import CircuitValueDataset, ctc_collate
from src.vision.ocr_v2.dataset_builder import MANIFEST_FIELDS, write_manifest
from src.vision.ocr_v2.normalize import CANONICAL_CHARS


def _make_manifest(tmp_path: Path, label: str = "10kΩ") -> Path:
    crop = tmp_path / "crops" / "a.png"
    crop.parent.mkdir()
    image = np.full((20, 50), 255, np.uint8)
    cv2.putText(image, "10", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    crop.write_bytes(encoded.tobytes())
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        sample_id="a",
        drafter_id="drafter_1",
        source_kind="cghd",
        raw_label=label,
        normalized_label=label,
        crop_path="crops/a.png",
        split="train",
    )
    manifest = tmp_path / "train_manifest.csv"
    write_manifest(manifest, [row])
    return manifest


def test_dataset_resolves_relative_crops_and_encodes_labels(tmp_path):
    manifest = _make_manifest(tmp_path)
    dataset = CircuitValueDataset(
        manifest, chars=CANONICAL_CHARS, img_h=32, img_w=160, augment=False
    )
    image, label, length, metadata = dataset[0]
    assert tuple(image.shape) == (1, 32, 160)
    assert length == len("10kΩ")
    assert len(label) == length
    assert metadata["sample_id"] == "a"
    batch = ctc_collate([dataset[0], dataset[0]])
    assert tuple(batch[0].shape) == (2, 1, 32, 160)
    assert batch[2].tolist() == [length, length]


def test_dataset_rejects_unknown_characters_with_sample_id(tmp_path):
    manifest = _make_manifest(tmp_path, label="10?Ω")
    with pytest.raises(ValueError, match="a"):
        CircuitValueDataset(
            manifest, chars=CANONICAL_CHARS, img_h=32, img_w=160, augment=False
        )


def test_dataset_reads_relative_crop_below_unicode_path(tmp_path):
    data_root = tmp_path / "含中文"
    crop = data_root / "crops" / "a.png"
    crop.parent.mkdir(parents=True)
    image = np.full((20, 50), 255, np.uint8)
    image[:, 10:14] = 0
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    crop.write_bytes(encoded.tobytes())
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        sample_id="unicode",
        drafter_id="drafter_1",
        source_kind="cghd",
        raw_label="10kΩ",
        normalized_label="10kΩ",
        crop_path="crops/a.png",
        split="train",
    )
    manifest = data_root / "train_manifest.csv"
    write_manifest(manifest, [row])
    dataset = CircuitValueDataset(
        manifest, chars=CANONICAL_CHARS, img_h=32, img_w=160, augment=False
    )
    image_tensor, _, _, metadata = dataset[0]
    assert tuple(image_tensor.shape) == (1, 32, 160)
    assert metadata["sample_id"] == "unicode"
