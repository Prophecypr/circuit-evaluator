from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.ocr_v2.dataset_builder import (
    assert_no_leakage,
    collect_file_hashes,
    extract_digitize_rows,
    sha256_file,
    split_cghd_rows,
)


def _row(tmp_path: Path, sample_id: str, drafter: str, source_hash: str, crop_hash: str):
    crop = tmp_path / f"{sample_id}.png"
    crop.write_bytes(sample_id.encode("ascii"))
    return {
        "sample_id": sample_id,
        "drafter_id": drafter,
        "source_kind": "cghd",
        "source_image": str(tmp_path / f"{sample_id}.jpg"),
        "source_xml": str(tmp_path / f"{sample_id}.xml"),
        "bbox": "1,2,3,4",
        "raw_label": "10KΩ",
        "normalized_label": "10kΩ",
        "source_sha256": source_hash,
        "crop_sha256": crop_hash,
        "crop_path": str(crop),
        "split": "",
    }


def test_writer_split_has_no_drafter_overlap(tmp_path):
    rows = [
        _row(tmp_path, "a", "drafter_1", "s1", "c1"),
        _row(tmp_path, "b", "drafter_2", "s2", "c2"),
        _row(tmp_path, "c", "drafter_2", "s3", "c3"),
    ]
    train, val = split_cghd_rows(rows, validation_drafters={"drafter_2"})
    assert {row["sample_id"] for row in train} == {"a"}
    assert {row["sample_id"] for row in val} == {"b", "c"}
    assert {row["drafter_id"] for row in train}.isdisjoint(
        {row["drafter_id"] for row in val}
    )
    assert all(row["split"] == "train" for row in train)
    assert all(row["split"] == "val" for row in val)


def test_hash_guard_rejects_final_test_source_or_crop(tmp_path):
    rows = [_row(tmp_path, "a", "drafter_1", "blocked", "c1")]
    with pytest.raises(ValueError, match="final-test hash"):
        assert_no_leakage(rows, forbidden_hashes={"blocked"})


def test_guard_rejects_duplicate_sample_ids_and_split_hash_overlap(tmp_path):
    duplicate = [
        _row(tmp_path, "a", "drafter_1", "s1", "c1"),
        _row(tmp_path, "a", "drafter_2", "s2", "c2"),
    ]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        assert_no_leakage(duplicate, forbidden_hashes=set())

    overlap = [
        {**_row(tmp_path, "b", "drafter_1", "shared", "c3"), "split": "train"},
        {**_row(tmp_path, "c", "drafter_2", "shared", "c4"), "split": "val"},
    ]
    with pytest.raises(ValueError, match="source_sha256 overlap"):
        assert_no_leakage(overlap, forbidden_hashes=set())


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_final_test_hash_collection_ignores_only_office_lock_files(tmp_path):
    real = tmp_path / "value_gt_template.xlsx"
    lock = tmp_path / "~$value_gt_template.xlsx"
    real.write_bytes(b"real-gt")
    lock.write_bytes(b"office-lock")
    hashes = collect_file_hashes([tmp_path])
    assert sha256_file(real) in hashes
    assert sha256_file(lock) not in hashes


def test_digitize_extraction_reads_images_below_unicode_paths(tmp_path):
    source = tmp_path / "含中文" / "ocr_training"
    source.mkdir(parents=True)
    image = np.full((12, 30), 255, np.uint8)
    image[:, 8:12] = 0
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    crop = source / "sample.png"
    crop.write_bytes(encoded.tobytes())
    (source / "train_labels.txt").write_text(
        f"{crop}\t10kΩ\n", encoding="utf-8"
    )
    rows = extract_digitize_rows(source, tmp_path / "output")
    assert len(rows) == 1
    assert rows[0]["normalized_label"] == "10kΩ"
