"""Build provenance-aware OCR crops and leakage-safe manifests."""

from __future__ import annotations

import csv
import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from .normalize import is_value_label, normalize_label, validate_normalized_label


MANIFEST_FIELDS = (
    "sample_id",
    "drafter_id",
    "source_kind",
    "source_image",
    "source_xml",
    "bbox",
    "raw_label",
    "normalized_label",
    "source_sha256",
    "crop_sha256",
    "crop_path",
    "split",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray_file(path: str | Path):
    """Read an image through Python bytes so Windows Unicode paths are safe."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def collect_file_hashes(roots: Iterable[str | Path]) -> set[str]:
    """Hash all regular files below *roots* for final-test exclusion."""
    hashes: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if path.is_file() and not path.name.startswith("~$"):
                hashes.add(sha256_file(path))
    return hashes


def split_cghd_rows(
    rows: Sequence[Mapping[str, str]], validation_drafters: set[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split CGHD samples by drafter without mutating input rows."""
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        if row["drafter_id"] in validation_drafters:
            row["split"] = "val"
            val.append(row)
        else:
            row["split"] = "train"
            train.append(row)
    return train, val


def _split_values(rows: Sequence[Mapping[str, str]], field: str, split: str) -> set[str]:
    return {row[field] for row in rows if row.get("split") == split and row.get(field)}


def assert_no_leakage(
    rows: Sequence[Mapping[str, str]],
    forbidden_hashes: set[str],
    forbidden_path_fragments: Iterable[str] = (),
) -> None:
    """Raise on final-test leakage or train/validation identity overlap."""
    ids = [row.get("sample_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id in OCR manifest")

    forbidden_fragments = [fragment.casefold() for fragment in forbidden_path_fragments]
    for row in rows:
        normalized = row.get("normalized_label", "")
        validate_normalized_label(normalized)
        if not is_value_label(normalized):
            raise ValueError(f"non-value label in OCR manifest: {normalized!r}")
        for field in ("source_sha256", "crop_sha256"):
            if row.get(field) in forbidden_hashes:
                raise ValueError(
                    f"final-test hash found in {field}: {row.get('sample_id', '')}"
                )
        for field in ("source_image", "source_xml", "crop_path"):
            value = str(row.get(field, "")).casefold()
            if any(fragment in value for fragment in forbidden_fragments):
                raise ValueError(
                    f"forbidden final-test path found in {field}: {row.get(field, '')}"
                )

    train_drafters = {
        row["drafter_id"]
        for row in rows
        if row.get("split") == "train" and row.get("source_kind") == "cghd"
    }
    val_drafters = {
        row["drafter_id"]
        for row in rows
        if row.get("split") == "val" and row.get("source_kind") == "cghd"
    }
    if train_drafters & val_drafters:
        raise ValueError("drafter_id overlap between train and validation")

    for field in ("source_sha256", "crop_sha256"):
        overlap = _split_values(rows, field, "train") & _split_values(rows, field, "val")
        if overlap:
            raise ValueError(f"{field} overlap between train and validation")


def _find_image(img_dir: Path, xml_root: ET.Element, xml_path: Path) -> Path | None:
    filename = xml_root.findtext("filename") or f"{xml_path.stem}.jpg"
    candidates = [img_dir / filename]
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        candidates.append(img_dir / f"{Path(filename).stem}{suffix}")
    return next((path for path in candidates if path.is_file()), None)


def extract_cghd_rows(cghd_root: str | Path, data_dir: str | Path) -> list[dict[str, str]]:
    """Extract value-like CGHD text objects into deterministic crop files."""
    root = Path(cghd_root)
    destination = Path(data_dir)
    crop_dir = destination / "crops" / "cghd"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for drafter_dir in sorted(root.glob("drafter_*"), key=lambda path: path.name):
        annotations = drafter_dir / "annotations"
        images = drafter_dir / "images"
        if not annotations.is_dir() or not images.is_dir():
            continue
        for xml_path in sorted(annotations.glob("*.xml"), key=lambda path: path.name):
            try:
                xml_root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            image_path = _find_image(images, xml_root, xml_path)
            if image_path is None:
                continue
            image = read_gray_file(image_path)
            if image is None:
                continue
            source_hash = sha256_file(image_path)
            img_h, img_w = image.shape
            for object_index, obj in enumerate(xml_root.findall("object")):
                if (obj.findtext("name") or "").strip() != "text":
                    continue
                raw_label = (obj.findtext("text") or "").strip()
                normalized = normalize_label(raw_label)
                if not is_value_label(normalized):
                    continue
                bbox = obj.find("bndbox")
                if bbox is None:
                    continue
                try:
                    x1 = int(float(bbox.findtext("xmin", "0")))
                    y1 = int(float(bbox.findtext("ymin", "0")))
                    x2 = int(float(bbox.findtext("xmax", "0")))
                    y2 = int(float(bbox.findtext("ymax", "0")))
                except (TypeError, ValueError):
                    continue
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = image[y1:y2, x1:x2]
                ok, encoded = cv2.imencode(".png", crop)
                if not ok:
                    continue
                crop_bytes = encoded.tobytes()
                identity = "|".join(
                    (
                        drafter_dir.name,
                        xml_path.name,
                        str(object_index),
                        f"{x1},{y1},{x2},{y2}",
                        normalized,
                    )
                ).encode("utf-8")
                sample_id = f"cghd_{hashlib.sha256(identity).hexdigest()[:20]}"
                relative_crop = Path("crops") / "cghd" / f"{sample_id}.png"
                (destination / relative_crop).write_bytes(crop_bytes)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "drafter_id": drafter_dir.name,
                        "source_kind": "cghd",
                        "source_image": f"{drafter_dir.name}/images/{image_path.name}",
                        "source_xml": f"{drafter_dir.name}/annotations/{xml_path.name}",
                        "bbox": f"{x1},{y1},{x2},{y2}",
                        "raw_label": raw_label,
                        "normalized_label": normalized,
                        "source_sha256": source_hash,
                        "crop_sha256": sha256_bytes(crop_bytes),
                        "crop_path": relative_crop.as_posix(),
                        "split": "",
                    }
                )
    return rows


def extract_digitize_rows(
    digitize_root: str | Path, data_dir: str | Path
) -> list[dict[str, str]]:
    """Copy value-like historical crops into the training-only partition."""
    root = Path(digitize_root)
    destination = Path(data_dir)
    crop_dir = destination / "crops" / "digitize_hcd"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for labels_path in sorted(root.glob("*labels.txt"), key=lambda path: path.name):
        with labels_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                if "\t" not in line:
                    continue
                raw_path, raw_label = line.split("\t", 1)
                normalized = normalize_label(raw_label)
                if not is_value_label(normalized):
                    continue
                source_path = Path(raw_path)
                if not source_path.is_file():
                    source_path = root / source_path.name
                if not source_path.is_file():
                    continue
                identity_key = (str(source_path.resolve()), normalized)
                if identity_key in seen:
                    continue
                seen.add(identity_key)
                image = read_gray_file(source_path)
                if image is None:
                    continue
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    continue
                crop_bytes = encoded.tobytes()
                identity = f"{source_path.name}|{normalized}".encode("utf-8")
                sample_id = f"digitize_{hashlib.sha256(identity).hexdigest()[:20]}"
                relative_crop = Path("crops") / "digitize_hcd" / f"{sample_id}.png"
                (destination / relative_crop).write_bytes(crop_bytes)
                source_hash = sha256_file(source_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "drafter_id": "unknown_digitize_hcd",
                        "source_kind": "digitize_hcd",
                        "source_image": source_path.name,
                        "source_xml": "",
                        "bbox": "",
                        "raw_label": raw_label,
                        "normalized_label": normalized,
                        "source_sha256": source_hash,
                        "crop_sha256": sha256_bytes(crop_bytes),
                        "crop_path": relative_crop.as_posix(),
                        "split": "train",
                    }
                )
    return rows


def write_manifest(path: str | Path, rows: Sequence[Mapping[str, str]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
