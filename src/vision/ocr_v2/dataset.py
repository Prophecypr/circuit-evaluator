"""Manifest-backed PyTorch dataset for circuit-value OCR."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_builder import read_gray_file, read_manifest
from .image_ops import augment_phone_photo, preprocess_gray


class CircuitValueDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        chars: str,
        img_h: int,
        img_w: int,
        augment: bool,
        augmentation_config: Mapping[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = self.manifest_path.parent
        self.rows = read_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"empty OCR manifest: {self.manifest_path}")
        self.chars = chars
        self.char_to_index = {char: index + 1 for index, char in enumerate(chars)}
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.augment = bool(augment)
        self.augmentation_config = dict(augmentation_config or {})
        self.seed = int(seed)
        self.epoch = 0
        for row in self.rows:
            label = row["normalized_label"]
            unsupported = sorted(set(label) - set(chars))
            if unsupported:
                raise ValueError(
                    f"sample {row['sample_id']} has unsupported characters: "
                    f"{''.join(unsupported)!r}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, sample_id: str) -> np.random.Generator:
        payload = f"{self.seed}|{self.epoch}|{sample_id}".encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return np.random.default_rng(sample_seed)

    def __getitem__(self, index: int):
        row = self.rows[index]
        crop_path = Path(row["crop_path"])
        if not crop_path.is_absolute():
            crop_path = self.data_root / crop_path
        image = read_gray_file(crop_path) if crop_path.is_file() else None
        if image is None:
            raise FileNotFoundError(
                f"OCR crop missing for sample {row['sample_id']}: {crop_path}"
            )
        if self.augment:
            image = augment_phone_photo(
                image, self._rng(row["sample_id"]), self.augmentation_config
            )
        processed = preprocess_gray(image, self.img_h, self.img_w)
        label = torch.tensor(
            [self.char_to_index[char] for char in row["normalized_label"]],
            dtype=torch.long,
        )
        metadata = {
            "sample_id": row["sample_id"],
            "raw_label": row["raw_label"],
            "normalized_label": row["normalized_label"],
            "source_kind": row["source_kind"],
            "drafter_id": row["drafter_id"],
        }
        return torch.from_numpy(processed), label, len(label), metadata


def ctc_collate(batch: Sequence[tuple[torch.Tensor, torch.Tensor, int, dict]]):
    images, labels, lengths, metadata = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.cat(labels, dim=0),
        torch.tensor(lengths, dtype=torch.long),
        list(metadata),
    )
