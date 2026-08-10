from pathlib import Path

import numpy as np
import torch

from src.vision.ocr_v2.runtime import OCRRuntime, load_ocr_runtime


class FakeV2Model(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 3, 3, device=images.device)
        logits[:, 0, 1] = 10
        logits[:, 1, 0] = 10
        logits[:, 2, 2] = 10
        return logits


def test_v2_runtime_uses_checkpoint_preprocessing_and_decoding():
    runtime = OCRRuntime(
        backend="v2",
        model=FakeV2Model(),
        chars="ab",
        index_to_char={1: "a", 2: "b"},
        img_h=32,
        img_w=160,
        metadata={"model_version": "crnn-v2-handwritten-1"},
        device=torch.device("cpu"),
    )
    crop = np.full((20, 50), 255, np.uint8)
    crop[5:15, 10:40] = 0
    assert runtime.predict(crop) == "ab"


def test_loader_selects_v2_checkpoint_contract(monkeypatch, tmp_path):
    checkpoint = tmp_path / "v2.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "model_version": "crnn-v2-handwritten-1",
        "chars": "ab",
        "img_h": 32,
        "img_w": 160,
    }
    monkeypatch.setattr(
        "src.vision.ocr_v2.runtime.torch.load",
        lambda *args, **kwargs: {"metadata": metadata},
    )
    monkeypatch.setattr(
        "src.vision.ocr_v2.runtime.load_v2_checkpoint",
        lambda *args, **kwargs: (FakeV2Model(), metadata),
    )
    runtime = load_ocr_runtime(checkpoint, device="cpu")
    assert runtime.backend == "v2"
    assert runtime.img_w == 160


def test_loader_keeps_explicit_legacy_rollback(monkeypatch, tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "src.vision.ocr_v2.runtime.torch.load",
        lambda *args, **kwargs: {"chars": "ab", "img_h": 32},
    )
    monkeypatch.setattr(
        "src.vision.ocr_v2.runtime.load_legacy_checkpoint",
        lambda *args, **kwargs: (
            FakeV2Model(),
            "ab",
            {"a": 1, "b": 2},
            {0: "", 1: "a", 2: "b"},
            32,
        ),
    )
    runtime = load_ocr_runtime(checkpoint, device="cpu")
    assert runtime.backend == "legacy"
    assert runtime.img_w == 128
