from pathlib import Path

import pytest
import torch

from src.vision.ocr_v2.checkpoint import (
    CheckpointContractError,
    load_checkpoint,
    save_checkpoint,
    validate_metadata,
)
from src.vision.ocr_v2.model import CRNNv2, greedy_decode
from src.vision.ocr_v2.normalize import CANONICAL_CHARS, NORMALIZATION_VERSION


VALID_METADATA = {
    "model_version": "crnn-v2-handwritten-1",
    "architecture": "cnn-bilstm2-ctc",
    "chars": CANONICAL_CHARS,
    "img_h": 32,
    "img_w": 160,
    "normalization_version": NORMALIZATION_VERSION,
    "split_id": "writers-v1",
    "seed": 42,
    "epoch": 1,
    "validation_metrics": {"normalized_exact": 0.5, "normalized_cer": 0.4},
    "git_commit": "test",
    "torch_version": torch.__version__,
    "training_platform": "pytest",
}


def test_crnn_forward_and_greedy_decode_contract():
    model = CRNNv2(num_classes=len(CANONICAL_CHARS) + 1)
    output = model(torch.zeros(2, 1, 32, 160))
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[2] == len(CANONICAL_CHARS) + 1
    assert greedy_decode([1, 1, 0, 2, 2, 0, 2], {1: "a", 2: "b"}) == "abb"


def test_checkpoint_round_trip_preserves_contract(tmp_path):
    path = tmp_path / "best.pt"
    model = CRNNv2(num_classes=len(CANONICAL_CHARS) + 1)
    save_checkpoint(path, model, metadata=VALID_METADATA)
    loaded, metadata = load_checkpoint(path, device="cpu")
    assert isinstance(loaded, CRNNv2)
    assert metadata["model_version"] == "crnn-v2-handwritten-1"
    assert metadata["chars"] == CANONICAL_CHARS


def test_checkpoint_rejects_wrong_image_width():
    with pytest.raises(CheckpointContractError, match="img_w"):
        validate_metadata({**VALID_METADATA, "img_w": 128})


def test_checkpoint_refuses_accidental_overwrite(tmp_path):
    path = tmp_path / "best.pt"
    model = CRNNv2(num_classes=len(CANONICAL_CHARS) + 1)
    save_checkpoint(path, model, metadata=VALID_METADATA)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, metadata=VALID_METADATA)
