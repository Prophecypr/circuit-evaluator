import numpy as np

from src.vision import unified_pipeline


class FakeRuntime:
    backend = "v2"
    chars = "abc"
    img_h = 32

    def predict(self, crop):
        return "10kΩ"


def test_pipeline_defaults_to_promoted_v2_weight():
    assert (
        unified_pipeline.DEFAULT_CONFIG["ocr_model_path"]
        == "runs/ocr_crnn_hand_v2/best.pt"
    )


def test_pipeline_reloads_when_ocr_path_changes_and_supports_rollback(monkeypatch):
    calls = []
    monkeypatch.setattr(unified_pipeline, "_CGH_MODEL", object())
    monkeypatch.setattr(unified_pipeline, "_OCR_RUNTIME", None)
    monkeypatch.setattr(unified_pipeline, "_OCR_MODEL_PATH", None)
    monkeypatch.setattr(
        unified_pipeline,
        "load_ocr_runtime",
        lambda path: calls.append(path) or FakeRuntime(),
    )

    unified_pipeline._load_models(
        {"ocr_model_path": "runs/ocr_crnn_hand_v2/best.pt"}
    )
    unified_pipeline._load_models(
        {"ocr_model_path": "runs/ocr_crnn_hand_v2/best.pt"}
    )
    unified_pipeline._load_models(
        {"ocr_model_path": "runs/ocr_crnn_machine/best.pt"}
    )

    assert calls == [
        "runs/ocr_crnn_hand_v2/best.pt",
        "runs/ocr_crnn_machine/best.pt",
    ]
    assert unified_pipeline._predict_ocr(np.zeros((8, 8), np.uint8)) == "10kΩ"
