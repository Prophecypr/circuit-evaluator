import json
import io
from pathlib import Path

import cv2
import numpy as np

from src.vision.ocr_v2.dataset_builder import MANIFEST_FIELDS, write_manifest
from src.vision.ocr_v2.evaluate import evaluate_checkpoint, metrics_summary, write_json
from src.vision.ocr_v2.train import train_from_config


def _row(sample_id: str, label: str, split: str) -> dict[str, str]:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        sample_id=sample_id,
        drafter_id="drafter_train" if split == "train" else "drafter_val",
        source_kind="cghd",
        raw_label=label,
        normalized_label=label,
        crop_path=f"crops/{sample_id}.png",
        split=split,
    )
    return row


def _make_smoke_data(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    crops = data_root / "crops"
    crops.mkdir(parents=True)
    train_rows = [_row("train_1", "10kΩ", "train"), _row("train_2", "4.7μF", "train")]
    val_rows = [_row("val_1", "10kΩ", "val"), _row("val_2", "4.7μF", "val")]
    for row in train_rows + val_rows:
        image = np.full((24, 80), 255, np.uint8)
        cv2.putText(image, row["normalized_label"][:2], (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        (data_root / row["crop_path"]).write_bytes(encoded.tobytes())
    write_manifest(data_root / "train_manifest.csv", train_rows)
    write_manifest(data_root / "val_manifest.csv", val_rows)

    source = json.loads(Path("configs/ocr_crnn_hand_v2.json").read_text(encoding="utf-8"))
    source["training"].update(batch_size=2, num_workers=0, epochs=1, early_stopping_patience=1)
    source["augmentation"]["probability"] = 0.0
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    return config_path, data_root


def test_one_batch_training_exports_and_reloads_all_artifacts(tmp_path):
    config_path, data_root = _make_smoke_data(tmp_path)
    output_dir = tmp_path / "run"
    result = train_from_config(
        config_path,
        data_root,
        output_dir,
        device="cpu",
        max_epochs=1,
        max_batches=1,
    )
    assert result["epochs_completed"] == 1
    for name in (
        "last.pt",
        "best.pt",
        "history.csv",
        "metrics.json",
        "errors.csv",
        "run_config.json",
        "environment.txt",
    ):
        assert (output_dir / name).is_file(), name

    eval_dir = tmp_path / "independent_eval"
    metrics = evaluate_checkpoint(
        output_dir / "best.pt",
        data_root / "val_manifest.csv",
        eval_dir,
        device="cpu",
        batch_size=2,
    )
    assert metrics["sample_count"] == 2
    assert (eval_dir / "predictions.csv").is_file()
    assert (eval_dir / "errors.csv").is_file()
    assert (eval_dir / "metrics.json").is_file()


def test_console_json_is_safe_on_windows_gbk_stream():
    binary = io.BytesIO()
    stream = io.TextIOWrapper(binary, encoding="gbk")
    write_json({"raw_label": "4.7µF", "unit": "μF"}, stream=stream)
    stream.flush()
    rendered = binary.getvalue().decode("gbk")
    assert "\\u00b5" in rendered
    assert "\\u03bc" in rendered


def test_console_summary_counts_errors_without_printing_every_row():
    summary = metrics_summary(
        {"normalized_exact": 0.5, "normalized_cer": 0.2, "errors": [{}, {}]}
    )
    assert summary == {
        "normalized_exact": 0.5,
        "normalized_cer": 0.2,
        "error_count": 2,
    }
