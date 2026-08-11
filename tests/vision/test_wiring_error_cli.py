import json
from pathlib import Path

import pytest

from src.vision.wiring_error_cli import prepare_output, validate_inputs


def _seed_valid_inputs(tmp_path, stem="case_a"):
    run_dir = tmp_path / "run"
    benchmark_dir = tmp_path / "benchmark"
    (run_dir / "predictions" / "strict_jj").mkdir(parents=True)
    (run_dir / "wiring_traces_strict_jj").mkdir(parents=True)
    (benchmark_dir / "result").mkdir(parents=True)
    (benchmark_dir / "detections").mkdir(parents=True)
    metadata = {
        "image_count": 1,
        "failure_count": 0,
        "configs": {
            "strict_jj": {
                "skip_llm": True,
                "skip_ocr": True,
                "use_strict_jj": True,
            }
        },
        "final_42_image_test_used": False,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run_dir / "predictions" / "strict_jj" / f"{stem}.json").write_text(
        json.dumps({"components": [], "raw_groups": []}), encoding="utf-8"
    )
    (run_dir / "wiring_traces_strict_jj" / f"{stem}.json").write_text(
        json.dumps({"events": [], "summary": {}}), encoding="utf-8"
    )
    (benchmark_dir / "result" / f"{stem}_gt.txt").write_text(
        "G1: R1.1, R2.2\n", encoding="utf-8"
    )
    (benchmark_dir / "detections" / f"{stem}.json").write_text(
        json.dumps({"components": []}), encoding="utf-8"
    )
    (benchmark_dir / f"{stem}.jpg").write_bytes(b"image")
    return run_dir, benchmark_dir, stem


def test_validate_inputs_requires_exact_expected_count(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "image_count": 49,
                "failure_count": 0,
                "configs": {"strict_jj": {"skip_llm": True, "skip_ocr": True}},
                "final_42_image_test_used": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 50 images, found 49"):
        validate_inputs(run_dir, expected_count=50)


def test_validate_inputs_rejects_llm_use(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "image_count": 50,
                "failure_count": 0,
                "configs": {"strict_jj": {"skip_llm": False, "skip_ocr": True}},
                "final_42_image_test_used": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="skip_llm"):
        validate_inputs(run_dir, expected_count=50)


def test_validate_inputs_loads_exact_joined_case(tmp_path):
    run_dir, benchmark_dir, stem = _seed_valid_inputs(tmp_path)

    metadata, cases = validate_inputs(run_dir, benchmark_dir, expected_count=1)

    assert metadata["image_count"] == 1
    assert [case.stem for case in cases] == [stem]
    assert cases[0].image_path == benchmark_dir / f"{stem}.jpg"
    assert cases[0].trace_path.name == f"{stem}.json"


def test_validate_inputs_rejects_missing_trace(tmp_path):
    run_dir, benchmark_dir, stem = _seed_valid_inputs(tmp_path)
    (run_dir / "wiring_traces_strict_jj" / f"{stem}.json").unlink()

    with pytest.raises(FileNotFoundError, match="missing trace"):
        validate_inputs(run_dir, benchmark_dir, expected_count=1)


def test_validate_inputs_rejects_duplicate_image_stem(tmp_path):
    run_dir, benchmark_dir, stem = _seed_valid_inputs(tmp_path)
    (benchmark_dir / f"{stem}.png").write_bytes(b"duplicate")

    with pytest.raises(RuntimeError, match="expected one image"):
        validate_inputs(run_dir, benchmark_dir, expected_count=1)


def test_prepare_output_refuses_nonempty_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output(output, resume=False)


def test_prepare_output_creates_new_directory(tmp_path):
    output = tmp_path / "new"

    assert prepare_output(output, resume=False) == output
    assert output.is_dir()
