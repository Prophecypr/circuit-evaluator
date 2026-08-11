import json
from pathlib import Path

import pytest

from src.vision.wiring_error_cli import (
    _select_render_error_rows,
    build_parser,
    prepare_output,
    reconcile_counts,
    run_analysis,
    validate_inputs,
)


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


def _seed_analysis_inputs(tmp_path):
    import cv2
    import numpy as np

    run_dir, benchmark_dir, stem = _seed_valid_inputs(tmp_path)
    components = []
    detections = []
    definitions = [
        ("R1", 10, 20),
        ("R2", 35, 20),
        ("C1", 10, 70),
        ("C2", 35, 70),
    ]
    for designator, x, y in definitions:
        components.append(
            {
                "xyxy": [x - 4, y - 4, x + 4, y + 4],
                "ports": [[x, y]],
                "designator": designator,
            }
        )
        detections.append(
            {
                "xyxy": [x - 4, y - 4, x + 4, y + 4],
                "ports": [[x, y]],
                "labels": ["1"],
                "designator": designator,
            }
        )
    prediction = {
        "components": components,
        "raw_groups": [[[0, 0], [1, 0]], [[1, 0], [2, 0]]],
        "evaluation": "(LLM skipped for experiment)",
    }
    trace = {
        "events": [
            {
                "accepted": True,
                "kind": "p2p",
                "stage": "los",
                "reason": "skeleton_supported",
                "source": {"component_index": 1, "port_index": 0},
                "target": {"component_index": 2, "port_index": 0},
                "evidence": {},
            }
        ],
        "summary": {},
    }
    (run_dir / "predictions" / "strict_jj" / f"{stem}.json").write_text(
        json.dumps(prediction), encoding="utf-8"
    )
    (run_dir / "wiring_traces_strict_jj" / f"{stem}.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )
    (benchmark_dir / "detections" / f"{stem}.json").write_text(
        json.dumps({"components": detections}), encoding="utf-8"
    )
    (benchmark_dir / "result" / f"{stem}_gt.txt").write_text(
        "G1: R1.1, R2.1\nG2: C1.1, C2.1\n", encoding="utf-8"
    )
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    (benchmark_dir / f"{stem}.jpg").write_bytes(encoded.tobytes())
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


def test_reconcile_counts_rejects_any_metric_drift():
    image_rows = [{"edge_tp": 1}]
    error_rows = [
        {"error_type": "FP"},
        {"error_type": "FN"},
    ]

    assert reconcile_counts(
        image_rows,
        error_rows,
        {"tp": 1, "fp": 1, "fn": 1},
    ) == {"tp": 1, "fp": 1, "fn": 1}
    with pytest.raises(RuntimeError, match="edge reconciliation failed"):
        reconcile_counts(
            image_rows,
            error_rows,
            {"tp": 1, "fp": 2, "fn": 1},
        )


def test_run_analysis_writes_reconciled_artifacts_under_unicode_path(tmp_path):
    run_dir, benchmark_dir, _ = _seed_analysis_inputs(tmp_path)
    output_dir = tmp_path / "中文诊断" / "analysis"

    output = run_analysis(
        run_dir,
        benchmark_dir,
        output_dir,
        expected_count=1,
        expected_counts={"tp": 1, "fp": 1, "fn": 1},
        worst_count=1,
    )

    assert (output / "edge_errors.csv").is_file()
    assert (output / "image_summary.csv").is_file()
    assert (output / "category_summary.json").is_file()
    assert (output / "wiring_error_report.md").is_file()
    assert (output / "run_metadata.json").is_file()
    assert len(list((output / "annotated_worst10").glob("*.jpg"))) == 1
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["actual_counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert metadata["llm_used"] is False
    assert metadata["final_42_image_test_used"] is False


def test_resume_rejects_matching_provenance_with_incomplete_artifacts(tmp_path):
    run_dir, benchmark_dir, _ = _seed_analysis_inputs(tmp_path)
    output_dir = tmp_path / "analysis"
    run_analysis(
        run_dir,
        benchmark_dir,
        output_dir,
        expected_count=1,
        expected_counts={"tp": 1, "fp": 1, "fn": 1},
        worst_count=1,
    )
    (output_dir / "edge_errors.csv").unlink()

    with pytest.raises(RuntimeError, match="resume output is incomplete"):
        run_analysis(
            run_dir,
            benchmark_dir,
            output_dir,
            expected_count=1,
            expected_counts={"tp": 1, "fp": 1, "fn": 1},
            worst_count=1,
            resume=True,
        )


def test_cli_defaults_to_sealed_safe_local_analysis():
    args = build_parser().parse_args([])

    assert args.run_dir == Path(
        "results/wiring_reliability_full_20260810_merged"
    )
    assert args.benchmark_dir == Path("benchmark")
    assert args.output_dir == Path("results/wiring_error_attribution_20260811")
    assert args.expected_count == 50
    assert args.expected_tp == 305
    assert args.expected_fp == 537
    assert args.expected_fn == 1002


def test_render_selection_hides_cascade_clique_and_deduplicates_fn_categories():
    rows = [
        {
            "error_type": "FP",
            "category": "cascade_fp",
            "is_root": False,
            "gt_network_a": "N1",
            "port_a": "A.1",
            "port_b": "B.1",
        },
        {
            "error_type": "FP",
            "category": "wrong_junction_merge",
            "is_root": True,
            "gt_network_a": "N1",
            "port_a": "A.1",
            "port_b": "C.1",
        },
        {
            "error_type": "FN",
            "category": "skeleton_break",
            "is_root": True,
            "gt_network_a": "N2",
            "port_a": "D.1",
            "port_b": "E.1",
        },
        {
            "error_type": "FN",
            "category": "skeleton_break",
            "is_root": True,
            "gt_network_a": "N2",
            "port_a": "D.1",
            "port_b": "F.1",
        },
    ]

    selected = _select_render_error_rows(rows)

    assert [(row["error_type"], row["category"], row["port_b"]) for row in selected] == [
        ("FN", "skeleton_break", "E.1"),
        ("FP", "wrong_junction_merge", "C.1"),
    ]
