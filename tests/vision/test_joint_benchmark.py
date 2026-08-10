import csv
import json
from pathlib import Path

import pytest

from src.vision import unified_pipeline
from src.vision.joint_benchmark import (
    average_precision_at_iou,
    detection_metrics,
    is_electrical_value,
    normalize_benchmark_value,
    ocr_metrics_from_detections,
    parse_cghd_xml,
    point_metrics,
    port_metrics,
    run_joint_benchmark,
)
from src.vision import joint_benchmark


def test_pipeline_result_exposes_raw_detector_outputs():
    result = unified_pipeline._make_pipeline_result(
        components=[{"name": "Resistor"}],
        text_values=[{"text": "10k"}],
        junctions=[(10, 20)],
        routes=[],
        conn_pairs=[],
        raw_groups=[],
        evaluation="skipped",
        raw_junction_detections=[{"label": "junction", "bbox": [8, 18, 12, 22]}],
        raw_text_detections=[{"label": "text", "bbox": [30, 40, 50, 60], "text": "10k"}],
    )

    assert result["raw_junction_detections"][0]["label"] == "junction"
    assert result["raw_text_detections"][0]["text"] == "10k"
    assert result["components"][0]["name"] == "Resistor"
    assert result["wiring_trace"] == {"events": [], "summary": {}}


def test_detection_metrics_counts_class_aware_matches():
    ground_truth = [
        {"image": "a", "label": "Resistor", "bbox": [0, 0, 10, 10]},
        {"image": "a", "label": "Capacitor", "bbox": [20, 0, 30, 10]},
    ]
    predictions = [
        {"image": "a", "label": "Resistor", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image": "a", "label": "Resistor", "bbox": [20, 0, 30, 10], "score": 0.8},
    ]

    metrics = detection_metrics(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_detection_metrics_normalize_case_and_separator_aliases():
    ground_truth = [
        {"image": "a", "label": "microphone", "bbox": [0, 0, 10, 10]},
        {"image": "a", "label": "Zener-Diode", "bbox": [20, 0, 30, 10]},
    ]
    predictions = [
        {"image": "a", "label": "Microphone", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image": "a", "label": "Zener Diode", "bbox": [20, 0, 30, 10], "score": 0.8},
    ]

    metrics = detection_metrics(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert set(metrics["per_class"]) == {"microphone", "zener diode"}


def test_average_precision_is_one_for_perfect_ranked_detections():
    ground_truth = [
        {"image": "a", "label": "text", "bbox": [0, 0, 10, 10]},
        {"image": "b", "label": "text", "bbox": [5, 5, 15, 15]},
    ]
    predictions = [
        {"image": "a", "label": "text", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image": "b", "label": "text", "bbox": [5, 5, 15, 15], "score": 0.8},
    ]

    assert average_precision_at_iou(predictions, ground_truth, 0.5) == pytest.approx(1.0)


def test_point_metrics_use_one_to_one_radius_matching():
    metrics = point_metrics(
        predictions=[(1, 1), (50, 50)],
        ground_truth=[(0, 0), (100, 100)],
        radius=5,
    )

    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("10k", "10kΩ"),
        ("8ohm", "8Ω"),
        ("1meg.", "1MΩ"),
        ("4.7uF", "4.7μF"),
        (" 12 V ", "12V"),
    ],
)
def test_benchmark_value_normalization_handles_cghd_notation(raw, normalized):
    assert normalize_benchmark_value(raw) == normalized
    assert is_electrical_value(raw)


def test_identifiers_are_not_treated_as_electrical_values():
    assert not is_electrical_value("TL431")
    assert not is_electrical_value("2N4401")
    assert not is_electrical_value("earphone")


def test_parse_cghd_xml_returns_junctions_and_text(tmp_path):
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<annotation>
        <object><name>junction</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>14</xmax><ymax>24</ymax></bndbox></object>
        <object><name>text</name><bndbox><xmin>30</xmin><ymin>40</ymin><xmax>60</xmax><ymax>55</ymax></bndbox><text>10k</text></object>
        <object><name>terminal</name><bndbox><xmin>70</xmin><ymin>80</ymin><xmax>74</xmax><ymax>84</ymax></bndbox></object>
        <object><name>resistor</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )

    parsed = parse_cghd_xml(xml_path)

    assert parsed["junctions"] == [{"label": "junction", "bbox": [10, 20, 14, 24]}]
    assert parsed["texts"] == [{"label": "text", "bbox": [30, 40, 60, 55], "text": "10k"}]
    assert parsed["terminals"] == [{"label": "terminal", "bbox": [70, 80, 74, 84]}]


def test_ocr_metrics_count_missed_text_and_value_subset():
    ground_truth = [
        {"image": "a", "label": "text", "bbox": [0, 0, 20, 10], "text": "10k"},
        {"image": "a", "label": "text", "bbox": [30, 0, 50, 10], "text": "TL431"},
    ]
    predictions = [
        {"image": "a", "label": "text", "bbox": [0, 0, 20, 10], "text": "10kΩ", "score": 0.9},
    ]

    metrics, rows = ocr_metrics_from_detections(predictions, ground_truth, iou_threshold=0.3)

    assert metrics["all_text"]["sample_count"] == 2
    assert metrics["all_text"]["normalized_exact"] == pytest.approx(0.5)
    assert metrics["value_text"]["sample_count"] == 1
    assert metrics["value_text"]["normalized_exact"] == pytest.approx(1.0)
    assert [row["prediction"] for row in rows] == ["10kΩ", ""]


def test_port_metrics_match_components_then_ports_by_distance():
    gt_components = [
        {
            "name": "Resistor",
            "xyxy": [0, 0, 20, 10],
            "ports": [[0, 5], [20, 5]],
            "labels": ["1", "2"],
        }
    ]
    predicted_components = [
        {
            "name": "Resistor",
            "xyxy": [0, 0, 20, 10],
            "ports": [(1, 5), (19, 5)],
        }
    ]

    metrics = port_metrics(predicted_components, gt_components, radius=3)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["label_correct"] == 2
    assert metrics["label_accuracy"] == pytest.approx(1.0)


def test_port_metrics_count_ports_on_unmatched_predicted_components_as_false_positives():
    gt_components = [
        {"name": "Resistor", "xyxy": [0, 0, 20, 10], "ports": [[0, 5], [20, 5]], "labels": ["1", "2"]}
    ]
    predicted_components = [
        {"name": "Resistor", "xyxy": [0, 0, 20, 10], "ports": [[0, 5], [20, 5]]},
        {"name": "Capacitor", "xyxy": [40, 0, 60, 10], "ports": [[40, 5], [60, 5]]},
    ]

    metrics = port_metrics(predicted_components, gt_components, radius=3)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 2
    assert metrics["fn"] == 0


def test_port_label_accuracy_uses_spatial_assignment_when_port_order_is_swapped():
    gt_components = [
        {"name": "Diode", "xyxy": [0, 0, 20, 10], "ports": [[0, 5], [20, 5]], "labels": ["A", "K"]}
    ]
    predicted_components = [
        {"name": "Diode", "xyxy": [0, 0, 20, 10], "ports": [[20, 5], [0, 5]], "labels": ["A", "K"]}
    ]

    metrics = port_metrics(predicted_components, gt_components, radius=3)

    assert metrics["tp"] == 2
    assert metrics["label_compared"] == 2
    assert metrics["label_correct"] == 0


def _seed_joint_case(root: Path, *, include_xml: bool = True) -> tuple[Path, Path]:
    benchmark = root / "benchmark"
    cghd = root / "cghd" / "drafter_1" / "annotations"
    (benchmark / "fixed").mkdir(parents=True)
    (benchmark / "result").mkdir()
    (benchmark / "sample.jpg").write_bytes(b"not-used-when-render-disabled")
    (benchmark / "fixed" / "sample.json").write_text(
        """{"width": 100, "height": 100, "components": [
        {"name": "Resistor", "xyxy": [0, 0, 20, 10],
         "ports": [[0, 5], [20, 5]], "labels": ["1", "2"],
         "designator": "R1"}
        ]}""",
        encoding="utf-8",
    )
    (benchmark / "result" / "sample_gt.txt").write_text(
        "G1: R1.1, R1.2\n", encoding="utf-8"
    )
    if include_xml:
        cghd.mkdir(parents=True)
        (cghd / "sample.xml").write_text(
            """<annotation>
            <object><name>junction</name><bndbox><xmin>48</xmin><ymin>48</ymin><xmax>52</xmax><ymax>52</ymax></bndbox></object>
            <object><name>text</name><bndbox><xmin>25</xmin><ymin>0</ymin><xmax>45</xmax><ymax>10</ymax></bndbox><text>10k</text></object>
            </annotation>""",
            encoding="utf-8",
        )
    return benchmark, root / "cghd"


def test_joint_runner_forces_llm_off_and_writes_isolated_outputs(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    calls = []

    def fake_process(_image_path, config):
        calls.append(config)
        return {
            "components": [
                {
                    "name": "Resistor",
                    "xyxy": [0, 0, 20, 10],
                    "ports": [(0, 5), (20, 5)],
                    "conf": 0.9,
                    "value": "10kΩ",
                }
            ],
            "text_values": [{"text": "10kΩ", "xyxy": [25, 0, 45, 10]}],
            "junctions": [(50, 50)],
            "routes": [],
            "conn_pairs": [],
            "raw_groups": [[(0, 0), (0, 1)]],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [
                {"label": "junction", "bbox": [48, 48, 52, 52], "score": 0.9}
            ],
            "raw_text_detections": [
                {"label": "text", "bbox": [25, 0, 45, 10], "score": 0.9, "text": "10kΩ"}
            ],
        }

    output = tmp_path / "output"
    summary = run_joint_benchmark(
        benchmark_dir=benchmark,
        cghd_root=cghd_root,
        output_dir=output,
        ocr_model_path="model.pt",
        process_fn=fake_process,
        render=False,
    )

    assert len(calls) == 1
    assert calls[0]["skip_llm"] is True
    assert calls[0]["save_artifacts"] is False
    assert calls[0]["ocr_model_path"] == "model.pt"
    assert summary["image_count"] == 1
    assert summary["failed_images"] == 0
    assert (output / "summary.json").is_file()
    assert (output / "per_image_results.csv").is_file()
    assert (output / "predictions" / "sample.json").is_file()
    assert not (benchmark / "sample_wired.jpg").exists()


def test_joint_runner_refuses_nonempty_output_directory(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_joint_benchmark(
            benchmark_dir=benchmark,
            cghd_root=cghd_root,
            output_dir=output,
            ocr_model_path="model.pt",
            process_fn=lambda *_args, **_kwargs: {},
            render=False,
        )

    assert (output / "user-file.txt").read_text(encoding="utf-8") == "preserve"


def test_joint_runner_refuses_missing_xml_ground_truth(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path, include_xml=False)

    with pytest.raises(FileNotFoundError, match="sample.*XML"):
        run_joint_benchmark(
            benchmark_dir=benchmark,
            cghd_root=cghd_root,
            output_dir=tmp_path / "output",
            ocr_model_path="model.pt",
            process_fn=lambda *_args, **_kwargs: {},
            render=False,
        )


def test_joint_runner_resume_reuses_saved_predictions(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    output = tmp_path / "output"

    def first_process(_image_path, config):
        return {
            "components": [],
            "text_values": [],
            "junctions": [],
            "routes": [],
            "conn_pairs": [],
            "raw_groups": [],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [],
            "raw_text_detections": [],
        }

    run_joint_benchmark(
        benchmark_dir=benchmark,
        cghd_root=cghd_root,
        output_dir=output,
        ocr_model_path="model.pt",
        process_fn=first_process,
        render=False,
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("cached prediction should be reused")

    summary = run_joint_benchmark(
        benchmark_dir=benchmark,
        cghd_root=cghd_root,
        output_dir=output,
        ocr_model_path="model.pt",
        process_fn=must_not_run,
        render=False,
        resume=True,
    )

    assert summary["image_count"] == 1
    assert summary["failed_images"] == 0


def test_joint_runner_resume_rejects_changed_model_provenance(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    output = tmp_path / "output"
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    def fake_process(_image_path, config):
        return {
            "components": [], "text_values": [], "junctions": [], "routes": [],
            "conn_pairs": [], "raw_groups": [],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [], "raw_text_detections": [],
        }

    run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root, output_dir=output,
        ocr_model_path=model_a, process_fn=fake_process, render=False,
    )

    with pytest.raises(RuntimeError, match="provenance"):
        run_joint_benchmark(
            benchmark_dir=benchmark, cghd_root=cghd_root, output_dir=output,
            ocr_model_path=model_b, process_fn=fake_process, render=False, resume=True,
        )


def test_joint_runner_imports_verified_prediction_cache_into_fresh_output(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    cache_output = tmp_path / "cache-output"
    fresh_output = tmp_path / "fresh-output"

    def first_process(_image_path, config):
        return {
            "components": [], "text_values": [], "junctions": [], "routes": [],
            "conn_pairs": [], "raw_groups": [],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [], "raw_text_detections": [],
        }

    run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root, output_dir=cache_output,
        ocr_model_path=model, process_fn=first_process, render=False,
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("verified external cache should be reused")

    summary = run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root, output_dir=fresh_output,
        ocr_model_path=model, process_fn=must_not_run, render=False,
        prediction_cache_dir=cache_output,
    )

    assert summary["failed_images"] == 0
    assert (fresh_output / "predictions" / "sample.json").is_file()
    metadata = json.loads(
        (fresh_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["provenance"]["prediction_cache"]["source_metadata_sha256"]


def test_joint_runner_enforces_expected_case_count(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)

    with pytest.raises(RuntimeError, match="expected 50.*found 1"):
        run_joint_benchmark(
            benchmark_dir=benchmark, cghd_root=cghd_root,
            output_dir=tmp_path / "output", ocr_model_path="model.pt",
            process_fn=lambda *_args, **_kwargs: {}, render=False,
            expected_case_count=50,
        )


def test_render_failure_does_not_duplicate_evaluation_rows(tmp_path, monkeypatch):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    output = tmp_path / "output"

    def fake_process(_image_path, config):
        return {
            "components": [], "text_values": [], "junctions": [], "routes": [],
            "conn_pairs": [], "raw_groups": [],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [], "raw_text_detections": [],
        }

    monkeypatch.setattr(joint_benchmark, "_render_prediction", lambda *_args: (_ for _ in ()).throw(RuntimeError("render boom")))
    summary = run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root, output_dir=output,
        ocr_model_path="model.pt", process_fn=fake_process, render=True,
    )

    with (output / "per_image_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "render_failed"
    assert summary["failed_images"] == 0
    assert summary["render_failed_images"] == 1


def test_renderer_supports_unicode_absolute_paths(tmp_path):
    import cv2
    import numpy as np

    source = tmp_path / "中文输入" / "图像.jpg"
    output = tmp_path / "中文输出" / "检测框.jpg"
    source.parent.mkdir()
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 30, 3), dtype=np.uint8))
    assert ok
    source.write_bytes(encoded.tobytes())

    joint_benchmark._render_prediction(
        source,
        {
            "components": [], "junctions": [],
            "raw_junction_detections": [], "raw_text_detections": [],
        },
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_failed_pipeline_counts_all_ground_truth_wiring_edges_as_false_negatives(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)

    def broken_process(_image_path, config):
        raise RuntimeError("inference failed")

    summary = run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root,
        output_dir=tmp_path / "output", ocr_model_path="model.pt",
        process_fn=broken_process, render=False,
    )

    assert summary["failed_images"] == 1
    assert summary["wiring"]["edge"]["tp"] == 0
    assert summary["wiring"]["edge"]["fn"] == 1


def test_strict_success_treats_image_without_value_gt_as_not_applicable(tmp_path):
    benchmark, cghd_root = _seed_joint_case(tmp_path)
    xml = cghd_root / "drafter_1" / "annotations" / "sample.xml"
    xml.write_text("<annotation></annotation>", encoding="utf-8")

    def perfect_process(_image_path, config):
        return {
            "components": [{"name": "Resistor", "xyxy": [0, 0, 20, 10], "ports": [[0, 5], [20, 5]], "labels": ["1", "2"], "conf": 1.0}],
            "text_values": [], "junctions": [], "routes": [], "conn_pairs": [],
            "raw_groups": [[(0, 0), (0, 1)]],
            "evaluation": "(LLM skipped for experiment)",
            "raw_junction_detections": [], "raw_text_detections": [],
        }

    summary = run_joint_benchmark(
        benchmark_dir=benchmark, cghd_root=cghd_root,
        output_dir=tmp_path / "output", ocr_model_path="model.pt",
        process_fn=perfect_process, render=False,
    )

    assert summary["strict_end_to_end_success_count"] == 1
