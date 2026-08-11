import json
import csv
from pathlib import Path

import pytest

import run_experiments
from run_experiments import (
    build_ablation_configs,
    build_wiring_reliability_configs,
    resolve_output_dir,
)
from src.vision.unified_pipeline import DEFAULT_CONFIG


def _seed_benchmark_case(root, stem, suffix):
    (root / "result").mkdir(exist_ok=True)
    (root / "detections").mkdir(exist_ok=True)
    (root / "fixed").mkdir(exist_ok=True)
    (root / "result" / f"{stem}_gt.txt").write_text(
        "G1: R1.1, R2.2\n", encoding="utf-8",
    )
    (root / "detections" / f"{stem}.json").write_text(
        '{"components": []}', encoding="utf-8",
    )
    (root / f"{stem}{suffix}").write_bytes(b"image")


def test_get_image_list_accepts_supported_extensions(tmp_path):
    for stem, suffix in (
        ("a", ".jpg"),
        ("b", ".JPG"),
        ("c", ".jpeg"),
        ("d", ".png"),
    ):
        _seed_benchmark_case(tmp_path, stem, suffix)

    images = run_experiments.get_image_list(tmp_path)

    assert [image[0] for image in images] == ["a", "b", "c", "d"]
    assert [Path(image[1]).suffix for image in images] == [
        ".jpg",
        ".JPG",
        ".jpeg",
        ".png",
    ]


def test_get_image_list_rejects_missing_image(tmp_path):
    _seed_benchmark_case(tmp_path, "missing", ".jpg")
    (tmp_path / "missing.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="missing.*image"):
        run_experiments.get_image_list(tmp_path)


def test_get_image_list_rejects_duplicate_stem(tmp_path):
    _seed_benchmark_case(tmp_path, "duplicate", ".jpg")
    (tmp_path / "duplicate.jpeg").write_bytes(b"duplicate image")

    with pytest.raises(RuntimeError, match="duplicate.*multiple images"):
        run_experiments.get_image_list(tmp_path)


def test_get_image_list_prefers_fixed_detection_json(tmp_path):
    _seed_benchmark_case(tmp_path, "corrected", ".jpg")
    fixed_path = tmp_path / "fixed" / "corrected.json"
    fixed_path.write_text('{"components": ["fixed"]}', encoding="utf-8")

    images = run_experiments.get_image_list(tmp_path)

    assert Path(images[0][3]) == fixed_path


def test_get_image_list_rejects_missing_detection_json(tmp_path):
    _seed_benchmark_case(tmp_path, "undetected", ".jpg")
    (tmp_path / "detections" / "undetected.json").unlink()

    with pytest.raises(FileNotFoundError, match="undetected.*detection"):
        run_experiments.get_image_list(tmp_path)


def test_get_image_list_ignores_fixed_detection_directory(tmp_path):
    _seed_benchmark_case(tmp_path, "fallback", ".jpg")
    fixed_path = tmp_path / "fixed" / "fallback.json"
    fixed_path.mkdir()

    images = run_experiments.get_image_list(tmp_path)

    assert Path(images[0][3]) == tmp_path / "detections" / "fallback.json"


def test_repository_benchmark_schedules_every_gt_file():
    benchmark_dir = Path("benchmark")
    scheduled_stems = {
        image[0] for image in run_experiments.get_image_list(benchmark_dir)
    }
    gt_stems = {
        path.stem.removesuffix("_gt")
        for path in (benchmark_dir / "result").glob("*_gt.txt")
    }

    assert scheduled_stems == gt_stems
    assert {"C170_D1_P1", "C171_D1_P1", "C274_D2_P1"} <= gt_stems


def test_ablation_configs_are_full_unique_and_llm_free():
    configs = build_ablation_configs()

    assert set(configs) == {
        "Ours",
        "Baseline",
        "w/o_Skeleton",
        "w/o_Sobel",
        "w/o_NN_Filter",
        "w/o_Close_Port",
        "w/o_Component_Mask",
        "CCL",
    }
    assert DEFAULT_CONFIG["save_artifacts"] is True
    assert configs["Ours"] == {
        **DEFAULT_CONFIG,
        "skip_llm": True,
        "save_artifacts": False,
    }
    assert configs["Baseline"] == {
        **{key: False for key in DEFAULT_CONFIG},
        "skip_llm": True,
    }
    assert configs["w/o_Skeleton"]["use_skeleton"] is False
    assert configs["w/o_Sobel"]["use_sobel"] is False
    assert configs["w/o_NN_Filter"]["use_nn_filter"] is False
    assert configs["w/o_Close_Port"]["use_close_port"] is False
    assert configs["w/o_Component_Mask"]["use_component_mask"] is False
    assert configs["CCL"]["use_ccl"] is True
    assert all(set(config) == set(DEFAULT_CONFIG) for config in configs.values())
    assert all(config["skip_llm"] is True for config in configs.values())
    assert all(config["save_artifacts"] is False for config in configs.values())

    fingerprints = {tuple(sorted(config.items())) for config in configs.values()}
    assert len(fingerprints) == len(configs)


def test_wiring_reliability_configs_keep_candidate_changes_independent_and_llm_free():
    configs = build_wiring_reliability_configs()

    assert list(configs) == [
        "frozen_baseline", "observability", "terminal", "strict_fallback",
        "outward_trace", "strict_jj", "directional_gap_bridge",
        "outward_port_anchors",
        "crossing_semantics",
    ]
    assert all(config["skip_llm"] is True for config in configs.values())
    assert all(config["skip_ocr"] is True for config in configs.values())
    assert all(config["save_artifacts"] is False for config in configs.values())
    assert configs["frozen_baseline"]["use_terminal_components"] is False
    assert configs["frozen_baseline"]["use_wiring_trace"] is False
    assert configs["observability"]["use_wiring_trace"] is True
    assert configs["terminal"]["use_terminal_components"] is True
    assert configs["strict_fallback"]["use_strict_p2j"] is True
    assert configs["outward_trace"]["use_outward_skeleton_trace"] is True
    assert configs["strict_jj"]["use_strict_jj"] is True
    assert configs["strict_jj"]["use_outward_port_anchors"] is False
    assert configs["directional_gap_bridge"]["use_directional_gap_bridge"] is True
    assert configs["outward_port_anchors"]["use_outward_port_anchors"] is True
    assert configs["crossing_semantics"]["use_crossing_semantics"] is True

    strict_jj = configs["strict_jj"]
    directional_gap_bridge = configs["directional_gap_bridge"]
    assert {
        key
        for key in strict_jj
        if strict_jj[key] != directional_gap_bridge[key]
    } == {"use_directional_gap_bridge"}
    for candidate_name, feature_name in (
        ("outward_port_anchors", "use_outward_port_anchors"),
        ("crossing_semantics", "use_crossing_semantics"),
    ):
        candidate = configs[candidate_name]
        assert {
            key for key in strict_jj if strict_jj[key] != candidate[key]
        } == {feature_name}


def test_output_directory_is_explicit_and_created(tmp_path):
    output_dir = resolve_output_dir(tmp_path / "visual-only-run")

    assert output_dir.is_dir()
    assert output_dir == tmp_path / "visual-only-run"


def test_default_output_directories_are_unique_and_explicit_nonempty_dirs_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    first_output_dir = resolve_output_dir()
    second_output_dir = resolve_output_dir()

    assert first_output_dir.is_dir()
    assert second_output_dir.is_dir()
    assert first_output_dir != second_output_dir

    occupied_dir = tmp_path / "occupied-output"
    occupied_dir.mkdir()
    (occupied_dir / "existing.csv").write_text("existing run", encoding="utf-8")
    with pytest.raises(FileExistsError):
        resolve_output_dir(occupied_dir)


def test_main_writes_all_outputs_to_requested_directory_only(tmp_path, monkeypatch):
    artifact_names = {
        "experiment_results.csv",
        "experiment_summary.png",
        "ablation_impact.png",
        "run_metadata.json",
    }
    for artifact_name in artifact_names:
        (tmp_path / artifact_name).write_text("historical artifact", encoding="utf-8")

    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps({"components": [
        {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]], "labels": ["1"], "designator": "R1"},
        {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]], "labels": ["1"], "designator": "R2"},
    ]}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_experiments,
        "get_image_list",
        lambda: [("img_000", "unused.jpg", "unused_gt.txt", str(detections_path))],
    )
    monkeypatch.setattr(
        run_experiments,
        "parse_gt",
        lambda _path: [[("R1", "1"), ("R2", "1")]],
    )

    def fake_process_image(_image_path, config):
        assert config["skip_llm"] is True
        assert config["save_artifacts"] is False
        return {
            "components": [
                {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]},
                {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]]},
            ],
            "raw_groups": [{(0, 0), (1, 0)}],
        }

    monkeypatch.setattr(run_experiments, "process_image", fake_process_image)
    output_dir = tmp_path / "requested-output"

    run_experiments.main(output_dir)

    assert {path.name for path in output_dir.iterdir()} == artifact_names
    assert all((output_dir / artifact_name).is_file() for artifact_name in artifact_names)
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_count"] == 1
    assert all(config["skip_llm"] is True for config in metadata["configs"].values())
    assert all(config["save_artifacts"] is False for config in metadata["configs"].values())
    assert "git_revision" in metadata
    if revision := run_experiments.get_git_revision():
        assert metadata["git_revision"] == revision
    assert all(
        (tmp_path / artifact_name).read_text(encoding="utf-8") == "historical artifact"
        for artifact_name in artifact_names
    )


def test_wiring_reliability_runner_writes_edge_and_trace_outputs(tmp_path, monkeypatch):
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps({"components": [
        {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]], "labels": ["1"], "designator": "R1"},
        {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]], "labels": ["1"], "designator": "R2"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(
        run_experiments, "get_image_list",
        lambda _benchmark_dir: [("sample", "unused.jpg", "unused_gt.txt", str(detections_path))],
    )
    monkeypatch.setattr(
        run_experiments, "parse_gt",
        lambda _path: [[("R1", "1"), ("R2", "1")]],
    )

    def fake_process(_image_path, config):
        return {
            "components": [
                {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]},
                {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]]},
            ],
            "raw_groups": [[[0, 0], [1, 0]]],
            "evaluation": "(LLM skipped for experiment)",
            "wiring_trace": {
                "events": [],
                "summary": {"p2j": {"candidates": 2, "accepted": 2, "rejected": 0, "reasons": {"test": 2}}},
            },
        }

    output = run_experiments.run_wiring_reliability_experiment(
        output_dir=tmp_path / "wiring-run",
        benchmark_dir=tmp_path,
        selected_images=["sample"],
        expected_count=1,
        process_fn=fake_process,
    )

    with (output / "experiment_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(build_wiring_reliability_configs())
    assert all(float(row["edge_f1"]) == pytest.approx(1.0) for row in rows)
    assert json.loads(rows[-1]["stage_summary"])["p2j"]["accepted"] == 2
    summary = json.loads((output / "wiring_reliability_summary.json").read_text(encoding="utf-8"))
    assert summary["crossing_semantics"]["edge"]["f1"] == pytest.approx(1.0)
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["final_42_image_test_used"] is False
    assert metadata["expected_case_count"] == 1


def test_wiring_reliability_runner_selected_configs_preserve_requested_order(
    tmp_path, monkeypatch,
):
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(json.dumps({"components": [
        {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]], "labels": ["1"], "designator": "R1"},
        {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]], "labels": ["1"], "designator": "R2"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(
        run_experiments, "get_image_list",
        lambda _benchmark_dir: [("sample", "unused.jpg", "unused_gt.txt", str(detections_path))],
    )
    monkeypatch.setattr(
        run_experiments, "parse_gt",
        lambda _path: [[("R1", "1"), ("R2", "1")]],
    )
    seen = []

    def fake_process(_image_path, config):
        seen.append((config["use_strict_jj"], config["use_crossing_semantics"]))
        return {
            "components": [
                {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]},
                {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]]},
            ],
            "raw_groups": [[[0, 0], [1, 0]]],
            "evaluation": "(LLM skipped for experiment)",
            "wiring_trace": {"events": [], "summary": {}},
        }

    output = run_experiments.run_wiring_reliability_experiment(
        output_dir=tmp_path / "focused-run",
        benchmark_dir=tmp_path,
        selected_images=["sample"],
        selected_configs=["strict_jj", "crossing_semantics"],
        expected_count=1,
        process_fn=fake_process,
    )

    with (output / "experiment_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["config"] for row in rows] == ["strict_jj", "crossing_semantics"]
    assert seen == [(True, False), (True, True)]
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["configs"]) == {"strict_jj", "crossing_semantics"}
    assert metadata["selected_config_names"] == ["strict_jj", "crossing_semantics"]


def test_wiring_reliability_runner_rejects_unknown_config_before_output(tmp_path):
    output = tmp_path / "unknown-config"

    with pytest.raises(ValueError, match="unknown wiring reliability config"):
        run_experiments.run_wiring_reliability_experiment(
            output_dir=output,
            selected_configs=["not-a-config"],
        )

    assert not output.exists()


def test_merge_wiring_reliability_shards_requires_exact_unique_coverage(tmp_path):
    configs = build_wiring_reliability_configs()

    def seed_shard(name, stem):
        shard = tmp_path / name
        shard.mkdir()
        (shard / "run_metadata.json").write_text(json.dumps({
            "suite": "wiring-reliability", "git_revision": "abc",
            "configs": configs, "image_count": 1,
            "ocr_model_sha256": "ocr", "detector_model_sha256": "det",
            "final_42_image_test_used": False,
        }), encoding="utf-8")
        columns = [
            "image", "config", "n_components", "n_gt_groups", "n_pred_groups",
            "edge_tp", "edge_fp", "edge_fn", "edge_precision", "edge_recall", "edge_f1",
            "port_correct_rate", "fp_rate", "fn_rate", "group_accuracy",
            "comp_neighbor_accuracy", "stage_summary", "error",
        ]
        with (shard / "experiment_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for config_name in configs:
                writer.writerow({
                    "image": stem, "config": config_name, "n_components": 2,
                    "n_gt_groups": 1, "n_pred_groups": 1,
                    "edge_tp": 1, "edge_fp": 0, "edge_fn": 0,
                    "edge_precision": 1, "edge_recall": 1, "edge_f1": 1,
                    "port_correct_rate": 1, "fp_rate": 0, "fn_rate": 0,
                    "group_accuracy": 1, "comp_neighbor_accuracy": 1,
                    "stage_summary": "{}", "error": "",
                })
                prediction = shard / "predictions" / config_name / f"{stem}.json"
                prediction.parent.mkdir(parents=True, exist_ok=True)
                prediction.write_text("{}", encoding="utf-8")
        (shard / "failures.json").write_text("[]", encoding="utf-8")
        return shard

    shard_a = seed_shard("shard-a", "a")
    shard_b = seed_shard("shard-b", "b")
    output = run_experiments.merge_wiring_reliability_shards(
        [shard_a, shard_b], tmp_path / "merged", expected_count=2,
    )

    with (output / "experiment_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(configs)
    assert {row["image"] for row in rows} == {"a", "b"}
    assert (output / "predictions" / "crossing_semantics" / "b.json").is_file()
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_count"] == 2
    assert metadata["shard_count"] == 2

    with pytest.raises(RuntimeError, match="duplicate"):
        run_experiments.merge_wiring_reliability_shards(
            [shard_a, shard_a], tmp_path / "duplicate-merge", expected_count=2,
        )


def test_recover_complete_wiring_predictions_ignores_incomplete_images(tmp_path):
    benchmark = tmp_path / "benchmark"
    partial = tmp_path / "partial"
    benchmark.mkdir()
    components = [
        {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]], "labels": ["1"], "designator": "R1"},
        {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]], "labels": ["1"], "designator": "R2"},
    ]
    prediction = {
        "components": [
            {"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]},
            {"xyxy": [20, 0, 30, 10], "ports": [[20, 5]]},
        ],
        "raw_groups": [[[0, 0], [1, 0]]],
        "evaluation": "(LLM skipped for experiment)",
        "wiring_trace": {"events": [], "summary": {}},
    }
    for stem in ("a", "b"):
        _seed_benchmark_case(benchmark, stem, ".jpg")
        (benchmark / "detections" / f"{stem}.json").write_text(
            json.dumps({"components": components}), encoding="utf-8",
        )
    configs = build_wiring_reliability_configs()
    for config_name in configs:
        path = partial / "predictions" / config_name / "a.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prediction), encoding="utf-8")
    for config_name in list(configs)[:-1]:
        path = partial / "predictions" / config_name / "b.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prediction), encoding="utf-8")

    output = run_experiments.recover_complete_wiring_predictions(
        partial, tmp_path / "recovered", benchmark_dir=benchmark,
    )

    with (output / "experiment_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(configs)
    assert {row["image"] for row in rows} == {"a"}
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["image_count"] == 1
    assert metadata["recovered_from_partial_predictions"] is True
