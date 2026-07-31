import json

import run_experiments
from run_experiments import build_ablation_configs, resolve_output_dir
from src.vision.unified_pipeline import DEFAULT_CONFIG


def test_ablation_configs_are_full_unique_and_llm_free():
    configs = build_ablation_configs()

    assert set(configs) == {
        "Ours",
        "Baseline",
        "w/o_Skeleton",
        "w/o_Sobel",
        "w/o_NN_Filter",
        "w/o_Close_Port",
        "CCL",
    }
    assert configs["Ours"] == {**DEFAULT_CONFIG, "skip_llm": True}
    assert configs["Baseline"] == {
        **{key: False for key in DEFAULT_CONFIG},
        "skip_llm": True,
    }
    assert configs["w/o_Skeleton"]["use_skeleton"] is False
    assert configs["w/o_Sobel"]["use_sobel"] is False
    assert configs["w/o_NN_Filter"]["use_nn_filter"] is False
    assert configs["w/o_Close_Port"]["use_close_port"] is False
    assert configs["CCL"]["use_ccl"] is True
    assert all(set(config) == set(DEFAULT_CONFIG) for config in configs.values())
    assert all(config["skip_llm"] is True for config in configs.values())

    fingerprints = {tuple(sorted(config.items())) for config in configs.values()}
    assert len(fingerprints) == len(configs)


def test_output_directory_is_explicit_and_created(tmp_path):
    output_dir = resolve_output_dir(tmp_path / "visual-only-run")

    assert output_dir.is_dir()
    assert output_dir == tmp_path / "visual-only-run"


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
    assert all(
        (tmp_path / artifact_name).read_text(encoding="utf-8") == "historical artifact"
        for artifact_name in artifact_names
    )
