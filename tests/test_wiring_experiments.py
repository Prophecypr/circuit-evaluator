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
