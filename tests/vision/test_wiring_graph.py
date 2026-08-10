import numpy as np

from src.vision.wiring_graph import (
    WiringTrace,
    accept_p2j_candidate,
    classify_connection_detection,
    network_color,
    strict_jj_decision,
    terminal_component,
    trace_port_to_anchor,
)


def test_wiring_trace_counts_acceptance_by_stage_and_reason():
    trace = WiringTrace(enabled=True)
    trace.record("p2j", "p2j", True, "continuous_skeleton_path")
    trace.record("p2j", "p2j", False, "no_skeleton_path")

    assert trace.summary() == {
        "p2j": {
            "candidates": 2,
            "accepted": 1,
            "rejected": 1,
            "reasons": {
                "continuous_skeleton_path": 1,
                "no_skeleton_path": 1,
            },
        }
    }


def test_disabled_wiring_trace_records_nothing():
    trace = WiringTrace(enabled=False)
    trace.record("p2j", "p2j", True, "continuous_skeleton_path")

    assert trace.to_dict() == {"events": [], "summary": {}}


def test_network_color_is_stable_and_network_specific():
    first = network_color(["C1.+", "R1.1"])

    assert first == network_color(["R1.1", "C1.+"])
    assert first != network_color(["C1.-", "R1.2"])
    assert all(64 <= channel <= 223 for channel in first)


def test_terminal_component_has_one_center_port_and_stable_semantics():
    component = terminal_component([10, 20, 30, 40], 0.8, index=4)

    assert component["idx"] == 4
    assert component["name"] == "Terminal"
    assert component["ports"] == [(20, 30)]
    assert component["labels"] == ["T"]
    assert component["raw_name"] == "terminal"


def test_terminal_detection_is_not_added_to_junction_centers_when_enabled():
    classified = classify_connection_detection(
        "terminal", [10, 20, 30, 40], 0.8, use_terminal_components=True, index=0
    )

    assert classified["junction"] is None
    assert classified["component"]["name"] == "Terminal"


def test_terminal_detection_preserves_legacy_junction_behavior_when_disabled():
    classified = classify_connection_detection(
        "terminal", [10, 20, 30, 40], 0.8, use_terminal_components=False, index=0
    )

    assert classified == {"junction": (20, 30), "component": None}


def test_junction_detection_never_becomes_a_component():
    classified = classify_connection_detection(
        "junction", [10, 20, 30, 40], 0.8, use_terminal_components=True, index=0
    )

    assert classified == {"junction": (20, 30), "component": None}


def test_pipeline_defaults_enable_terminal_components_with_one_port():
    from src.vision import unified_pipeline

    assert unified_pipeline.DEFAULT_CONFIG["use_terminal_components"] is True
    assert unified_pipeline.PORT_LABELS["Terminal"] == ["T"]
    assert unified_pipeline.PORT_POSITIONS["Terminal"] == [(0.5, 0.5)]
    assert unified_pipeline.DESIG["Terminal"] == "T"


def test_publication_selected_defaults_match_best_full50_configuration():
    from src.vision import unified_pipeline

    assert unified_pipeline.DEFAULT_CONFIG["use_strict_p2j"] is True
    assert unified_pipeline.DEFAULT_CONFIG["use_outward_skeleton_trace"] is True
    assert unified_pipeline.DEFAULT_CONFIG["use_strict_jj"] is True
    assert unified_pipeline.DEFAULT_CONFIG["use_crossing_semantics"] is False


def test_pipeline_exposes_strict_jj_feature_flags():
    from src.vision import unified_pipeline

    assert unified_pipeline.DEFAULT_CONFIG["use_strict_jj"] is True
    assert unified_pipeline.DEFAULT_CONFIG["use_crossing_semantics"] is False


def test_terminal_never_accepts_an_ocr_value():
    from src.vision.unified_pipeline import _component_accepts_value

    assert _component_accepts_value({"name": "Terminal"}) is False
    assert _component_accepts_value({"name": "GND"}) is False
    assert _component_accepts_value({"name": "Resistor"}) is True


def test_strict_p2j_rejects_distance_only_candidate():
    decision = accept_p2j_candidate(
        distance=20,
        path_found=False,
        crosses_component=False,
        strict=True,
    )

    assert decision == (False, "no_skeleton_path")


def test_legacy_p2j_keeps_distance_candidate_for_ablation():
    decision = accept_p2j_candidate(
        distance=20,
        path_found=False,
        crosses_component=False,
        strict=False,
    )

    assert decision == (True, "legacy_distance_fallback")


def test_p2j_rejects_component_crossing_even_with_skeleton():
    decision = accept_p2j_candidate(
        distance=20,
        path_found=True,
        crosses_component=True,
        strict=True,
    )

    assert decision == (False, "crosses_component")


def test_strict_p2j_accepts_continuous_skeleton_path():
    decision = accept_p2j_candidate(
        distance=20,
        path_found=True,
        crosses_component=False,
        strict=True,
    )

    assert decision == (True, "continuous_skeleton_path")


def test_trace_port_to_anchor_returns_first_outward_anchor():
    skeleton = np.zeros((21, 41), dtype=np.uint8)
    skeleton[10, 5:36] = 255

    result = trace_port_to_anchor(
        skeleton=skeleton,
        port=(5, 10),
        component_center=(0, 10),
        anchors=[("J1", (20, 10)), ("J2", (35, 10))],
        search_radius=2,
        anchor_radius=2,
        max_steps=100,
    )

    assert result["anchor_id"] == "J1"
    assert result["anchor"] == (20, 10)
    assert result["reason"] == "continuous_skeleton_path"


def test_trace_port_to_anchor_rejects_anchor_behind_component():
    skeleton = np.zeros((21, 41), dtype=np.uint8)
    skeleton[10, 2:20] = 255

    result = trace_port_to_anchor(
        skeleton=skeleton,
        port=(10, 10),
        component_center=(0, 10),
        anchors=[("behind", (2, 10))],
        search_radius=2,
        anchor_radius=2,
        max_steps=100,
    )

    assert result is None


def test_trace_port_to_anchor_returns_none_when_port_has_no_skeleton():
    skeleton = np.zeros((21, 41), dtype=np.uint8)

    result = trace_port_to_anchor(
        skeleton=skeleton,
        port=(5, 10),
        component_center=(0, 10),
        anchors=[("J1", (20, 10))],
        search_radius=2,
        anchor_radius=2,
        max_steps=100,
    )

    assert result is None


def test_strict_jj_accepts_continuous_aligned_wire():
    skeleton = np.zeros((31, 41), dtype=np.uint8)
    skeleton[15, 5:36] = 255

    decision = strict_jj_decision(
        skeleton=skeleton,
        start=(5, 15),
        end=(35, 15),
        detected_junctions=[],
        crossing_semantics=True,
    )

    assert decision == (True, "continuous_skeleton_path")


def test_strict_jj_rejects_blank_gap():
    skeleton = np.zeros((31, 41), dtype=np.uint8)
    skeleton[15, 5:18] = 255
    skeleton[15, 23:36] = 255

    decision = strict_jj_decision(
        skeleton=skeleton,
        start=(5, 15),
        end=(35, 15),
        detected_junctions=[],
        crossing_semantics=True,
    )

    assert decision == (False, "no_skeleton_path")


def test_strict_jj_rejects_turn_through_unmarked_crossing():
    skeleton = np.zeros((41, 41), dtype=np.uint8)
    skeleton[20, 5:36] = 255
    skeleton[5:36, 20] = 255

    decision = strict_jj_decision(
        skeleton=skeleton,
        start=(5, 20),
        end=(20, 5),
        detected_junctions=[],
        crossing_semantics=True,
    )

    assert decision == (False, "ambiguous_crossing")


def test_strict_jj_accepts_turn_through_detected_crossing():
    skeleton = np.zeros((41, 41), dtype=np.uint8)
    skeleton[20, 5:36] = 255
    skeleton[5:36, 20] = 255

    decision = strict_jj_decision(
        skeleton=skeleton,
        start=(5, 20),
        end=(20, 5),
        detected_junctions=[(20, 20)],
        crossing_semantics=True,
    )

    assert decision == (True, "continuous_skeleton_path")
