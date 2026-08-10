from src.vision.wiring_graph import WiringTrace, network_color


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
