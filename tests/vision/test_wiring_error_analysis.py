import numpy as np
import pytest

from src.vision.wiring_error_analysis import (
    FnEvidence,
    attribute_fn_edges,
    classify_fn,
    index_trace_events,
    skeleton_near_port,
)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (FnEvidence(component_matched=False), "component_unmatched"),
        (
            FnEvidence(component_matched=True, port_mapped=False),
            "port_unmatched",
        ),
        (
            FnEvidence(
                component_matched=True,
                port_mapped=True,
                skeleton_near_port=False,
            ),
            "no_port_skeleton",
        ),
        (
            FnEvidence(
                component_matched=True,
                port_mapped=True,
                skeleton_near_port=True,
                trace_reached_network=False,
            ),
            "skeleton_break",
        ),
        (
            FnEvidence(
                component_matched=True,
                port_mapped=True,
                skeleton_near_port=True,
                trace_reached_network=True,
                rejection_reason="ambiguous_crossing",
            ),
            "candidate_rejected",
        ),
        (
            FnEvidence(
                component_matched=True,
                port_mapped=True,
                skeleton_near_port=True,
                trace_reached_network=True,
                candidate_generated=False,
            ),
            "candidate_not_generated",
        ),
        (
            FnEvidence(
                component_matched=True,
                port_mapped=True,
                skeleton_near_port=True,
                trace_reached_network=True,
                candidate_generated=True,
            ),
            "network_split_unresolved",
        ),
    ],
)
def test_classify_fn_uses_mutually_exclusive_priority(evidence, expected):
    assert classify_fn(evidence).category == expected


def test_index_trace_events_assigns_stable_ids_and_groups_by_source_port():
    trace = {
        "events": [
            {
                "stage": "p2j_trace",
                "accepted": False,
                "reason": "no_skeleton_path",
                "source": {"component_index": 2, "port_index": 1},
            },
            {
                "stage": "los",
                "accepted": True,
                "reason": "skeleton_supported",
                "source": {"component_index": 2, "port_index": 1},
            },
        ]
    }

    indexed = index_trace_events(trace)

    assert [event["event_id"] for event in indexed[(2, 1)]] == ["E00000", "E00001"]


def test_skeleton_near_port_detects_local_pixels_without_leaving_image():
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    skeleton[1, 1] = 255

    assert skeleton_near_port(skeleton, (0, 0), radius=2) is True
    assert skeleton_near_port(skeleton, (9, 9), radius=2) is False


def test_attribute_fn_edges_emits_exactly_one_row_per_edge():
    edges = {("C1.1", "R1.1"), ("R1.1", "R2.2")}
    evidence_by_port = {
        "C1.1": FnEvidence(component_matched=False),
        "R1.1": FnEvidence(
            component_matched=True,
            port_mapped=True,
            skeleton_near_port=True,
            trace_reached_network=True,
            candidate_generated=False,
        ),
        "R2.2": FnEvidence(
            component_matched=True,
            port_mapped=True,
            skeleton_near_port=True,
            trace_reached_network=True,
            candidate_generated=False,
        ),
    }

    rows = attribute_fn_edges(edges, evidence_by_port)

    assert len(rows) == len(edges)
    assert {row.edge: row.category for row in rows} == {
        ("C1.1", "R1.1"): "component_unmatched",
        ("R1.1", "R2.2"): "candidate_not_generated",
    }
