import numpy as np
import pytest

from src.vision.wiring_error_analysis import (
    FnEvidence,
    attribute_fp_edges,
    attribute_fn_edges,
    build_physical_edges,
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


def test_network_merge_marks_one_root_and_remaining_cross_edges_as_cascade():
    gt_groups = [
        frozenset({"R1.1", "R2.1"}),
        frozenset({"C1.1", "C2.1"}),
    ]
    pred_groups = [
        frozenset({"R1.1", "R2.1", "C1.1", "C2.1"}),
    ]
    physical_edges = [
        {"event_id": "E00001", "kind": "p2j", "ports": ("R1.1", "R2.1")},
        {"event_id": "E00002", "kind": "jj", "ports": ("R2.1", "C1.1")},
        {"event_id": "E00003", "kind": "p2j", "ports": ("C1.1", "C2.1")},
    ]

    rows = attribute_fp_edges(gt_groups, pred_groups, physical_edges)

    roots = [row for row in rows if row.is_root]
    cascades = [row for row in rows if row.category == "cascade_fp"]
    assert [(row.category, row.root_event_id) for row in roots] == [
        ("wrong_junction_merge", "E00002")
    ]
    assert len(rows) == 4
    assert len(cascades) == 3
    assert {row.root_event_id for row in cascades} == {"E00002"}


def test_ambiguous_merge_has_explicit_unattributed_root():
    rows = attribute_fp_edges(
        [
            frozenset({"R1.1", "R2.1"}),
            frozenset({"C1.1", "C2.1"}),
        ],
        [frozenset({"R1.1", "R2.1", "C1.1", "C2.1"})],
        [],
    )

    roots = [row for row in rows if row.is_root]
    assert len(roots) == 1
    assert roots[0].category == "unattributed_merge"
    assert roots[0].root_event_id.startswith("UNATTRIBUTED")


@pytest.mark.parametrize(
    ("physical", "expected"),
    [
        ({"event_id": "E1", "kind": "p2j", "ports": ("R1.1", "C1.1")}, "wrong_port_to_junction"),
        ({"event_id": "E1", "kind": "los", "ports": ("R1.1", "C1.1")}, "wrong_port_to_port"),
        ({"event_id": "E1", "kind": "jj", "ports": ("R1.1", "C1.1")}, "wrong_junction_merge"),
        ({"event_id": "E1", "kind": "jj", "ports": ("R1.1", "C1.1"), "reason": "ambiguous_crossing"}, "crossing_ambiguity"),
        ({"event_id": "E1", "kind": "p2j", "ports": ("R1.1", "C1.1"), "reason": "crosses_component"}, "component_crossing"),
    ],
)
def test_merge_root_kind_maps_to_actionable_category(physical, expected):
    rows = attribute_fp_edges(
        [frozenset({"R1.1", "R2.1"}), frozenset({"C1.1", "C2.1"})],
        [frozenset({"R1.1", "R2.1", "C1.1", "C2.1"})],
        [physical],
    )

    assert [row.category for row in rows if row.is_root] == [expected]


def test_false_edge_inside_one_gt_network_is_local_not_merge():
    rows = attribute_fp_edges(
        [frozenset({"R1.1", "R2.1"})],
        [frozenset({"R1.1", "R2.1", "X1.1"})],
        [],
    )

    assert {row.category for row in rows} == {"local_false_edge"}
    assert all(row.is_root for row in rows)


def test_build_physical_edges_normalizes_port_and_junction_nodes():
    trace = {
        "events": [
            {
                "accepted": True,
                "kind": "p2j",
                "reason": "continuous_skeleton_path",
                "source": {"component_index": 0, "port_index": 1},
                "target": {"junction": [10, 20]},
            },
            {
                "accepted": False,
                "kind": "p2j",
                "source": {"component_index": 1, "port_index": 0},
                "target": {"junction": [10, 20]},
            },
        ]
    }

    edges = build_physical_edges(trace, {(0, 1): "R1.2", (1, 0): "C1.1"})

    assert edges == (
        {
            "event_id": "E00000",
            "kind": "p2j",
            "stage": "",
            "reason": "continuous_skeleton_path",
            "source_node": "P:R1.2",
            "target_node": "J:10,20",
            "evidence": {},
        },
    )


def test_junction_bridge_between_two_gt_networks_is_the_merge_root():
    gt_groups = [
        frozenset({"R1.1", "R2.1"}),
        frozenset({"C1.1", "C2.1"}),
    ]
    pred_groups = [
        frozenset({"R1.1", "R2.1", "C1.1", "C2.1"}),
    ]
    physical_edges = [
        {"event_id": "E1", "kind": "p2j", "source_node": "P:R1.1", "target_node": "J:1,1"},
        {"event_id": "E2", "kind": "p2j", "source_node": "P:R2.1", "target_node": "J:1,1"},
        {"event_id": "E3", "kind": "jj", "source_node": "J:1,1", "target_node": "J:9,9"},
        {"event_id": "E4", "kind": "p2j", "source_node": "P:C1.1", "target_node": "J:9,9"},
        {"event_id": "E5", "kind": "p2j", "source_node": "P:C2.1", "target_node": "J:9,9"},
    ]

    rows = attribute_fp_edges(gt_groups, pred_groups, physical_edges)

    roots = [row for row in rows if row.is_root]
    assert [(row.category, row.root_event_id) for row in roots] == [
        ("wrong_junction_merge", "E3")
    ]
