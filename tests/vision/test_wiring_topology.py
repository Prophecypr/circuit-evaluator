from src.vision.wiring_topology import build_edge_inventory, groups_to_edges


def test_groups_to_edges_is_undirected_and_order_independent():
    groups = [{"R1.1", "R2.2", "C1.+"}]

    assert groups_to_edges(groups) == {
        ("C1.+", "R1.1"),
        ("C1.+", "R2.2"),
        ("R1.1", "R2.2"),
    }


def test_build_edge_inventory_preserves_unmatched_component_and_port_sets():
    pipeline = {
        "components": [{"xyxy": [0, 0, 10, 10], "ports": [[0, 5]]}],
        "raw_groups": [[[0, 0]]],
    }
    detections = [
        {
            "xyxy": [0, 0, 10, 10],
            "ports": [[0, 5]],
            "labels": ["1"],
            "designator": "R1",
        },
        {
            "xyxy": [20, 0, 30, 10],
            "ports": [[20, 5]],
            "labels": ["2"],
            "designator": "R2",
        },
    ]

    inventory = build_edge_inventory(
        pipeline,
        [[("R1", "1"), ("R2", "2")]],
        detections,
    )

    assert inventory.gt_edges == {("R1.1", "R2.2")}
    assert inventory.pred_edges == set()
    assert inventory.unmatched_detection_components == {1}
    assert inventory.unmapped_gt_ports == {"R2.2"}


def test_build_edge_inventory_uses_position_fallback_when_port_counts_differ():
    pipeline = {
        "components": [
            {
                "xyxy": [0, 0, 20, 10],
                "ports": [[19, 5]],
            },
            {
                "xyxy": [30, 0, 40, 10],
                "ports": [[30, 5]],
            },
        ],
        "raw_groups": [[[0, 0], [1, 0]]],
    }
    detections = [
        {
            "xyxy": [0, 0, 20, 10],
            "ports": [[1, 5], [19, 5]],
            "labels": ["1", "2"],
            "designator": "R1",
        },
        {
            "xyxy": [30, 0, 40, 10],
            "ports": [[30, 5]],
            "labels": ["1"],
            "designator": "T1",
        },
    ]

    inventory = build_edge_inventory(
        pipeline,
        [[("R1", "2"), ("T1", "1")]],
        detections,
    )

    assert inventory.pred_groups == ({"R1.2", "T1.1"},)
    assert inventory.tp_edges == {("R1.2", "T1.1")}
