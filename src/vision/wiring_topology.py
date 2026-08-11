"""Pure helpers for naming circuit ports and comparing wiring topology."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


NamedGroup = frozenset[str]
NamedEdge = tuple[str, str]


@dataclass(frozen=True)
class EdgeInventory:
    gt_groups: tuple[NamedGroup, ...]
    pred_groups: tuple[NamedGroup, ...]
    gt_edges: frozenset[NamedEdge]
    pred_edges: frozenset[NamedEdge]
    tp_edges: frozenset[NamedEdge]
    fp_edges: frozenset[NamedEdge]
    fn_edges: frozenset[NamedEdge]
    pipeline_port_ids: dict[tuple[int, int], str]
    port_points: dict[str, tuple[int, int]]
    component_matches: dict[int, int]
    unmatched_detection_components: frozenset[int]
    unmapped_gt_ports: frozenset[str]


def groups_to_edges(groups: Iterable[Iterable[str]]) -> set[NamedEdge]:
    """Expand each network into canonical undirected port-pair edges."""
    edges: set[NamedEdge] = set()
    for group in groups:
        ports = sorted(set(group))
        for first_index in range(len(ports)):
            for second_index in range(first_index + 1, len(ports)):
                edges.add((ports[first_index], ports[second_index]))
    return edges


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    first_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    second_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_components(
    pipeline_components: Sequence[Mapping],
    detection_components: Sequence[Mapping],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Greedily match pipeline components to GT-facing detections by IoU."""
    matches: dict[int, int] = {}
    used_detection_indices: set[int] = set()
    for pipeline_index, pipeline_component in enumerate(pipeline_components):
        best_iou = 0.0
        best_detection_index = None
        pipeline_box = pipeline_component.get("xyxy", [0, 0, 0, 0])
        for detection_index, detection_component in enumerate(detection_components):
            if detection_index in used_detection_indices:
                continue
            iou = _bbox_iou(pipeline_box, detection_component["xyxy"])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_detection_index = detection_index
        if best_detection_index is not None:
            matches[pipeline_index] = best_detection_index
            used_detection_indices.add(best_detection_index)
    return matches


def _port_pairs(
    pipeline_ports: Sequence[Sequence[float]],
    detection_ports: Sequence[Sequence[float]],
    port_match_radius: float,
) -> list[tuple[int, int]]:
    if len(pipeline_ports) == len(detection_ports):
        return [(index, index) for index in range(len(pipeline_ports))]

    pairs = []
    for pipeline_port_index, (px, py) in enumerate(pipeline_ports):
        candidates = []
        for detection_port_index, (dpx, dpy) in enumerate(detection_ports):
            distance = math.hypot(px - dpx, py - dpy)
            if distance < port_match_radius:
                candidates.append((distance, detection_port_index))
        if candidates:
            _, detection_port_index = min(candidates)
            pairs.append((pipeline_port_index, detection_port_index))
    return pairs


def build_edge_inventory(
    pipeline_result: Mapping,
    gt_groups: Iterable[Iterable[tuple[str, str]]],
    detection_components: Sequence[Mapping],
    port_match_radius: float = 60,
) -> EdgeInventory:
    """Build named GT/predicted networks and their exact edge differences."""
    pipeline_components = pipeline_result.get("components", [])
    matches = match_components(pipeline_components, detection_components)
    pipeline_port_ids: dict[tuple[int, int], str] = {}
    port_points: dict[str, tuple[int, int]] = {}

    for pipeline_index, component in enumerate(pipeline_components):
        detection_index = matches.get(pipeline_index)
        if detection_index is None:
            continue
        detection = detection_components[detection_index]
        pipeline_ports = component.get("ports", [])
        detection_ports = detection.get("ports", [])
        for pipeline_port_index, detection_port_index in _port_pairs(
            pipeline_ports,
            detection_ports,
            port_match_radius,
        ):
            labels = detection.get("labels", [])
            label = labels[detection_port_index] if detection_port_index < len(labels) else "?"
            port_id = f"{detection['designator']}.{label}"
            pipeline_port_ids[(pipeline_index, pipeline_port_index)] = port_id
            port_points[port_id] = tuple(map(int, pipeline_ports[pipeline_port_index]))

    gt_named_groups = tuple(
        frozenset(f"{designator}.{label}" for designator, label in group)
        for group in gt_groups
    )
    pred_named_groups = []
    for raw_group in pipeline_result.get("raw_groups", []):
        named_group = frozenset(
            pipeline_port_ids[key]
            for raw_component_index, raw_port_index in raw_group
            if (key := (int(raw_component_index), int(raw_port_index)))
            in pipeline_port_ids
        )
        if len(named_group) >= 2:
            pred_named_groups.append(named_group)

    gt_edges = groups_to_edges(gt_named_groups)
    pred_edges = groups_to_edges(pred_named_groups)
    gt_ports = set().union(*gt_named_groups) if gt_named_groups else set()
    unmatched_detection_components = (
        set(range(len(detection_components))) - set(matches.values())
    )

    return EdgeInventory(
        gt_groups=gt_named_groups,
        pred_groups=tuple(pred_named_groups),
        gt_edges=frozenset(gt_edges),
        pred_edges=frozenset(pred_edges),
        tp_edges=frozenset(gt_edges & pred_edges),
        fp_edges=frozenset(pred_edges - gt_edges),
        fn_edges=frozenset(gt_edges - pred_edges),
        pipeline_port_ids=pipeline_port_ids,
        port_points=port_points,
        component_matches=matches,
        unmatched_detection_components=frozenset(unmatched_detection_components),
        unmapped_gt_ports=frozenset(gt_ports - set(pipeline_port_ids.values())),
    )
