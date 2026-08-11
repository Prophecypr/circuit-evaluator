"""Pure classification helpers for wiring-edge false positives and negatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .wiring_topology import groups_to_edges


@dataclass(frozen=True)
class FnEvidence:
    component_matched: bool
    port_mapped: bool = False
    skeleton_near_port: bool | None = None
    trace_reached_network: bool | None = None
    candidate_generated: bool | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ErrorAttribution:
    error_type: str
    edge: tuple[str, str]
    category: str
    secondary_reason: str = ""
    root_event_id: str = ""
    is_root: bool = True


def classify_fn(
    evidence: FnEvidence,
    edge: tuple[str, str] = ("", ""),
) -> ErrorAttribution:
    """Assign exactly one FN category using the documented priority order."""
    if not evidence.component_matched:
        return ErrorAttribution("FN", edge, "component_unmatched")
    if not evidence.port_mapped:
        return ErrorAttribution("FN", edge, "port_unmatched")
    if evidence.skeleton_near_port is False:
        return ErrorAttribution("FN", edge, "no_port_skeleton")
    if evidence.trace_reached_network is False:
        return ErrorAttribution("FN", edge, "skeleton_break")
    if evidence.rejection_reason:
        return ErrorAttribution(
            "FN",
            edge,
            "candidate_rejected",
            evidence.rejection_reason,
        )
    if evidence.candidate_generated is False:
        return ErrorAttribution("FN", edge, "candidate_not_generated")
    return ErrorAttribution("FN", edge, "network_split_unresolved")


def index_trace_events(trace_payload: Mapping) -> dict[tuple[int, int], list[dict]]:
    """Index trace events by pipeline port with deterministic event IDs."""
    indexed: dict[tuple[int, int], list[dict]] = {}
    for event_index, raw_event in enumerate(trace_payload.get("events", [])):
        event = {"event_id": f"E{event_index:05d}", **raw_event}
        source = event.get("source", {})
        component_index = source.get("component_index")
        port_index = source.get("port_index")
        if component_index is None or port_index is None:
            continue
        key = (int(component_index), int(port_index))
        indexed.setdefault(key, []).append(event)
    return indexed


def skeleton_near_port(skeleton, point: tuple[int, int], radius: int = 15) -> bool:
    """Return whether a skeleton pixel exists in a clipped square around a port."""
    x, y = map(int, point)
    height, width = skeleton.shape[:2]
    crop = skeleton[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    return bool(crop.size and (crop > 0).any())


def _combine_optional(values: Iterable[bool | None]) -> bool | None:
    known = [value for value in values if value is not None]
    if any(value is False for value in known):
        return False
    if known and all(value is True for value in known):
        return True
    return None


def _combine_endpoint_evidence(
    first: FnEvidence,
    second: FnEvidence,
) -> FnEvidence:
    rejection_reason = first.rejection_reason or second.rejection_reason
    return FnEvidence(
        component_matched=first.component_matched and second.component_matched,
        port_mapped=first.port_mapped and second.port_mapped,
        skeleton_near_port=_combine_optional(
            [first.skeleton_near_port, second.skeleton_near_port]
        ),
        trace_reached_network=_combine_optional(
            [first.trace_reached_network, second.trace_reached_network]
        ),
        candidate_generated=_combine_optional(
            [first.candidate_generated, second.candidate_generated]
        ),
        rejection_reason=rejection_reason,
    )


def attribute_fn_edges(
    fn_edges: Iterable[tuple[str, str]],
    evidence_by_port: Mapping[str, FnEvidence],
) -> tuple[ErrorAttribution, ...]:
    """Classify every FN edge exactly once in stable order."""
    rows = []
    missing = FnEvidence(component_matched=False)
    for edge in sorted(fn_edges):
        first = evidence_by_port.get(edge[0], missing)
        second = evidence_by_port.get(edge[1], missing)
        rows.append(classify_fn(_combine_endpoint_evidence(first, second), edge))
    return tuple(rows)


def _physical_category(physical_edge: Mapping) -> str:
    reason = str(physical_edge.get("reason", ""))
    if reason == "ambiguous_crossing" or physical_edge.get("crossing_ambiguity"):
        return "crossing_ambiguity"
    if reason == "crosses_component" or physical_edge.get("component_crossing"):
        return "component_crossing"
    kind = str(physical_edge.get("kind", ""))
    if kind == "p2j":
        return "wrong_port_to_junction"
    if kind == "jj":
        return "wrong_junction_merge"
    if kind in {"p2p", "los", "close_port", "p2p_direct", "p2p_aggressive"}:
        return "wrong_port_to_port"
    return "unattributed_merge"


def _trace_node(endpoint: Mapping, pipeline_port_ids: Mapping[tuple[int, int], str]):
    if "component_index" in endpoint and "port_index" in endpoint:
        key = (int(endpoint["component_index"]), int(endpoint["port_index"]))
        port_id = pipeline_port_ids.get(key)
        return f"P:{port_id}" if port_id else None
    junction = endpoint.get("junction")
    if isinstance(junction, (list, tuple)) and len(junction) == 2:
        return f"J:{int(junction[0])},{int(junction[1])}"
    return None


def build_physical_edges(
    trace_payload: Mapping,
    pipeline_port_ids: Mapping[tuple[int, int], str],
) -> tuple[dict, ...]:
    """Normalize accepted trace events into a port/junction physical graph."""
    edges = []
    for event_index, event in enumerate(trace_payload.get("events", [])):
        if event.get("accepted") is not True:
            continue
        source_node = _trace_node(event.get("source", {}), pipeline_port_ids)
        target_node = _trace_node(event.get("target", {}), pipeline_port_ids)
        if source_node is None or target_node is None:
            continue
        edge = {
            "event_id": f"E{event_index:05d}",
            "kind": str(event.get("kind", "")),
            "stage": str(event.get("stage", "")),
            "reason": str(event.get("reason", "")),
            "source_node": source_node,
            "target_node": target_node,
            "evidence": dict(event.get("evidence", {})),
        }
        if source_node.startswith("P:") and target_node.startswith("P:"):
            edge["ports"] = tuple(
                sorted((source_node.removeprefix("P:"), target_node.removeprefix("P:")))
            )
        edges.append(edge)
    return tuple(edges)


def _reachable_nodes(
    physical_edges: tuple[Mapping, ...],
    start: str,
    omitted_event_id: str,
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in physical_edges:
        if str(edge.get("event_id", "")) == omitted_event_id:
            continue
        source = edge.get("source_node")
        target = edge.get("target_node")
        if not source or not target:
            continue
        adjacency.setdefault(str(source), set()).add(str(target))
        adjacency.setdefault(str(target), set()).add(str(source))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def _bridge_root_candidates(
    pred_group: frozenset[str],
    physical_edges: tuple[Mapping, ...],
    gt_network_by_port: Mapping[str, int],
) -> list[tuple[str, tuple[str, str], Mapping]]:
    candidates = []
    pred_port_nodes = {f"P:{port}" for port in pred_group}
    for physical_edge in physical_edges:
        source_node = physical_edge.get("source_node")
        target_node = physical_edge.get("target_node")
        event_id = str(physical_edge.get("event_id", ""))
        if not source_node or not target_node or not event_id:
            continue
        left_nodes = _reachable_nodes(physical_edges, str(source_node), event_id)
        left_ports = sorted(
            node.removeprefix("P:")
            for node in pred_port_nodes & left_nodes
        )
        right_ports = sorted(
            node.removeprefix("P:")
            for node in pred_port_nodes - left_nodes
        )
        left_networks = {
            gt_network_by_port[port]
            for port in left_ports
            if port in gt_network_by_port
        }
        right_networks = {
            gt_network_by_port[port]
            for port in right_ports
            if port in gt_network_by_port
        }
        if not left_networks or not right_networks or not left_networks.isdisjoint(right_networks):
            continue
        cross_pairs = sorted(
            tuple(sorted((left_port, right_port)))
            for left_port in left_ports
            for right_port in right_ports
            if gt_network_by_port.get(left_port) != gt_network_by_port.get(right_port)
        )
        if cross_pairs:
            candidates.append((event_id, cross_pairs[0], physical_edge))
    return candidates


def attribute_fp_edges(
    gt_groups: Iterable[Iterable[str]],
    pred_groups: Iterable[Iterable[str]],
    physical_edges: Iterable[Mapping],
) -> tuple[ErrorAttribution, ...]:
    """Separate one network-merge root from its derived clique FP edges."""
    gt_groups = tuple(frozenset(group) for group in gt_groups)
    pred_groups = tuple(frozenset(group) for group in pred_groups)
    gt_edges = groups_to_edges(gt_groups)
    gt_network_by_port = {
        port: network_index
        for network_index, group in enumerate(gt_groups)
        for port in group
    }
    physical_edges = tuple(physical_edges)
    rows: list[ErrorAttribution] = []

    for pred_network_index, pred_group in enumerate(pred_groups):
        fp_edges = sorted(groups_to_edges([pred_group]) - gt_edges)
        if not fp_edges:
            continue
        gt_networks = {
            gt_network_by_port[port]
            for port in pred_group
            if port in gt_network_by_port
        }
        if len(gt_networks) < 2:
            for edge_index, edge in enumerate(fp_edges):
                rows.append(
                    ErrorAttribution(
                        "FP",
                        edge,
                        "local_false_edge",
                        root_event_id=f"LOCAL{pred_network_index:03d}_{edge_index:04d}",
                    )
                )
            continue

        root_candidates = []
        for physical_edge in physical_edges:
            raw_ports = physical_edge.get("ports", ())
            if len(raw_ports) != 2:
                continue
            edge = tuple(sorted(map(str, raw_ports)))
            first_network = gt_network_by_port.get(edge[0])
            second_network = gt_network_by_port.get(edge[1])
            if (
                edge[0] in pred_group
                and edge[1] in pred_group
                and first_network is not None
                and second_network is not None
                and first_network != second_network
            ):
                root_candidates.append(
                    (
                        str(physical_edge.get("event_id", "")),
                        edge,
                        physical_edge,
                    )
                )

        graph_candidates = _bridge_root_candidates(
            pred_group,
            physical_edges,
            gt_network_by_port,
        )
        known_event_ids = {candidate[0] for candidate in root_candidates}
        root_candidates.extend(
            candidate
            for candidate in graph_candidates
            if candidate[0] not in known_event_ids
        )

        if root_candidates:
            root_event_id, root_edge, physical_edge = min(
                root_candidates,
                key=lambda candidate: (candidate[0], candidate[1]),
            )
            if not root_event_id:
                root_event_id = f"PHYSICAL{pred_network_index:03d}"
            root_category = _physical_category(physical_edge)
            secondary_reason = str(physical_edge.get("reason", ""))
        else:
            root_edge = fp_edges[0]
            root_event_id = f"UNATTRIBUTED{pred_network_index:03d}"
            root_category = "unattributed_merge"
            secondary_reason = ""

        rows.append(
            ErrorAttribution(
                "FP",
                root_edge,
                root_category,
                secondary_reason=secondary_reason,
                root_event_id=root_event_id,
                is_root=True,
            )
        )
        for edge in fp_edges:
            if edge == root_edge:
                continue
            rows.append(
                ErrorAttribution(
                    "FP",
                    edge,
                    "cascade_fp",
                    root_event_id=root_event_id,
                    is_root=False,
                )
            )

    return tuple(sorted(rows, key=lambda row: (row.edge, not row.is_root)))
