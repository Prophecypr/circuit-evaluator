"""Pure classification helpers for wiring-edge false positives and negatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


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
