"""Pure helpers for traceable circuit wiring-graph construction."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class WiringEvent:
    stage: str
    kind: str
    accepted: bool
    reason: str
    source: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


class WiringTrace:
    """Collect reproducible accept/reject evidence for wiring candidates."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: list[WiringEvent] = []

    def record(
        self,
        stage: str,
        kind: str,
        accepted: bool,
        reason: str,
        source: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
        **evidence: Any,
    ) -> None:
        if not self.enabled:
            return
        self.events.append(
            WiringEvent(
                stage=stage,
                kind=kind,
                accepted=accepted,
                reason=reason,
                source=source or {},
                target=target or {},
                evidence=evidence,
            )
        )

    def summary(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[WiringEvent]] = defaultdict(list)
        for event in self.events:
            grouped[event.stage].append(event)
        return {
            stage: {
                "candidates": len(events),
                "accepted": sum(event.accepted for event in events),
                "rejected": sum(not event.accepted for event in events),
                "reasons": dict(Counter(event.reason for event in events)),
            }
            for stage, events in grouped.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [asdict(event) for event in self.events],
            "summary": self.summary(),
        }


def network_color(port_ids: Iterable[str]) -> tuple[int, int, int]:
    """Return a deterministic visible BGR color for one network identity."""
    token = "|".join(sorted(str(port_id) for port_id in port_ids)).encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return tuple(64 + byte % 160 for byte in digest[:3])


def build_network_render_data(raw_groups, components) -> list[dict[str, Any]]:
    """Turn Union-Find port groups into stable labels, points, and colors."""
    groups = []
    for raw_group in raw_groups:
        members = []
        for raw_ci, raw_pi in raw_group:
            ci, pi = int(raw_ci), int(raw_pi)
            if not (0 <= ci < len(components)):
                continue
            component = components[ci]
            ports = component.get("ports", [])
            if not (0 <= pi < len(ports)):
                continue
            labels = component.get("port_labels", component.get("labels", []))
            label = str(labels[pi]) if pi < len(labels) else f"P{pi + 1}"
            designator = str(component.get("designator") or f"C{ci + 1}")
            point = tuple(map(int, ports[pi]))
            members.append(
                {
                    "component_index": ci,
                    "port_index": pi,
                    "port_id": f"{designator}.{label}",
                    "point": point,
                }
            )
        if members:
            members.sort(key=lambda member: member["port_id"])
            groups.append(members)

    groups.sort(key=lambda members: tuple(member["port_id"] for member in members))
    return [
        {
            "network_id": f"N{index}",
            "color": network_color(member["port_id"] for member in members),
            "members": members,
        }
        for index, members in enumerate(groups, start=1)
    ]


def terminal_component(
    bbox: Iterable[float], confidence: float, index: int
) -> dict[str, Any]:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return {
        "idx": index,
        "name": "Terminal",
        "display": "Terminal",
        "raw_name": "terminal",
        "xyxy": (x1, y1, x2, y2),
        "cx": center[0],
        "cy": center[1],
        "conf": float(confidence),
        "value": "",
        "ports": [center],
        "labels": ["T"],
        "designator": "",
        "label_swap": False,
    }


def classify_connection_detection(
    name: str,
    bbox: Iterable[float],
    confidence: float,
    use_terminal_components: bool,
    index: int,
) -> dict[str, Any]:
    values = tuple(int(round(value)) for value in bbox)
    x1, y1, x2, y2 = values
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    if name == "junction" or (name == "terminal" and not use_terminal_components):
        return {"junction": center, "component": None}
    if name == "terminal":
        return {
            "junction": None,
            "component": terminal_component(values, confidence, index),
        }
    raise ValueError(f"unsupported connection detection: {name}")


def accept_p2j_candidate(
    *,
    distance: float,
    path_found: bool,
    crosses_component: bool,
    strict: bool,
) -> tuple[bool, str]:
    """Apply the evidence policy for one port-to-junction candidate."""
    if crosses_component:
        return False, "crosses_component"
    if path_found:
        return True, "continuous_skeleton_path"
    if strict:
        return False, "no_skeleton_path"
    return True, "legacy_distance_fallback"


def trace_port_to_anchor(
    *,
    skeleton,
    port: tuple[int, int],
    component_center: tuple[int, int],
    anchors: Iterable[tuple[str, tuple[int, int]]],
    search_radius: int,
    anchor_radius: int,
    max_steps: int,
    gap_bridge: int = 0,
    min_bridge_cosine: float = 0.85,
    max_gap_bridges: int = 1,
) -> dict[str, Any] | None:
    """Follow skeleton pixels outward and stop at the first reachable anchor.

    An optional bridge may cross one short blank gap, but only from a real
    skeleton dead end to a locally collinear continuation in the port-outward
    direction. The input skeleton is never modified.
    """
    height, width = skeleton.shape[:2]
    px, py = map(int, port)
    cx, cy = map(int, component_center)
    outward = (px - cx, py - cy)
    outward_norm = math.hypot(*outward)

    def projection(x: int, y: int) -> float:
        if outward_norm == 0:
            return 0.0
        return ((x - px) * outward[0] + (y - py) * outward[1]) / outward_norm

    forward_anchors = [
        (str(anchor_id), (int(point[0]), int(point[1])))
        for anchor_id, point in anchors
        if outward_norm == 0 or projection(int(point[0]), int(point[1])) > 0
    ]
    if not forward_anchors:
        return None

    starts: list[tuple[float, int, int]] = []
    for y in range(max(0, py - search_radius), min(height, py + search_radius + 1)):
        for x in range(max(0, px - search_radius), min(width, px + search_radius + 1)):
            if skeleton[y, x] <= 0 or projection(x, y) < -1.0:
                continue
            starts.append((math.hypot(x - px, y - py), x, y))
    if not starts:
        return None
    _, start_x, start_y = min(starts)

    def cosine(first: tuple[float, float], second: tuple[float, float]) -> float:
        first_norm = math.hypot(*first)
        second_norm = math.hypot(*second)
        if first_norm == 0 or second_norm == 0:
            return -1.0
        return (
            first[0] * second[0] + first[1] * second[1]
        ) / (first_norm * second_norm)

    def skeleton_neighbors(x: int, y: int) -> list[tuple[int, int]]:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and skeleton[ny, nx] > 0
                    and projection(nx, ny) >= -1.0
                ):
                    neighbors.append((nx, ny))
        return neighbors

    queue = deque([(
        start_x,
        start_y,
        0,
        0,
        ((start_x, start_y),),
        (),
    )])
    visited = {(start_x, start_y, 0)}
    while queue:
        x, y, depth, bridge_count, history, bridge_gaps = queue.popleft()
        reached = [
            (math.hypot(x - ax, y - ay), anchor_id, (ax, ay))
            for anchor_id, (ax, ay) in forward_anchors
            if math.hypot(x - ax, y - ay) <= anchor_radius
        ]
        if reached:
            _, anchor_id, anchor = min(reached, key=lambda item: (item[0], item[1]))
            return {
                "anchor_id": anchor_id,
                "anchor": anchor,
                "reason": (
                    "directional_gap_bridge"
                    if bridge_count
                    else "continuous_skeleton_path"
                ),
                "path_length": depth,
                "visited_pixels": len({(vx, vy) for vx, vy, _ in visited}),
                "gap_bridges": bridge_count,
                "max_gap": max(bridge_gaps, default=0.0),
            }
        if depth >= max_steps:
            continue
        neighbors = skeleton_neighbors(x, y)
        regular_neighbors = [
            (nx, ny)
            for nx, ny in neighbors
            if (nx, ny, bridge_count) not in visited
        ]
        for nx, ny in regular_neighbors:
            visited.add((nx, ny, bridge_count))
            next_history = (history + ((nx, ny),))[-5:]
            queue.append((
                nx,
                ny,
                depth + 1,
                bridge_count,
                next_history,
                bridge_gaps,
            ))

        if gap_bridge <= 0 or bridge_count >= max_gap_bridges:
            continue

        recent_points = set(history[:-1])
        physical_continuations = [
            point for point in neighbors if point not in recent_points
        ]
        if physical_continuations:
            continue

        tangent_origin = history[0]
        incoming = (x - tangent_origin[0], y - tangent_origin[1])
        if math.hypot(*incoming) < 1.5:
            incoming = outward

        bridge_candidates = []
        for gy in range(max(0, y - gap_bridge), min(height, y + gap_bridge + 1)):
            for gx in range(max(0, x - gap_bridge), min(width, x + gap_bridge + 1)):
                gap_vector = (gx - x, gy - y)
                gap_distance = math.hypot(*gap_vector)
                if not (math.sqrt(2.0) < gap_distance <= gap_bridge):
                    continue
                if skeleton[gy, gx] <= 0:
                    continue
                if (gx, gy, bridge_count + 1) in visited:
                    continue
                if projection(gx, gy) <= projection(x, y) + 0.5:
                    continue
                direction_cosine = cosine(incoming, gap_vector)
                if direction_cosine < min_bridge_cosine:
                    continue

                target_neighbors = skeleton_neighbors(gx, gy)
                if len(target_neighbors) > 2:
                    continue
                forward_continuations = [
                    (nx, ny)
                    for nx, ny in target_neighbors
                    if cosine(gap_vector, (nx - gx, ny - gy)) >= min_bridge_cosine
                ]
                if not forward_continuations:
                    continue
                bridge_candidates.append((
                    gap_distance,
                    -direction_cosine,
                    gx,
                    gy,
                ))

        if bridge_candidates:
            gap_distance, _, gx, gy = min(bridge_candidates)
            visited.add((gx, gy, bridge_count + 1))
            queue.append((
                gx,
                gy,
                depth + int(math.ceil(gap_distance)),
                bridge_count + 1,
                ((x, y), (gx, gy)),
                bridge_gaps + (float(gap_distance),),
            ))
    return None


def _nearest_skeleton_pixel(skeleton, point: tuple[int, int], radius: int):
    height, width = skeleton.shape[:2]
    px, py = map(int, point)
    candidates = []
    for y in range(max(0, py - radius), min(height, py + radius + 1)):
        for x in range(max(0, px - radius), min(width, px + radius + 1)):
            if skeleton[y, x] > 0:
                candidates.append((math.hypot(x - px, y - py), x, y))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def _skeleton_path(skeleton, start, end, max_steps: int):
    queue = deque([start])
    parent = {start: None}
    depth = {start: 0}
    height, width = skeleton.shape[:2]
    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        if depth[current] >= max_steps:
            continue
        x, y = current
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (x + dx, y + dy)
                nx, ny = nxt
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if nxt in parent or skeleton[ny, nx] <= 0:
                    continue
                parent[nxt] = current
                depth[nxt] = depth[current] + 1
                queue.append(nxt)
    return None


def _skeleton_branch_pixels(
    skeleton,
    path: Iterable[tuple[int, int]],
    neighborhood: int = 2,
):
    height, width = skeleton.shape[:2]
    branches = []
    candidates = set()
    for path_x, path_y in path:
        for y in range(max(0, path_y - neighborhood), min(height, path_y + neighborhood + 1)):
            for x in range(max(0, path_x - neighborhood), min(width, path_x + neighborhood + 1)):
                if skeleton[y, x] > 0:
                    candidates.add((x, y))
    for x, y in sorted(candidates):
        directions = set()
        for dx, dy, label in ((1, 0, "h"), (-1, 0, "h"), (0, 1, "v"), (0, -1, "v")):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx] > 0:
                directions.add((dx, dy, label))
        horizontal = sum(1 for _, _, label in directions if label == "h")
        vertical = sum(1 for _, _, label in directions if label == "v")
        if horizontal and vertical and len(directions) >= 3:
            branches.append((x, y))
    return branches


def strict_jj_decision(
    *,
    skeleton,
    start: tuple[int, int],
    end: tuple[int, int],
    detected_junctions: Iterable[tuple[int, int]],
    crossing_semantics: bool,
    search_radius: int = 4,
    junction_radius: int = 6,
    max_steps: int = 2000,
) -> tuple[bool, str]:
    """Accept a JJ edge only when a continuous, unambiguous skeleton path exists."""
    if skeleton is None:
        return False, "no_skeleton_path"
    skel_start = _nearest_skeleton_pixel(skeleton, start, search_radius)
    skel_end = _nearest_skeleton_pixel(skeleton, end, search_radius)
    if skel_start is None or skel_end is None:
        return False, "no_skeleton_path"
    path = _skeleton_path(skeleton, skel_start, skel_end, max_steps)
    if not path:
        return False, "no_skeleton_path"

    if crossing_semantics:
        dx = abs(skel_end[0] - skel_start[0])
        dy = abs(skel_end[1] - skel_start[1])
        path_turns = dx > search_radius and dy > search_radius
        if path_turns:
            junctions = [(int(x), int(y)) for x, y in detected_junctions]
            for bx, by in _skeleton_branch_pixels(skeleton, path):
                marked = any(
                    math.hypot(bx - jx, by - jy) <= junction_radius
                    for jx, jy in junctions
                )
                if not marked:
                    return False, "ambiguous_crossing"
    return True, "continuous_skeleton_path"
