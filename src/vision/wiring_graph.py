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
) -> dict[str, Any] | None:
    """Follow skeleton pixels outward and stop at the first reachable anchor."""
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

    queue = deque([(start_x, start_y, 0)])
    visited = {(start_x, start_y)}
    while queue:
        x, y, depth = queue.popleft()
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
                "reason": "continuous_skeleton_path",
                "path_length": depth,
                "visited_pixels": len(visited),
            }
        if depth >= max_steps:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in visited or skeleton[ny, nx] <= 0:
                    continue
                if projection(nx, ny) < -1.0:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny, depth + 1))
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
