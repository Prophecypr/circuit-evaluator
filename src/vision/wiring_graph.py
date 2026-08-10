"""Pure helpers for traceable circuit wiring-graph construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
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
