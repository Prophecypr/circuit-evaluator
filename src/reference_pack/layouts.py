"""Deterministic component and net-hub layouts for blind references."""

from __future__ import annotations

from typing import Literal, TypedDict

from .specs import CircuitSpec


ComponentPlacement = tuple[int, int, Literal["h", "v"]]
Point = tuple[int, int]


class Layout(TypedDict):
    components: dict[str, ComponentPlacement]
    hubs: dict[str, Point]


LAYOUTS: dict[str, Layout] = {
    "C01": {
        "components": {
            "V1": (180, 350, "v"),
            "R1": (500, 160, "h"),
            "GND": (500, 560, "v"),
        },
        "hubs": {"N1": (300, 160), "N0": (500, 500)},
    },
    "C02": {
        "components": {
            "V1": (150, 350, "v"),
            "R1": (390, 150, "h"),
            "R2": (690, 150, "h"),
            "GND": (500, 560, "v"),
        },
        "hubs": {"N1": (260, 150), "N2": (540, 150), "N0": (500, 500)},
    },
    "C03": {
        "components": {
            "V1": (150, 350, "v"),
            "R1": (440, 250, "h"),
            "R2": (440, 450, "h"),
            "GND": (760, 560, "v"),
        },
        "hubs": {"N1": (280, 350), "N0": (720, 500)},
    },
    "C04": {
        "components": {
            "V1": (150, 350, "v"),
            "R1": (400, 160, "h"),
            "LED1": (700, 350, "v"),
            "GND": (500, 560, "v"),
        },
        "hubs": {"N1": (270, 160), "N2": (700, 220), "N0": (500, 500)},
    },
    "C05": {
        "components": {
            "V1": (140, 350, "v"),
            "R1": (380, 150, "h"),
            "R2": (610, 320, "v"),
            "R3": (790, 320, "v"),
            "GND": (610, 580, "v"),
        },
        "hubs": {"N1": (260, 150), "N2": (610, 210), "N0": (610, 520)},
    },
    "C06": {
        "components": {
            "V1": (140, 350, "v"),
            "R1": (380, 150, "h"),
            "R2": (610, 340, "v"),
            "C1": (790, 340, "v"),
            "GND": (610, 580, "v"),
        },
        "hubs": {"N1": (260, 150), "N2": (610, 230), "N0": (610, 520)},
    },
    "C07": {
        "components": {
            "V1": (120, 350, "v"),
            "R1": (320, 150, "h"),
            "L1": (570, 150, "h"),
            "C1": (800, 350, "v"),
            "GND": (500, 580, "v"),
        },
        "hubs": {
            "N1": (220, 150),
            "N2": (445, 150),
            "N3": (720, 150),
            "N0": (500, 520),
        },
    },
    "C08": {
        "components": {
            "V1": (130, 350, "v"),
            "R1": (370, 150, "h"),
            "ZD1": (610, 340, "v"),
            "R2": (800, 340, "v"),
            "GND": (610, 580, "v"),
        },
        "hubs": {"N1": (250, 150), "N2": (610, 220), "N0": (610, 520)},
    },
    "C09": {
        "components": {
            "V1": (110, 350, "v"),
            "R1": (340, 140, "h"),
            "LED1": (610, 245, "v"),
            "Q1": (650, 420, "v"),
            "R2": (350, 360, "h"),
            "GND": (620, 600, "v"),
        },
        "hubs": {
            "N1": (240, 220),
            "N2": (610, 160),
            "N3": (650, 330),
            "N4": (500, 420),
            "N0": (620, 550),
        },
    },
    "C10": {
        "components": {
            "V1": (100, 350, "v"),
            "R1": (360, 200, "v"),
            "R2": (360, 480, "v"),
            "R3": (720, 200, "v"),
            "R4": (720, 480, "v"),
            "R5": (540, 350, "h"),
            "GND": (540, 600, "v"),
        },
        "hubs": {
            "N1": (540, 100),
            "NL": (360, 350),
            "NR": (720, 350),
            "N0": (540, 570),
        },
    },
}


def validate_layout(circuit: CircuitSpec, layout: Layout) -> None:
    """Require an exact component placement and hub for every semantic net."""

    if set(layout["components"]) != {component.id for component in circuit.components}:
        raise ValueError(f"{circuit.id}: component layout mismatch")
    if set(layout["hubs"]) != set(circuit.nets):
        raise ValueError(f"{circuit.id}: net hub mismatch")
