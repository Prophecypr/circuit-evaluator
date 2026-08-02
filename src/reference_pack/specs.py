"""Canonical semantic specifications for the blind reference circuits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal


Difficulty = Literal["basic", "medium", "challenge"]


PORTS: dict[str, tuple[str, ...]] = {
    "resistor": ("1", "2"),
    "capacitor.unpolarized": ("1", "2"),
    "inductor": ("1", "2"),
    "diode.light_emitting": ("+", "-"),
    "diode.zener": ("A", "K"),
    "voltage.dc": ("+", "-"),
    "gnd": ("GND",),
    "transistor.bjt": ("B", "C", "E"),
}


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    class_name: str
    value: str | None
    ports: tuple[str, ...]
    symbol: str


@dataclass(frozen=True)
class CircuitSpec:
    id: str
    title: str
    difficulty: Difficulty
    components: tuple[ComponentSpec, ...]
    nets: dict[str, tuple[str, ...]]


def component(
    component_id: str,
    class_name: str,
    value: str | None = None,
) -> ComponentSpec:
    """Build a component using the canonical port and symbol conventions."""
    symbol = "ansi_zigzag" if class_name == "resistor" else class_name
    return ComponentSpec(
        id=component_id,
        class_name=class_name,
        value=value,
        ports=PORTS[class_name],
        symbol=symbol,
    )


def circuit(
    circuit_id: str,
    title: str,
    difficulty: Difficulty,
    components: list[ComponentSpec] | tuple[ComponentSpec, ...],
    nets: dict[str, list[str] | tuple[str, ...]],
) -> CircuitSpec:
    """Build a circuit while normalizing component and net references to tuples."""
    return CircuitSpec(
        id=circuit_id,
        title=title,
        difficulty=difficulty,
        components=tuple(components),
        nets={net_id: tuple(refs) for net_id, refs in nets.items()},
    )


def validate_circuit(spec: CircuitSpec) -> None:
    """Validate that component ports form an exact partition across the nets."""
    component_ids = [item.id for item in spec.components]
    duplicate_ids = sorted(
        component_id
        for component_id, count in Counter(component_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"{spec.id}: duplicate component ids: {', '.join(duplicate_ids)}"
        )

    for item in spec.components:
        duplicate_port_labels = sorted(
            port for port, count in Counter(item.ports).items() if count > 1
        )
        if duplicate_port_labels:
            raise ValueError(
                f"{spec.id}: component {item.id} has duplicate port labels: "
                f"{', '.join(duplicate_port_labels)}"
            )

    declared_ports = {
        f"{item.id}.{port}"
        for item in spec.components
        for port in item.ports
    }
    observed_ports: list[str] = []

    for net_id, refs in spec.nets.items():
        if len(refs) < 2:
            raise ValueError(f"{spec.id}: net {net_id} must contain at least 2 ports")
        observed_ports.extend(refs)

    port_counts = Counter(observed_ports)
    duplicate_ports = sorted(ref for ref, count in port_counts.items() if count > 1)
    if duplicate_ports:
        raise ValueError(
            f"{spec.id}: duplicate port references: {', '.join(duplicate_ports)}"
        )

    unknown_ports = sorted(set(observed_ports) - declared_ports)
    if unknown_ports:
        raise ValueError(
            f"{spec.id}: undeclared port references: {', '.join(unknown_ports)}"
        )

    missing_ports = sorted(declared_ports - set(observed_ports))
    if missing_ports:
        raise ValueError(
            f"{spec.id}: ports missing from nets: {', '.join(missing_ports)}"
        )


CIRCUITS: dict[str, CircuitSpec] = {
    "C01": circuit(
        "C01",
        "单电阻直流回路",
        "basic",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "1kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N0": ("R1.2", "V1.-", "GND.GND"),
        },
    ),
    "C02": circuit(
        "C02",
        "双电阻串联",
        "basic",
        [
            component("V1", "voltage.dc", "9V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "2.2kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1"),
            "N0": ("R2.2", "V1.-", "GND.GND"),
        },
    ),
    "C03": circuit(
        "C03",
        "双电阻并联",
        "basic",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "10kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1", "R2.1"),
            "N0": ("V1.-", "R1.2", "R2.2", "GND.GND"),
        },
    ),
    "C04": circuit(
        "C04",
        "LED限流回路",
        "basic",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "330Ω"),
            component("LED1", "diode.light_emitting"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "LED1.+"),
            "N0": ("LED1.-", "V1.-", "GND.GND"),
        },
    ),
    "C05": circuit(
        "C05",
        "带负载电压分压器",
        "medium",
        [
            component("V1", "voltage.dc", "9V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "2.2kΩ"),
            component("R3", "resistor", "10kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1", "R3.1"),
            "N0": ("R2.2", "R3.2", "V1.-", "GND.GND"),
        },
    ),
    "C06": circuit(
        "C06",
        "带泄放电阻的RC网络",
        "medium",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "10kΩ"),
            component("R2", "resistor", "2.2kΩ"),
            component("C1", "capacitor.unpolarized", "10uF"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1", "C1.1"),
            "N0": ("R2.2", "C1.2", "V1.-", "GND.GND"),
        },
    ),
    "C07": circuit(
        "C07",
        "RLC串联回路",
        "medium",
        [
            component("V1", "voltage.dc", "12V"),
            component("R1", "resistor", "100Ω"),
            component("L1", "inductor", "10mH"),
            component("C1", "capacitor.unpolarized", "1uF"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "L1.1"),
            "N3": ("L1.2", "C1.1"),
            "N0": ("C1.2", "V1.-", "GND.GND"),
        },
    ),
    "C08": circuit(
        "C08",
        "稳压二极管并联稳压",
        "medium",
        [
            component("V1", "voltage.dc", "12V"),
            component("R1", "resistor", "1kΩ"),
            component("ZD1", "diode.zener"),
            component("R2", "resistor", "2.2kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "ZD1.K", "R2.1"),
            "N0": ("ZD1.A", "R2.2", "V1.-", "GND.GND"),
        },
    ),
    "C09": circuit(
        "C09",
        "BJT-LED驱动电路",
        "challenge",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "330Ω"),
            component("R2", "resistor", "10kΩ"),
            component("LED1", "diode.light_emitting"),
            component("Q1", "transistor.bjt"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1", "R2.1"),
            "N2": ("R1.2", "LED1.+"),
            "N3": ("LED1.-", "Q1.C"),
            "N4": ("R2.2", "Q1.B"),
            "N0": ("Q1.E", "V1.-", "GND.GND"),
        },
    ),
    "C10": circuit(
        "C10",
        "惠斯通电桥",
        "challenge",
        [
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "100Ω"),
            component("R2", "resistor", "220Ω"),
            component("R3", "resistor", "330Ω"),
            component("R4", "resistor", "470Ω"),
            component("R5", "resistor", "1kΩ"),
            component("GND", "gnd"),
        ],
        {
            "N1": ("V1.+", "R1.1", "R3.1"),
            "NL": ("R1.2", "R2.1", "R5.1"),
            "NR": ("R3.2", "R4.1", "R5.2"),
            "N0": ("R2.2", "R4.2", "V1.-", "GND.GND"),
        },
    ),
}


for _circuit in CIRCUITS.values():
    validate_circuit(_circuit)
