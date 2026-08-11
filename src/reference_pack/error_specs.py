"""Independent error cases for the balanced LLM scoring set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .specs import ComponentSpec, circuit, component, validate_circuit


ErrorType = Literal[
    "polarity_reversal",
    "series_parallel_confusion",
    "numeric_value_error",
    "wrong_bjt_port",
    "wrong_junction",
]


@dataclass(frozen=True)
class ErrorCircuitSpec:
    id: str
    title: str
    difficulty: Literal["basic", "medium", "challenge"]
    error_type: ErrorType
    severity: Literal["minor", "major", "critical"]
    expected_components: tuple[ComponentSpec, ...]
    observed_components: tuple[ComponentSpec, ...]
    expected_nets: dict[str, tuple[str, ...]]
    observed_nets: dict[str, tuple[str, ...]]
    task: str
    error_description: str

    @property
    def expected_values(self) -> dict[str, str | None]:
        return {item.id: item.value for item in self.expected_components}

    @property
    def observed_values(self) -> dict[str, str | None]:
        return {item.id: item.value for item in self.observed_components}


def _error(
    error_id: str,
    difficulty: Literal["basic", "medium", "challenge"],
    error_type: ErrorType,
    severity: Literal["minor", "major", "critical"],
    components: tuple[ComponentSpec, ...],
    expected_nets: dict[str, tuple[str, ...]],
    observed_nets: dict[str, tuple[str, ...]],
    task: str,
    error_description: str,
    observed_components: tuple[ComponentSpec, ...] | None = None,
) -> ErrorCircuitSpec:
    return ErrorCircuitSpec(
        id=error_id,
        title="评分样本",
        difficulty=difficulty,
        error_type=error_type,
        severity=severity,
        expected_components=components,
        observed_components=observed_components or components,
        expected_nets=expected_nets,
        observed_nets=observed_nets,
        task=task,
        error_description=error_description,
    )


ERROR_CIRCUITS: dict[str, ErrorCircuitSpec] = {
    "E01": _error(
        "E01",
        "basic",
        "polarity_reversal",
        "major",
        (
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "330Ω"),
            component("LED1", "diode.light_emitting"),
            component("GND", "gnd"),
        ),
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "LED1.+"),
            "N0": ("LED1.-", "V1.-", "GND.GND"),
        },
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "LED1.-"),
            "N0": ("LED1.+", "V1.-", "GND.GND"),
        },
        "5V 电源驱动 LED，电阻应串联限流且 LED 正极接电阻、负极接地",
        "LED 正负极反接",
    ),
    "E02": _error(
        "E02",
        "basic",
        "series_parallel_confusion",
        "major",
        (
            component("V1", "voltage.dc", "9V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "2.2kΩ"),
            component("GND", "gnd"),
        ),
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1"),
            "N0": ("R2.2", "V1.-", "GND.GND"),
        },
        {
            "N1": ("V1.+", "R1.1", "R2.1"),
            "N0": ("R1.2", "R2.2", "V1.-", "GND.GND"),
        },
        "9V 电源应通过两个电阻串联后接地",
        "两个电阻被画成并联，串联中间节点消失",
    ),
    "E03": _error(
        "E03",
        "medium",
        "numeric_value_error",
        "major",
        (
            component("V1", "voltage.dc", "9V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "2.2kΩ"),
            component("R3", "resistor", "10kΩ"),
            component("GND", "gnd"),
        ),
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1", "R3.1"),
            "N0": ("R2.2", "R3.2", "V1.-", "GND.GND"),
        },
        {
            "N1": ("V1.+", "R1.1"),
            "N2": ("R1.2", "R2.1", "R3.1"),
            "N0": ("R2.2", "R3.2", "V1.-", "GND.GND"),
        },
        "9V 带负载分压器，R2 应为 2.2kΩ",
        "R2 被误标为 22kΩ，连接拓扑仍正确",
        observed_components=(
            component("V1", "voltage.dc", "9V"),
            component("R1", "resistor", "1kΩ"),
            component("R2", "resistor", "22kΩ"),
            component("R3", "resistor", "10kΩ"),
            component("GND", "gnd"),
        ),
    ),
    "E04": _error(
        "E04",
        "challenge",
        "wrong_bjt_port",
        "critical",
        (
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "330Ω"),
            component("R2", "resistor", "10kΩ"),
            component("LED1", "diode.light_emitting"),
            component("Q1", "transistor.bjt"),
            component("GND", "gnd"),
        ),
        {
            "N1": ("V1.+", "R1.1", "R2.1"),
            "N2": ("R1.2", "LED1.+"),
            "N3": ("LED1.-", "Q1.C"),
            "N4": ("R2.2", "Q1.B"),
            "N0": ("Q1.E", "V1.-", "GND.GND"),
        },
        {
            "N1": ("V1.+", "R1.1", "R2.1"),
            "N2": ("R1.2", "LED1.+"),
            "N3": ("LED1.-", "Q1.C"),
            "N4": ("R2.2", "Q1.E"),
            "N0": ("Q1.B", "V1.-", "GND.GND"),
        },
        "BJT 基极由 10kΩ 电阻驱动，发射极接地，集电极驱动 LED",
        "10kΩ 电阻误接到发射极，基极被接到地",
    ),
    "E05": _error(
        "E05",
        "challenge",
        "wrong_junction",
        "critical",
        (
            component("V1", "voltage.dc", "5V"),
            component("R1", "resistor", "100Ω"),
            component("R2", "resistor", "220Ω"),
            component("R3", "resistor", "330Ω"),
            component("R4", "resistor", "470Ω"),
            component("R5", "resistor", "1kΩ"),
            component("GND", "gnd"),
        ),
        {
            "N1": ("V1.+", "R1.1", "R3.1"),
            "NL": ("R1.2", "R2.1", "R5.1"),
            "NR": ("R3.2", "R4.1", "R5.2"),
            "N0": ("R2.2", "R4.2", "V1.-", "GND.GND"),
        },
        {
            "N1": ("V1.+", "R1.1", "R3.1"),
            "NL": ("R1.2", "R2.1", "R5.1", "R5.2"),
            "NR": ("R3.2", "R4.1"),
            "N0": ("R2.2", "R4.2", "V1.-", "GND.GND"),
        },
        "惠斯通电桥的 R5 应连接左右两个中点",
        "R5 的右端误接回左中点，左右中点被错误短接",
    ),
}


def validate_error_circuit(case: ErrorCircuitSpec) -> None:
    """Validate both the intended circuit and the drawn-error circuit."""

    expected = circuit(
        case.id,
        case.title,
        case.difficulty,
        list(case.expected_components),
        case.expected_nets,
    )
    observed = circuit(
        case.id,
        case.title,
        case.difficulty,
        list(case.observed_components),
        case.observed_nets,
    )
    validate_circuit(expected)
    validate_circuit(observed)
    if {item.id for item in case.expected_components} != {
        item.id for item in case.observed_components
    }:
        raise ValueError(f"{case.id}: expected and observed component ids differ")


for _case in ERROR_CIRCUITS.values():
    validate_error_circuit(_case)
