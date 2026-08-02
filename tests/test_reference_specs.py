from collections import Counter

import pytest

from src.reference_pack.specs import (
    CIRCUITS,
    ComponentSpec,
    CircuitSpec,
    circuit,
    component,
    validate_circuit,
)


def _invalid_spec(
    circuit_id: str,
    components: list[ComponentSpec],
    nets: dict[str, tuple[str, ...]],
) -> CircuitSpec:
    return circuit(circuit_id, "Invalid test circuit", "basic", components, nets)


def test_reference_pack_has_confirmed_difficulty_balance():
    assert list(CIRCUITS) == [f"C{i:02d}" for i in range(1, 11)]
    assert Counter(circuit.difficulty for circuit in CIRCUITS.values()) == {
        "basic": 4,
        "medium": 4,
        "challenge": 2,
    }


def test_every_port_belongs_to_exactly_one_net():
    for circuit in CIRCUITS.values():
        validate_circuit(circuit)


def test_resistors_use_ansi_and_values_fit_current_ocr_scope():
    allowed = set("0123456789.kKmMΩμunpFVAHz-+")
    for circuit in CIRCUITS.values():
        for component in circuit.components:
            if component.class_name == "resistor":
                assert component.symbol == "ansi_zigzag"
            if component.value:
                assert set(component.value) <= allowed


def test_c09_uses_generic_bjt_family_with_bce_ports():
    transistor = next(
        component
        for component in CIRCUITS["C09"].components
        if component.class_name == "transistor.bjt"
    )
    assert transistor.ports == ("B", "C", "E")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        pytest.param(
            _invalid_spec(
                "INVALID_DUPLICATE_COMPONENT",
                [component("R1", "resistor"), component("R1", "resistor")],
                {"N1": ("R1.1", "R1.2")},
            ),
            "duplicate component id",
            id="duplicate-component-id",
        ),
        pytest.param(
            _invalid_spec(
                "INVALID_MULTIPLE_NETS",
                [component("R1", "resistor"), component("R2", "resistor")],
                {
                    "N1": ("R1.1", "R2.1"),
                    "N2": ("R1.1", "R1.2", "R2.2"),
                },
            ),
            "duplicate port reference",
            id="port-in-multiple-nets",
        ),
        pytest.param(
            _invalid_spec(
                "INVALID_UNKNOWN_PORT",
                [component("R1", "resistor")],
                {"N1": ("R1.1", "X1.1")},
            ),
            "undeclared port reference",
            id="unknown-port-ref",
        ),
        pytest.param(
            _invalid_spec(
                "INVALID_MISSING_PORT",
                [component("R1", "resistor"), component("R2", "resistor")],
                {"N1": ("R1.1", "R2.1")},
            ),
            "ports missing from nets",
            id="missing-declared-port",
        ),
        pytest.param(
            _invalid_spec(
                "INVALID_EMPTY_NET",
                [component("R1", "resistor")],
                {"N1": ()},
            ),
            "net N1 must contain at least 2 ports",
            id="empty-net",
        ),
        pytest.param(
            _invalid_spec(
                "INVALID_ONE_PORT_NET",
                [component("R1", "resistor")],
                {"N1": ("R1.1",)},
            ),
            "net N1 must contain at least 2 ports",
            id="one-port-net",
        ),
    ],
)
def test_validate_circuit_rejects_invalid_net_partitions(
    spec: CircuitSpec,
    message: str,
):
    with pytest.raises(ValueError, match=message) as exc_info:
        validate_circuit(spec)

    assert str(exc_info.value).startswith(f"{spec.id}:")


def test_validate_circuit_rejects_duplicate_component_port_labels():
    duplicate_port_component = ComponentSpec(
        id="X1",
        class_name="custom",
        value=None,
        ports=("1", "1"),
        symbol="custom",
    )
    spec = _invalid_spec(
        "INVALID_DUPLICATE_PORT_LABEL",
        [duplicate_port_component, component("GND", "gnd")],
        {"N1": ("X1.1", "GND.GND")},
    )

    with pytest.raises(ValueError, match="duplicate port label") as exc_info:
        validate_circuit(spec)

    assert str(exc_info.value).startswith(f"{spec.id}:")
