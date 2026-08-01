from collections import Counter

from src.reference_pack.specs import CIRCUITS, validate_circuit


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
