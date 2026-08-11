from pathlib import Path

import pytest


def test_error_pack_has_five_balanced_independent_error_cases():
    from src.reference_pack.error_specs import ERROR_CIRCUITS, validate_error_circuit

    assert len(ERROR_CIRCUITS) == 5
    assert {case.error_type for case in ERROR_CIRCUITS.values()} == {
        "polarity_reversal",
        "series_parallel_confusion",
        "numeric_value_error",
        "wrong_bjt_port",
        "wrong_junction",
    }
    for case in ERROR_CIRCUITS.values():
        validate_error_circuit(case)
        assert case.expected_nets != case.observed_nets or (
            case.expected_values != case.observed_values
        )


def test_error_pack_has_explicit_port_connections_and_target_values():
    from src.reference_pack.error_specs import ERROR_CIRCUITS

    for case in ERROR_CIRCUITS.values():
        assert case.expected_nets
        assert case.observed_nets
        assert set(case.expected_values) == set(case.observed_values)
        assert all("." in port for refs in case.observed_nets.values() for port in refs)


def test_reversed_led_routes_avoid_running_wire_through_led_symbol():
    from generate_error_reference_pack import _observed_circuit
    from src.reference_pack.error_layouts import ERROR_LAYOUTS
    from src.reference_pack.error_specs import ERROR_CIRCUITS
    from src.reference_pack.render import draw_reference

    case = ERROR_CIRCUITS["E01"]
    canvas = draw_reference(_observed_circuit(case), ERROR_LAYOUTS["E01"])
    try:
        path = canvas.route_records["N2"][1]
        assert path == [(700, 420), (820, 420), (820, 480)]
        assert (700, 280) not in path
    finally:
        canvas.figure.clf()


def test_error_pack_rejects_unconnected_observed_port():
    from src.reference_pack.error_specs import ERROR_CIRCUITS, validate_error_circuit

    case = ERROR_CIRCUITS["E01"]
    broken = case.__class__(
        **{
            **case.__dict__,
            "observed_nets": {
                **case.observed_nets,
                "BROKEN": ("LED1.+",),
            },
        }
    )
    with pytest.raises(ValueError, match="must contain at least 2 ports"):
        validate_error_circuit(broken)


def test_generate_error_pack_is_complete_and_byte_deterministic(tmp_path: Path):
    from generate_error_reference_pack import generate_error_pack

    first = generate_error_pack(tmp_path / "first")
    second = generate_error_pack(tmp_path / "second")
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert len(first_files) == 19
    assert set(first_files) == set(second_files)
    assert first_files == second_files
