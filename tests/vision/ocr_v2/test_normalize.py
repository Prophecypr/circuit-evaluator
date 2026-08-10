from src.vision.ocr_v2.normalize import (
    CANONICAL_CHARS,
    is_value_label,
    normalize_label,
    validate_normalized_label,
)


def test_normalize_unit_aliases_without_collapsing_mega_and_milli():
    assert normalize_label(" 10KΩ ") == "10kΩ"
    assert normalize_label("4.7uF") == "4.7μF"
    assert normalize_label("4.7µF") == "4.7μF"
    assert normalize_label("1MΩ") == "1MΩ"
    assert normalize_label("1mA") == "1mA"


def test_normalize_ohm_aliases_and_internal_whitespace():
    assert normalize_label("  220 Ω ") == "220Ω"
    assert normalize_label("1 ω") == "1Ω"


def test_value_filter_accepts_circuit_values_and_rejects_ic_identifiers():
    for value in ("10kΩ", "3V3", "4.7μF", "100nF", "12V", "2mA", "50Hz"):
        assert is_value_label(value), value
    for identifier in ("LM358", "NE555", "GND", "R1", "LED"):
        assert not is_value_label(identifier), identifier


def test_charset_is_unique_and_validation_reports_unsupported_characters():
    assert len(CANONICAL_CHARS) == len(set(CANONICAL_CHARS))
    assert validate_normalized_label("4.7μF") == "4.7μF"
    try:
        validate_normalized_label("10?Ω")
    except ValueError as exc:
        assert "?" in str(exc)
    else:
        raise AssertionError("unsupported character was accepted")
