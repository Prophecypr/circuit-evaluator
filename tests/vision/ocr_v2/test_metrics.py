from src.vision.ocr_v2.metrics import cer, evaluate_pairs, parse_value


def test_cer_can_exceed_one_and_uses_total_reference_characters():
    assert cer("1", "111") == 2.0
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "adc") == 1 / 3


def test_metric_report_separates_numeric_and_unit_accuracy():
    report = evaluate_pairs([("10kΩ", "10MΩ"), ("4.7μF", "4.7μF")])
    assert report["numeric_exact"] == 1.0
    assert report["unit_exact"] == 0.5
    assert report["normalized_exact"] == 0.5
    assert report["per_unit"]["kΩ"] == {"count": 1, "exact": 0.0}
    assert len(report["errors"]) == 1


def test_raw_and_normalized_metrics_are_both_retained():
    report = evaluate_pairs([("4.7uF", "4.7μF"), ("10KΩ", "10kΩ")])
    assert report["raw_exact"] == 0.0
    assert report["normalized_exact"] == 1.0
    assert report["raw_cer"] > 0.0
    assert report["normalized_cer"] == 0.0


def test_parse_value_handles_embedded_voltage_notation():
    assert parse_value("3V3") == ("3.3", "V")
    assert parse_value("50Hz") == ("50", "Hz")


def test_unit_accuracy_excludes_ground_truth_without_units():
    report = evaluate_pairs([("12", ""), ("10kΩ", "")])
    assert report["unit_labeled_count"] == 1
    assert report["unit_exact"] == 0.0
