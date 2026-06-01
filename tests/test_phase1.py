import json
from pathlib import Path
from src.phase1_verify.evaluate import evaluate_circuit, load_circuit
from src.phase1_verify.prompts import SYSTEM_PROMPT, EVALUATION_PROMPT


def test_prompt_format_includes_placeholders():
    prompt = EVALUATION_PROMPT.format(circuit_description="TEST")
    assert "TEST" in prompt
    assert "overall_score" in prompt
    assert "fatal_errors" in prompt


def test_load_circuit():
    desc = load_circuit("data/samples/circuit_01_good.txt")
    assert "5V" in desc
    assert "220Ω" in desc


def test_evaluate_good_circuit():
    desc = load_circuit("data/samples/circuit_01_good.txt")
    result = evaluate_circuit(desc)
    assert "overall_score" in result
    assert isinstance(result["overall_score"], int)
    assert 0 <= result["overall_score"] <= 100
    # Good circuit should score high
    assert result["overall_score"] >= 70


def test_evaluate_short_circuit():
    desc = load_circuit("data/samples/circuit_02_short.txt")
    result = evaluate_circuit(desc)
    # Short circuit should have fatal errors
    assert len(result["fatal_errors"]) > 0
    error_types = [e["type"] for e in result["fatal_errors"]]
    assert "short_circuit" in error_types or "reverse_polarity" in error_types
    assert result["overall_score"] < 50


def test_evaluate_no_resistor():
    desc = load_circuit("data/samples/circuit_04_no_r.txt")
    result = evaluate_circuit(desc)
    # Missing current-limiting resistor should be flagged
    all_issues = result["fatal_errors"] + result["correctness_issues"] + result["quality_issues"]
    assert len(all_issues) > 0
    assert result["overall_score"] < 70


def test_evaluate_high_resistor():
    desc = load_circuit("data/samples/circuit_03_high_r.txt")
    result = evaluate_circuit(desc)
    # High resistor should at least flag quality issue
    has_param_issue = any(
        "resistor" in str(e).lower() or "阻值" in str(e) or "电阻" in str(e)
        for e in result["quality_issues"] + result["correctness_issues"]
    )
    assert has_param_issue or result["overall_score"] < 85
