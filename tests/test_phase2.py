import pytest
from src.phase2_pipeline.schema import (
    Circuit, Component, Connection,
    SafetyIssue, CorrectnessIssue, QualityIssue, EvaluationReport
)
from src.phase2_pipeline.safety import (
    check_short_circuits, check_reverse_polarity, check_open_circuits
)
from src.phase2_pipeline.report import _compute_score


def make_test_circuit() -> Circuit:
    """Good LED circuit: 5V → 220Ω → LED → GND"""
    return Circuit(
        name="LED驱动电路",
        expected_function="5V电源驱动LED发光",
        components=[
            Component(id="V1", type="voltage_source", value="5V", nodes=["A", "GND"]),
            Component(id="R1", type="resistor", value="220Ω", nodes=["A", "B"]),
            Component(id="LED1", type="led", value="red, Vf=2.0V", nodes=["B", "GND"]),
        ],
        connections=[
            Connection(from_node="A", to_node="B", via="R1"),
            Connection(from_node="B", to_node="GND", via="LED1"),
        ],
    )


def make_short_circuit() -> Circuit:
    """Short: 5V directly to GND"""
    return Circuit(
        name="短路电路",
        expected_function="5V电源驱动LED发光",
        components=[
            Component(id="V1", type="voltage_source", value="5V", nodes=["A", "GND"]),
            Component(id="LED1", type="led", value="red", nodes=["A", "GND"]),
        ],
        connections=[
            Connection(from_node="A", to_node="GND", via="LED1"),
        ],
    )


def make_reverse_circuit() -> Circuit:
    """LED reversed: cathode to VCC"""
    return Circuit(
        name="LED反接电路",
        expected_function="5V电源驱动LED发光",
        components=[
            Component(id="V1", type="voltage_source", value="5V", nodes=["A", "GND"]),
            Component(id="LED1", type="led", value="red", nodes=["GND", "A"]),
        ],
        connections=[
            Connection(from_node="A", to_node="GND", via="LED1"),
        ],
    )


class TestSafetyChecker:
    def test_good_circuit_no_short(self):
        circuit = make_test_circuit()
        issues = check_short_circuits(circuit)
        assert len(issues) == 0

    def test_reverse_polarity_detected(self):
        circuit = make_reverse_circuit()
        issues = check_reverse_polarity(circuit)
        assert len(issues) > 0
        assert issues[0].type == "reverse_polarity"

    def test_open_circuit(self):
        circuit = Circuit(
            name="断路",
            expected_function="驱动LED",
            components=[
                Component(id="V1", type="voltage_source", value="5V", nodes=["A", "GND"]),
                Component(id="R1", type="resistor", value="220Ω", nodes=["B", "C"]),
                Component(id="LED1", type="led", value="red", nodes=["D", "E"]),
            ],
            connections=[],
        )
        issues = check_open_circuits(circuit)
        # Many floating nodes
        assert len(issues) >= 3


class TestScoring:
    def test_perfect_score(self):
        score = _compute_score([], [], [])
        assert score == 100

    def test_critical_deduction(self):
        score = _compute_score(
            [SafetyIssue(type="short_circuit", severity="critical", nodes=["A"], description="短路")],
            [], []
        )
        assert score == 70  # 100 - 30

    def test_floor_at_zero(self):
        score = _compute_score(
            [SafetyIssue(type="short_circuit", severity="critical", nodes=["A"], description="e") for _ in range(10)],
            [], []
        )
        assert score == 0


class TestSchema:
    def test_circuit_validation(self):
        circuit = make_test_circuit()
        assert circuit.name == "LED驱动电路"
        assert len(circuit.components) == 3

    def test_report_validation(self):
        report = EvaluationReport(
            overall_score=85,
            safety_issues=[],
            correctness_issues=[],
            quality_issues=[],
            summary="电路良好",
        )
        assert report.overall_score == 85
