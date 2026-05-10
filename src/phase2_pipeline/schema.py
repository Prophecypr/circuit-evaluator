from pydantic import BaseModel, Field
from typing import Literal


class Component(BaseModel):
    id: str = Field(description="Unique component identifier, e.g. R1, LED1, V1")
    type: Literal[
        "resistor", "capacitor", "inductor", "diode", "led",
        "voltage_source", "current_source", "transistor_npn",
        "transistor_pnp", "mosfet_n", "mosfet_p", "opamp",
        "switch", "ground", "wire"
    ]
    value: str | None = Field(default=None, description="Component value, e.g. 220Ω, 10μF")
    nodes: list[str] = Field(description="Node IDs this component connects to, ordered for polarized components")


class Connection(BaseModel):
    from_node: str
    to_node: str
    via: str | None = Field(default=None, description="Component ID if via a component, None if direct wire")


class Circuit(BaseModel):
    name: str
    expected_function: str = Field(description="What the circuit is supposed to do")
    components: list[Component]
    connections: list[Connection]
    notes: str | None = Field(default=None, description="Additional context")


class SafetyIssue(BaseModel):
    type: Literal["short_circuit", "open_circuit", "reverse_polarity", "power_conflict", "overcurrent_risk"]
    severity: Literal["critical"]
    nodes: list[str]
    description: str


class CorrectnessIssue(BaseModel):
    type: Literal["missing_component", "unnecessary_component", "wrong_topology", "unrealizable_function", "wrong_value"]
    severity: Literal["major", "minor"]
    components: list[str]
    description: str
    suggestion: str


class QualityIssue(BaseModel):
    type: Literal["parameter_suboptimal", "efficiency", "robustness", "cost", "thermal"]
    severity: Literal["minor", "suggestion"]
    components: list[str]
    description: str
    suggestion: str


class EvaluationReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    safety_issues: list[SafetyIssue]
    correctness_issues: list[CorrectnessIssue]
    quality_issues: list[QualityIssue]
    summary: str
