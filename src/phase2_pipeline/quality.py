from src.phase2_pipeline.schema import Circuit, QualityIssue
from src.llm import ask_json

QUALITY_PROMPT = """评估以下电路的质量（假设电路结构和元件值在功能上是正确的）：

电路名称: {name}
预期功能: {expected_function}

元件列表:
{components}

连接关系:
{connections}

从以下维度评估：
1. 参数优化：元件值是否可以调整以获得更好性能？
2. 功耗效率：是否存在不必要的功耗？
3. 鲁棒性：是否考虑了元件公差、温度影响？
4. 成本：是否有更经济的替代方案？

以JSON数组格式输出（只输出存在问题的项）：
[
  {{
    "type": "parameter_suboptimal|efficiency|robustness|cost|thermal",
    "severity": "minor|suggestion",
    "components": ["R1"],
    "description": "<问题描述>",
    "suggestion": "<优化建议>"
  }}
]
如果没有优化建议，输出空数组 []。不要输出其他内容。"""


def evaluate_quality(circuit: Circuit, model: str = "claude-sonnet-4-6") -> list[QualityIssue]:
    comp_lines = "\n".join(
        f"- {c.id}: {c.type}, value={c.value}, nodes={c.nodes}" for c in circuit.components
    )
    conn_lines = "\n".join(
        f"- {c.from_node} → {c.to_node}" + (f" via {c.via}" if c.via else "")
        for c in circuit.connections
    )
    prompt = QUALITY_PROMPT.format(
        name=circuit.name,
        expected_function=circuit.expected_function,
        components=comp_lines,
        connections=conn_lines,
    )
    data = ask_json(prompt, model=model)
    issues = []
    for item in data:
        # Normalize severity: quality issues should be minor or suggestion
        if item.get("severity") not in ("minor", "suggestion"):
            item["severity"] = "minor"
        issues.append(QualityIssue(**item))
    return issues
