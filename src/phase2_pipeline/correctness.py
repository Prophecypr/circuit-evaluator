from src.phase2_pipeline.schema import Circuit, CorrectnessIssue
from src.llm import ask_json

CORRECTNESS_SYSTEM = """你是一位电路评审专家。基于电路的结构化描述，检查电路正确性。
只关注功能层面的问题：元件是否缺失/多余、拓扑是否正确、参数是否在合理范围。
请以JSON数组格式输出，每个问题包含 type, severity, components, description, suggestion。"""

CORRECTNESS_PROMPT = """检查以下电路的正确性：

电路名称: {name}
预期功能: {expected_function}

元件列表:
{components}

连接关系:
{connections}

请检查：
1. 是否缺少实现预期功能所必需的元件？（如LED电路缺少限流电阻）
2. 是否有不必要的元件？
3. 拓扑结构是否可以实现预期功能？
4. 关键元件参数值是否在合理范围？（如LED限流电阻应在100Ω-1kΩ之间，10kΩ过大）

以JSON数组格式输出问题列表：
[
  {{
    "type": "missing_component|unnecessary_component|wrong_topology|unrealizable_function|wrong_value",
    "severity": "major|minor",
    "components": ["R1"],
    "description": "<问题描述>",
    "suggestion": "<修改建议>"
  }}
]
如果没有问题，输出空数组 []。不要输出其他内容。"""


def evaluate_correctness(circuit: Circuit, model: str = "claude-sonnet-4-6") -> list[CorrectnessIssue]:
    comp_lines = "\n".join(
        f"- {c.id}: {c.type}, value={c.value}, nodes={c.nodes}" for c in circuit.components
    )
    conn_lines = "\n".join(
        f"- {c.from_node} → {c.to_node}" + (f" via {c.via}" if c.via else "")
        for c in circuit.connections
    )
    prompt = CORRECTNESS_PROMPT.format(
        name=circuit.name,
        expected_function=circuit.expected_function,
        components=comp_lines,
        connections=conn_lines,
    )
    data = ask_json(prompt, system=CORRECTNESS_SYSTEM, model=model)
    issues = []
    for item in data:
        if item.get("severity") not in ("major", "minor"):
            item["severity"] = "minor"
        issues.append(CorrectnessIssue(**item))
    return issues
