"""Phase 4: Netlist-First Pipeline.

Key insight: Replace the custom JSON intermediate representation with SPICE netlist.
- LLM generates SPICE netlist from natural language (LLMs know SPICE well)
- Deterministic parser extracts structured data (no LLM hallucination risk)
- Rule-based safety checks on parsed data
- LLM evaluates from netlist (more precise than natural language)
"""

import re
from dataclasses import dataclass, field
from src.llm import ask, extract_json


# -- Deterministic SPICE Parser (no LLM involved) --

@dataclass
class ParsedComponent:
    id: str
    type: str  # R, C, L, D, V, I, Q, E, etc.
    nodes: list[str]
    value: str | None = None
    model: str | None = None

@dataclass
class ParsedNetlist:
    title: str
    components: list[ParsedComponent] = field(default_factory=list)
    analysis: str = "op"
    raw: str = ""

    @property
    def component_list_text(self) -> str:
        lines = []
        for c in self.components:
            extra = f", value={c.value}" if c.value else ""
            extra += f", model={c.model}" if c.model else ""
            lines.append(f"  {c.id}: {c.type}, nodes={c.nodes}{extra}")
        return "\n".join(lines)


def parse_netlist(netlist: str) -> ParsedNetlist:
    """Deterministically parse a SPICE netlist into structured data."""
    result = ParsedNetlist(title="", raw=netlist)
    lines = netlist.split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("*"):
            if line.startswith("* ") and not result.title:
                result.title = line[2:].strip()
            continue
        if line.startswith("."):
            if line.upper().startswith(".OP"):
                result.analysis = "op"
            elif line.upper().startswith(".DC"):
                result.analysis = "dc"
            elif line.upper().startswith(".TRAN"):
                result.analysis = "tran"
            elif line.upper().startswith(".END"):
                break
            continue

        # Parse component line: <id> <node1> <node2> [node3...] <value> [model]
        parts = line.split()
        if len(parts) < 3:
            continue

        comp_id = parts[0]
        first_char = comp_id[0].upper()

        # Determine type from first letter
        type_map = {"R": "resistor", "C": "capacitor", "L": "inductor",
                     "D": "diode", "V": "voltage_source", "I": "current_source",
                     "Q": "transistor", "M": "mosfet", "E": "vcvs",
                     "G": "vccs", "H": "ccvs", "F": "cccs",
                     "X": "subcircuit", "S": "switch"}

        comp_type = type_map.get(first_char, "unknown")

        # Extract nodes and value
        if comp_type in ("transistor", "mosfet"):
            if len(parts) >= 4:
                nodes = parts[1:4]
                model = parts[4] if len(parts) > 4 else None
                result.components.append(ParsedComponent(
                    id=comp_id, type=comp_type, nodes=nodes, model=model))
        elif comp_type == "diode":
            if len(parts) >= 3:
                nodes = parts[1:3]
                model = parts[3] if len(parts) > 3 else None
                result.components.append(ParsedComponent(
                    id=comp_id, type=comp_type, nodes=nodes, model=model))
        elif comp_type in ("voltage_source", "current_source"):
            if len(parts) >= 4:
                nodes = parts[1:3]
                # parts[3] might be "DC" followed by value, or just value
                if parts[3].upper() == "DC" and len(parts) > 4:
                    value = parts[4]
                elif parts[3].upper() == "AC":
                    value = parts[4] if len(parts) > 4 else "AC"
                else:
                    value = parts[3]
                result.components.append(ParsedComponent(
                    id=comp_id, type=comp_type, nodes=nodes, value=value))
        else:
            # 2-node components with value: R, C, L
            if len(parts) >= 3:
                nodes = parts[1:3]
                value = parts[3] if len(parts) > 3 else None
                result.components.append(ParsedComponent(
                    id=comp_id, type=comp_type, nodes=nodes, value=value))

    return result


# -- Netlist Generation via LLM --

NETLIST_SYSTEM = """你是一位资深的SPICE仿真工程师。将电路文字描述转换为标准SPICE网表。
只输出SPICE网表，不要有多余说明。网表必须语法正确。"""

NETLIST_PROMPT = """请将以下电路描述转换为SPICE网表：

{description}

要求：
1. 使用 .OP 进行工作点分析
2. 节点编号：0为GND，从1开始编号其他节点
3. 在注释中标注节点映射关系
4. 对于LED，使用标准二极管模型：.MODEL LED D (IS=1e-18 N=2.0 RS=5)
5. 对于NPN三极管，使用：.MODEL NPN NPN (BF=200)
6. 电压源格式：Vname n+ n- DC value
7. 输出标准SPICE格式，以.END结尾

只输出网表，不要输出其他内容。"""


def generate_netlist_from_description(description: str, model: str = "claude-sonnet-4-6") -> str:
    """LLM generates SPICE netlist directly from natural language description."""
    prompt = NETLIST_PROMPT.format(description=description)
    response = ask(prompt, system=NETLIST_SYSTEM, model=model)
    # Clean up: extract just the netlist
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove ```spice or ``` and trailing ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    return response.strip()


# -- Netlist-Based Evaluation --

EVAL_FROM_NETLIST_SYSTEM = """你是一位资深电路评审专家。根据SPICE网表对电路进行专业评价。
网表是电路精确的数学描述，请基于它做出准确判断。"""

EVAL_FROM_NETLIST_PROMPT = """评价以下电路：

电路描述:
{description}

SPICE网表:
{netlist}

解析出的元件:
{components}

请评价（按以下JSON格式输出）：
{{
  "overall_score": <0-100>,
  "fatal_errors": [
    {{"type": "short_circuit|open_circuit|reverse_polarity|other", "severity": "critical", "description": "<说明>"}}
  ],
  "correctness_issues": [
    {{"type": "missing_component|wrong_topology|wrong_value", "severity": "major|minor", "description": "<说明>", "suggestion": "<建议>"}}
  ],
  "quality_issues": [
    {{"type": "parameter_suboptimal|efficiency|robustness|cost", "severity": "minor|suggestion", "description": "<说明>", "suggestion": "<建议>"}}
  ],
  "summary": "<50字总结>"
}}
如果某类问题不存在，对应数组留空 []。不要输出其他内容。"""


def evaluate_from_netlist(description: str, model: str = "claude-sonnet-4-6") -> dict:
    """Full Phase 4 pipeline: NL → SPICE netlist → parse → LLM evaluation."""
    # Step 1: Generate netlist via LLM
    netlist = generate_netlist_from_description(description, model=model)

    # Step 2: Deterministic parse (no LLM)
    parsed = parse_netlist(netlist)

    # Step 3: LLM evaluates from netlist + parsed components
    prompt = EVAL_FROM_NETLIST_PROMPT.format(
        description=description[:2000],
        netlist=netlist,
        components=parsed.component_list_text,
    )
    response = ask(prompt, system=EVAL_FROM_NETLIST_SYSTEM, model=model)
    import json
    from src.llm import extract_json
    json_str = extract_json(response)
    return json.loads(json_str)


def evaluate_and_print(description: str, model: str = "claude-sonnet-4-6") -> dict:
    """Run Phase 4 and print results."""
    result = evaluate_from_netlist(description, model=model)

    print(f"\n{'='*60}")
    print("Phase 4: Netlist-First 评价")
    print(f"{'='*60}")
    print(f"评分: {result['overall_score']}/100")
    print(f"摘要: {result.get('summary', 'N/A')}")

    if result.get("fatal_errors"):
        print(f"\n致命错误 ({len(result['fatal_errors'])}):")
        for e in result["fatal_errors"]:
            print(f"  🔴 [{e['type']}] {e['description']}")

    if result.get("correctness_issues"):
        print(f"\n正确性问题 ({len(result['correctness_issues'])}):")
        for e in result["correctness_issues"]:
            icon = "🟠" if e.get("severity") == "major" else "🟡"
            print(f"  {icon} [{e['type']}] {e['description']}")

    if result.get("quality_issues"):
        print(f"\n质量问题 ({len(result['quality_issues'])}):")
        for e in result["quality_issues"]:
            print(f"  🔵 [{e['type']}] {e['description']}")

    return result


# -- Simulation-Enhanced Evaluation (Phase 4.5) --

SIM_EVAL_PROMPT = """评价以下电路。以下是SPICE仿真结果，请基于实际仿真数据做出准确判断：

电路描述:
{description}

SPICE网表:
{netlist}

仿真输出:
{simulation_output}

请评价（按以下JSON格式输出）：
{{
  "overall_score": <0-100>,
  "simulation_ok": true/false,
  "node_voltages": {{"<节点>": <电压(V)>}},
  "fatal_errors": [
    {{"type": "short_circuit|open_circuit|reverse_polarity|other", "severity": "critical", "description": "<说明>"}}
  ],
  "correctness_issues": [
    {{"type": "missing_component|wrong_topology|wrong_value", "severity": "major|minor", "description": "<说明>", "suggestion": "<建议>"}}
  ],
  "quality_issues": [
    {{"type": "parameter_suboptimal|efficiency|robustness|cost", "severity": "minor|suggestion", "description": "<说明>", "suggestion": "<建议>"}}
  ],
  "summary": "<50字总结>"
}}
如果某类问题不存在，对应数组留空 []。不要输出其他内容。"""


def evaluate_with_simulation(description: str, model: str = "claude-sonnet-4-6") -> dict:
    """Phase 4.5: Netlist + actual SPICE simulation + LLM interpretation.

    Runs ngspice if available, falls back to LLM-only if not.
    """
    from src.phase3_spice.simulate import _run_ngspice, _find_ngspice

    # Step 1: Generate netlist
    netlist = generate_netlist_from_description(description, model=model)

    # Step 2: Try actual simulation
    ngspice_path = _find_ngspice()
    if ngspice_path:
        sim_output = _run_ngspice(netlist)
        if "NGSPICE_NOT_FOUND" not in sim_output:
            # Simulation succeeded or gave meaningful error
            prompt = SIM_EVAL_PROMPT.format(
                description=description[:2000],
                netlist=netlist,
                simulation_output=sim_output[:3000],
            )
            response = ask(prompt, system=EVAL_FROM_NETLIST_SYSTEM, model=model)
            import json
            from src.llm import extract_json
            json_str = extract_json(response)
            result = json.loads(json_str)
            result["_simulation_used"] = True
            return result

    # Fallback: LLM-only evaluation from netlist
    result = evaluate_from_netlist(description, model=model)
    result["_simulation_used"] = False
    return result


if __name__ == "__main__":
    from pathlib import Path
    import sys

    if len(sys.argv) > 1:
        desc = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        desc = Path("data/samples/circuit_01_good.txt").read_text(encoding="utf-8")

    # Show generated netlist
    netlist = generate_netlist_from_description(desc)
    print("Generated SPICE Netlist:")
    print(netlist)
    print()

    # Parse it
    parsed = parse_netlist(netlist)
    print(f"Parsed: {len(parsed.components)} components")
    print(parsed.component_list_text)
    print()

    # Evaluate
    evaluate_and_print(desc)
