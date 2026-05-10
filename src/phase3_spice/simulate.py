"""Run SPICE simulations and interpret results with LLM."""

import subprocess
import tempfile
import os
from pathlib import Path
from src.phase2_pipeline.schema import Circuit
from src.phase3_spice.netlist import generate_netlist
from src.llm import ask

SIM_INTERPRETER_SYSTEM = """你是一位资深的电路仿真分析专家。根据SPICE仿真结果对电路进行专业评价。
如果仿真成功，解读各节点的电压电流是否在合理范围。
如果仿真失败，分析可能的原因。
始终以JSON格式输出。"""

SIM_INTERPRETER_PROMPT = """分析以下电路的SPICE仿真结果：

电路信息:
  名称: {name}
  预期功能: {expected_function}

SPICE Netlist:
{netlist}

仿真输出:
{simulation_output}

请按以下JSON格式输出分析：
{{
  "simulation_success": true/false,
  "node_voltages": {{
    "<节点名>": <电压值(V)>
  }},
  "branch_currents": {{
    "<元件名>": <电流值(A)>
  }},
  "issues_found": [
    {{
      "type": "overvoltage|undervoltage|overcurrent|bias_point_wrong|oscillation|other",
      "severity": "critical|major|minor",
      "component": "<元件ID>",
      "description": "<问题描述>",
      "expected": "<期望值>",
      "actual": "<实际值>"
    }}
  ],
  "summary": "<仿真结果总结>"
}}

如果仿真失败，issues_found中说明可能原因。"""


def _run_ngspice(netlist: str) -> str:
    """Run ngspice with the given netlist, return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False, encoding="utf-8") as f:
        f.write(netlist)
        netlist_path = f.name

    try:
        result = subprocess.run(
            ["ngspice", "-b", netlist_path],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout + "\n" + result.stderr
    except FileNotFoundError:
        return "NGSPICE_NOT_FOUND: Ngspice is not installed. Install from https://ngspice.sourceforge.io/"
    except subprocess.TimeoutExpired:
        return "SIMULATION_TIMEOUT: Simulation took too long (possible convergence issue)"
    finally:
        try:
            os.unlink(netlist_path)
        except OSError:
            pass


def run_simulation(circuit: Circuit, analysis: str = "op", model: str = "claude-sonnet-4-6") -> dict:
    """Run SPICE simulation on a circuit and interpret results with LLM.

    Returns a dict with keys: simulation_success, node_voltages, issues_found, summary
    """
    netlist = generate_netlist(circuit, analysis=analysis)
    sim_output = _run_ngspice(netlist)

    prompt = SIM_INTERPRETER_PROMPT.format(
        name=circuit.name,
        expected_function=circuit.expected_function,
        netlist=netlist,
        simulation_output=sim_output,
    )

    from src.llm import extract_json
    response = ask(prompt, system=SIM_INTERPRETER_SYSTEM, model=model)
    import json
    json_str = extract_json(response)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {
            "simulation_success": False,
            "node_voltages": {},
            "branch_currents": {},
            "issues_found": [],
            "summary": f"Failed to parse LLM response. Raw simulation output:\n{sim_output[:1000]}",
        }


def simulate_and_report(circuit: Circuit, model: str = "claude-sonnet-4-6") -> None:
    """Run simulation and print a formatted report."""
    result = run_simulation(circuit, model=model)

    print(f"\n{'='*60}")
    print(f"SPICE 仿真报告: {circuit.name}")
    print(f"{'='*60}")

    if not result["simulation_success"]:
        print("❌ 仿真失败或不可用")
        if result.get("issues_found"):
            for issue in result["issues_found"]:
                print(f"  - {issue.get('description', str(issue))}")
    else:
        print("✅ 仿真成功")
        if result.get("node_voltages"):
            print("\n节点电压:")
            for node, voltage in result["node_voltages"].items():
                print(f"  {node}: {voltage}")

    print(f"\n分析: {result.get('summary', 'N/A')}")

    if result.get("issues_found"):
        print(f"\n发现 {len(result['issues_found'])} 个问题:")
        for issue in result["issues_found"]:
            sev = issue.get("severity", "unknown")
            icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(sev, "⚪")
            print(f"  {icon} [{issue.get('type')}] {issue.get('description')}")


if __name__ == "__main__":
    from src.phase2_pipeline.converter import convert_to_circuit
    from pathlib import Path

    desc = Path("data/samples/circuit_01_good.txt").read_text(encoding="utf-8")
    circuit = convert_to_circuit(desc)
    print("Generated Netlist:")
    print(generate_netlist(circuit))
    print()
    simulate_and_report(circuit)
