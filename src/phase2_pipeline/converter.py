from src.phase2_pipeline.schema import Circuit
from src.llm import ask, extract_json

CONVERTER_SYSTEM = """你是一个电路描述解析器。将自然语言电路描述转换为结构化JSON。
严格按照JSON Schema输出，不要输出任何其他内容。"""

CONVERTER_PROMPT = """将以下电路描述转换为结构化JSON：

{description}

输出格式：
{{
  "name": "<电路名称>",
  "expected_function": "<预期功能>",
  "components": [
    {{"id": "R1", "type": "resistor", "value": "220Ω", "nodes": ["A", "B"]}},
    {{"id": "LED1", "type": "led", "value": "red, Vf=2.0V", "nodes": ["B", "GND"]}}
  ],
  "connections": [
    {{"from_node": "A", "to_node": "B", "via": "R1"}},
    {{"from_node": "B", "to_node": "GND", "via": "LED1"}}
  ],
  "notes": "<额外说明，没有则为空字符串>"
}}

规则：
1. 为每个电气节点分配唯一ID（如 A, B, C... GND, VCC 等）
2. 极性元件（LED、电解电容、二极管）的 nodes[0] 是正极/阳极，nodes[1] 是负极/阴极
3. 电源的 nodes[0] 是正极，nodes[1] 是负极
4. 无极性元件（电阻、电容）节点顺序无关
5. type 必须是以下之一：resistor, capacitor, inductor, diode, led, voltage_source, current_source, transistor_npn, transistor_pnp, mosfet_n, mosfet_p, opamp, switch, ground, wire
"""


def convert_to_circuit(description: str, model: str = "claude-sonnet-4-6") -> Circuit:
    prompt = CONVERTER_PROMPT.format(description=description)
    response = ask(prompt, system=CONVERTER_SYSTEM, model=model)
    json_str = extract_json(response)
    try:
        return Circuit.model_validate_json(json_str)
    except Exception:
        response = ask(prompt + "\n\nIMPORTANT: Output ONLY valid JSON matching the specified format.", system=CONVERTER_SYSTEM, model=model)
        json_str = extract_json(response)
        return Circuit.model_validate_json(json_str)
