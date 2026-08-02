"""Generate the five independent error cases for the balanced LLM scoring set."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from generate_blind_reference_pack import _write_text, write_checksums
from src.reference_pack.error_layouts import ERROR_LAYOUTS
from src.reference_pack.error_specs import ERROR_CIRCUITS, ErrorCircuitSpec
from src.reference_pack.render import draw_reference
from src.reference_pack.specs import circuit


SCORING_PROTOCOL = """# LLM 评分错误样本包

本目录包含 5 张独立错误图。正确图不重复生成，直接从
`benchmark/blind_reference_pack/v1/references/` 中选取 5 张，组成 5 张正确 + 5 张错误的均衡评分集。

错误图类型：LED 极性反接、串并联混淆、电阻数值错误、BJT 端口错误、惠斯通电桥结点错误。

LLM 评分时必须同时提供每张图对应的目标任务要求；不能只让模型凭“像不像电路”打分。
模型输入应包括识别出的元件、数值和连接图，输出错误类型、严重程度、理由和评分。
评分真值位于 `score_key/`，该目录不得提供给模型或人工评分者。
"""


MANIFEST = "sample_id,source_domain,error_type,image_path,score_key_path\n"


def _observed_circuit(case: ErrorCircuitSpec):
    return circuit(
        case.id,
        case.title,
        case.difficulty,
        list(case.observed_components),
        case.observed_nets,
    )


def _score_key(case: ErrorCircuitSpec) -> dict:
    return {
        "sample_id": case.id,
        "title": case.title,
        "difficulty": case.difficulty,
        "error_type": case.error_type,
        "severity": case.severity,
        "task": case.task,
        "error_description": case.error_description,
        "expected_components": [asdict(item) for item in case.expected_components],
        "observed_components": [asdict(item) for item in case.observed_components],
        "expected_values": case.expected_values,
        "observed_values": case.observed_values,
        "expected_nets": case.expected_nets,
        "observed_nets": case.observed_nets,
    }


def _port_connections(case: ErrorCircuitSpec) -> str:
    def format_nets(nets: dict[str, tuple[str, ...]]) -> str:
        return "\n".join(
            f"- `{net_id}`: " + ", ".join(refs)
            for net_id, refs in nets.items()
        )

    return (
        f"## {case.id}｜{case.error_type}\n\n"
        f"目标：{case.task}\n\n"
        f"实际错误：{case.error_description}\n\n"
        "### 目标连接\n\n"
        f"{format_nets(case.expected_nets)}\n\n"
        "### 手画/识别到的连接\n\n"
        f"{format_nets(case.observed_nets)}\n"
    )


def generate_error_pack(output_dir: str | Path) -> Path:
    """Generate a non-overwriting, deterministic five-error scoring pack."""

    output_dir = Path(output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite non-empty pack: {output_dir}")

    references_dir = output_dir / "references"
    score_key_dir = output_dir / "score_key"
    references_dir.mkdir(parents=True, exist_ok=True)
    score_key_dir.mkdir(parents=True, exist_ok=True)

    port_sections: list[str] = []
    manifest_lines = [MANIFEST.rstrip("\n")]
    for case in ERROR_CIRCUITS.values():
        observed = _observed_circuit(case)
        svg_path = references_dir / f"{case.id}_error.svg"
        svg_canvas = draw_reference(observed, ERROR_LAYOUTS[case.id])
        svg_canvas.save(svg_path)
        _write_text(svg_path, svg_path.read_text(encoding="utf-8"))

        png_canvas = draw_reference(observed, ERROR_LAYOUTS[case.id])
        png_canvas.save(references_dir / f"{case.id}_error.png")

        score_path = score_key_dir / f"{case.id}_gt.json"
        _write_text(
            score_path,
            json.dumps(_score_key(case), ensure_ascii=False, indent=2) + "\n",
        )
        manifest_lines.append(
            f"{case.id},llm_scoring,{case.error_type},"
            f"references/{case.id}_error.png,score_key/{case.id}_gt.json"
        )
        port_sections.append(_port_connections(case))

    _write_text(output_dir / "scoring_protocol.md", SCORING_PROTOCOL)
    _write_text(output_dir / "port_connections.md", "\n\n".join(port_sections) + "\n")
    _write_text(output_dir / "manifest_template.csv", "\n".join(manifest_lines) + "\n")
    write_checksums(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate_error_pack(args.output_dir)


if __name__ == "__main__":
    main()
