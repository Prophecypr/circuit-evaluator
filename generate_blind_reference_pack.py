"""Generate a versioned blind circuit-reference collection pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.reference_pack.layouts import LAYOUTS
from src.reference_pack.render import draw_reference
from src.reference_pack.specs import CIRCUITS


COLLECTION_PROTOCOL = """# 盲测参考图采集协议

## 样本规模与纸面绘制

- 由 3 位参与者各自完整绘制 C01-C10，共形成 3 人 × 10 个电路 = 30 张独立纸面原稿。
- 使用空白 A4 纸，一页只画一个电路；使用黑色或深色笔；不用尺，不描图、不透写，按参考图手工临摹。
- 电阻必须画成 ANSI 锯齿形。只抄参考图中可见的数值，不添加 R1、C1、V1、Q1、LED1 等组件设计号。
- 连接点必须按参考图画成清晰的实心点。
- 不得查看模型预测，不得接受模型反馈；数据必须在进入模型前完成质检和必要重画。
- 不允许 LLM 参与识别或提供采集反馈。

## 模型运行前的重画与重拍规则

质检必须在任何模型运行前完成。出现下列任一情形时，整页重画或重新拍摄，并记录决定、原因和时间：缺少元件或数值；导线断开或误连；关键连接点不清；严重裁切、模糊或阴影；与参考图拓扑不一致。是否重画或重拍必须在任何模型运行前决定，不能根据模型输出倒推修改样本。

## 手机双域拍摄

每张纸面原稿用手机拍摄两个域：

- `controlled`：相机位于纸面正上方，光照均匀，纸张平整，尽量避免透视变形。
- `handheld`：自然手持拍摄，允许轻微透视和光照变化，但电路仍须完整、清晰。

最终得到 60 张图像，但统计单位是 30 个 `paper_id` 配对样本；同一纸稿的 controlled/handheld 图像必须使用同一个 `paper_id`。纸面原稿必须保留。建议 `paper_id` 使用 `P01-C01` 一类清晰、唯一的命名，图像文件分别命名为 `P01-C01_controlled.jpg` 和 `P01-C01_handheld.jpg`，并在 manifest 中记录路径。
"""


ANNOTATION_PROTOCOL = """# 盲测手机图标注协议

## 独立标注与仲裁

- 两位标注者必须独立标注，不得互看对方结果；分歧由第三人仲裁。
- 标注对象是手机图中实际可见的手绘内容，不是参考图表达的意图；即使参与者画错，也必须按图中实际内容标注。
- 每张图必须标注：组件 `class` 与 component bbox；port 位置与语义；value bbox 与原始字符串；nets/连接组。IoU、ports、values、nets 均需具备可核验的标注数据。
- 同一纸稿的 controlled 和 handheld 图像须分别标注、分别核对，并通过同一个 `paper_id` 关联。

## GT 冻结与版本化更正

- GT 必须在模型运行前冻结；任何人看过模型输出后不得修改已冻结 GT。
- 冻结后如发现数据错误，必须执行有记录的版本化更正并重新冻结，记录原版本、新版本、修改原因、修改人和日期；不得无痕编辑。
- 仲裁人应在分歧处理完成后形成最终 schema，并签字或以等效方式记录姓名、日期和版本。
"""


MANIFEST_HEADER = (
    "paper_id,circuit_id,participant_id,difficulty,domain,image_path,gt_path\n"
)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file using bounded-memory reads."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: str | Path) -> Path:
    """Write deterministic checksums for every generated file except itself."""

    output_dir = Path(output_dir)
    checksum_path = output_dir / "checksums.sha256"
    relative_paths = sorted(
        (
            path.relative_to(output_dir)
            for path in output_dir.rglob("*")
            if path.is_file() and path != checksum_path
        ),
        key=lambda path: path.as_posix(),
    )
    lines = [
        f"{_sha256_file(output_dir / relative_path)}  {relative_path.as_posix()}"
        for relative_path in relative_paths
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def generate_pack(output_dir: str | Path) -> Path:
    """Generate one complete pack while refusing to overwrite existing content."""

    output_dir = Path(output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite non-empty pack: {output_dir}")

    references_dir = output_dir / "references"
    answer_key_dir = output_dir / "answer_key"
    references_dir.mkdir(parents=True, exist_ok=True)
    answer_key_dir.mkdir(parents=True, exist_ok=True)

    for spec in CIRCUITS.values():
        svg_canvas = draw_reference(spec, LAYOUTS[spec.id])
        svg_canvas.save(references_dir / f"{spec.id}_reference.svg")

        png_canvas = draw_reference(spec, LAYOUTS[spec.id])
        png_canvas.save(references_dir / f"{spec.id}_reference.png")

        answer = {
            "circuit_id": spec.id,
            "title": spec.title,
            "difficulty": spec.difficulty,
            "components": [asdict(component) for component in spec.components],
            "nets": spec.nets,
        }
        answer_path = answer_key_dir / f"{spec.id}_gt.json"
        answer_path.write_text(
            json.dumps(answer, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (output_dir / "collection_protocol.md").write_text(
        COLLECTION_PROTOCOL,
        encoding="utf-8",
    )
    (output_dir / "annotation_protocol.md").write_text(
        ANNOTATION_PROTOCOL,
        encoding="utf-8",
    )
    (output_dir / "manifest_template.csv").write_text(
        MANIFEST_HEADER,
        encoding="utf-8",
    )
    write_checksums(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a versioned blind circuit-reference pack."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    generate_pack(args.output_dir)


if __name__ == "__main__":
    main()
