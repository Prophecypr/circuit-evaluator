from src.phase2_pipeline.schema import Circuit, EvaluationReport
from src.phase2_pipeline.converter import convert_to_circuit
from src.phase2_pipeline.safety import evaluate_safety
from src.phase2_pipeline.correctness import evaluate_correctness
from src.phase2_pipeline.quality import evaluate_quality


def evaluate(description: str, model: str = "claude-sonnet-4-6") -> EvaluationReport:
    # Step 1: Convert to structured circuit
    circuit = convert_to_circuit(description, model=model)

    # Step 2: Run checks
    safety_issues = evaluate_safety(circuit)
    correctness_issues = evaluate_correctness(circuit, model=model)
    quality_issues = evaluate_quality(circuit, model=model)

    # Step 3: Compute score
    score = _compute_score(safety_issues, correctness_issues, quality_issues)

    # Step 4: Generate summary
    summary = _generate_summary(safety_issues, correctness_issues, quality_issues)

    return EvaluationReport(
        overall_score=score,
        safety_issues=safety_issues,
        correctness_issues=correctness_issues,
        quality_issues=quality_issues,
        summary=summary,
    )


def _compute_score(safety, correctness, quality) -> int:
    score = 100
    score -= len([i for i in safety if i.severity == "critical"]) * 30
    score -= len([i for i in correctness if i.severity == "major"]) * 15
    score -= len([i for i in correctness if i.severity == "minor"]) * 5
    score -= len([i for i in quality]) * 3
    return max(0, min(100, score))


def _generate_summary(safety, correctness, quality) -> str:
    parts = []
    critical = len([i for i in safety if i.severity == "critical"])
    major = len([i for i in correctness if i.severity == "major"])
    if critical > 0:
        parts.append(f"发现{critical}处致命错误")
    if major > 0:
        parts.append(f"发现{major}处功能问题")
    if not parts:
        minor_count = len(correctness) + len(quality)
        if minor_count > 0:
            parts.append(f"电路基本正确，有{minor_count}处优化建议")
        else:
            parts.append("电路设计正确，质量良好")
    return "；".join(parts)


def print_report(report: EvaluationReport) -> None:
    print(f"\n{'='*60}")
    print(f"电路评价报告")
    print(f"{'='*60}")
    print(f"总体评分: {report.overall_score}/100")
    print(f"摘要: {report.summary}")

    if report.safety_issues:
        print(f"\n【安全/致命错误】({len(report.safety_issues)}条)")
        for i in report.safety_issues:
            print(f"  🔴 [{i.type}] {i.description}")

    if report.correctness_issues:
        print(f"\n【正确性问题】({len(report.correctness_issues)}条)")
        for i in report.correctness_issues:
            icon = "🟠" if i.severity == "major" else "🟡"
            print(f"  {icon} [{i.type}] {i.description}")
            print(f"     建议: {i.suggestion}")

    if report.quality_issues:
        print(f"\n【质量优化】({len(report.quality_issues)}条)")
        for i in report.quality_issues:
            print(f"  🔵 [{i.type}] {i.description}")
            print(f"     建议: {i.suggestion}")

    if not any([report.safety_issues, report.correctness_issues, report.quality_issues]):
        print("\n✅ 未发现任何问题，电路设计优秀。")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        desc = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        desc = Path("data/samples/circuit_04_no_r.txt").read_text(encoding="utf-8")

    report = evaluate(desc)
    print_report(report)
