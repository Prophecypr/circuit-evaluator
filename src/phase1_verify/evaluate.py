from pathlib import Path
from src.llm import ask_json
from src.phase1_verify.prompts import SYSTEM_PROMPT, EVALUATION_PROMPT


def load_circuit(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def evaluate_circuit(description: str, model: str = "claude-sonnet-4-6") -> dict:
    prompt = EVALUATION_PROMPT.format(circuit_description=description)
    return ask_json(prompt, system=SYSTEM_PROMPT, model=model)


def evaluate_all_samples(samples_dir: str = "data/samples") -> list[dict]:
    results = []
    for txt_file in sorted(Path(samples_dir).glob("*.txt")):
        description = txt_file.read_text(encoding="utf-8")
        result = evaluate_circuit(description)
        results.append({
            "file": txt_file.name,
            "description": description,
            "result": result,
        })
    return results


def print_results(results: list[dict]) -> None:
    for r in results:
        print(f"\n{'='*60}")
        print(f"文件: {r['file']}")
        res = r["result"]
        print(f"总分: {res['overall_score']}/100")
        print(f"摘要: {res['summary']}")
        if res["fatal_errors"]:
            print(f"致命错误 ({len(res['fatal_errors'])}):")
            for e in res["fatal_errors"]:
                print(f"  - [{e['type']}] {e['description']}")
        if res["correctness_issues"]:
            print(f"正确性问题 ({len(res['correctness_issues'])}):")
            for e in res["correctness_issues"]:
                print(f"  - [{e['type']}] {e['description']}")
        if res["quality_issues"]:
            print(f"质量问题 ({len(res['quality_issues'])}):")
            for e in res["quality_issues"]:
                print(f"  - [{e['type']}] {e['description']}")


if __name__ == "__main__":
    results = evaluate_all_samples()
    print_results(results)
