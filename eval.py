"""电路评价快捷脚本

用法（在项目根目录运行）:

    python eval.py                          # 评价所有 circuits
    python eval.py circuit_01_good          # 评价单个
    python eval.py circuit_01_good --sim    # 带 Ngspice 仿真
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def eval_one(fname: str, use_sim: bool = False):
    path = Path(f"data/samples/{fname}.txt") if not fname.endswith(".txt") else Path(f"data/samples/{fname}")
    if not path.exists():
        print(f"文件不存在: {path}")
        return
    desc = path.read_text(encoding="utf-8")

    if use_sim:
        from src.phase4_netlist.evaluate import evaluate_with_simulation
        r = evaluate_with_simulation(desc)
        print(f"[仿真: {'✅ 真实仿真' if r.get('_simulation_used') else '⚠️ 回退LLM'}]")
    else:
        from src.phase1_verify.evaluate import evaluate_circuit
        r = evaluate_circuit(desc)

    print(f"电路: {path.name}")
    print(f"评分: {r['overall_score']}/100")
    print(f"总结: {r['summary']}")

    if r.get("fatal_errors"):
        print(f"\n致命错误 ({len(r['fatal_errors'])}):")
        for e in r["fatal_errors"]:
            print(f"  X {e['description']}")

    if r.get("correctness_issues"):
        print(f"\n功能问题 ({len(r['correctness_issues'])}):")
        for e in r["correctness_issues"]:
            print(f"  ! {e['description']}")

    if r.get("quality_issues"):
        print(f"\n优化建议 ({len(r['quality_issues'])}):")
        for e in r["quality_issues"]:
            print(f"  ~ {e['description']}")

    if r.get("node_voltages"):
        print(f"\n仿真电压: {r['node_voltages']}")


def eval_all():
    from src.phase1_verify.evaluate import evaluate_all_samples, print_results
    results = evaluate_all_samples()
    print_results(results)


if __name__ == "__main__":
    use_sim = "--sim" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        eval_all()
    else:
        eval_one(args[0], use_sim=use_sim)
