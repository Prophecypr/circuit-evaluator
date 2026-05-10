"""电路评价快捷脚本

用法（在项目根目录运行）:

    python eval.py                          # 评价所有 circuits（并行）
    python eval.py circuit_01_good          # 评价单个
    python eval.py circuit_01_good --sim    # 带 Ngspice 仿真
    python eval.py --fast                   # 并行 + 简洁输出
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')


def _eval_single(desc: str, use_sim: bool = False) -> dict:
    if use_sim:
        from src.phase4_netlist.evaluate import evaluate_with_simulation
        return evaluate_with_simulation(desc)
    else:
        from src.phase1_verify.evaluate import evaluate_circuit
        return evaluate_circuit(desc)


def eval_one(fname: str, use_sim: bool = False):
    path = Path(f"data/samples/{fname}.txt") if not fname.endswith(".txt") else Path(f"data/samples/{fname}")
    if not path.exists():
        print(f"文件不存在: {path}")
        return
    desc = path.read_text(encoding="utf-8")

    t0 = time.time()
    r = _eval_single(desc, use_sim=use_sim)
    elapsed = time.time() - t0

    if use_sim:
        print(f"[仿真: {'✅ 真实仿真' if r.get('_simulation_used') else '⚠️ 回退LLM'}] 耗时: {elapsed:.0f}s")
    print(f"电路: {path.name}")
    print(f"评分: {r['overall_score']}/100")
    print(f"总结: {r['summary']}")
    if r.get("fatal_errors"):
        for e in r["fatal_errors"]:
            print(f"  X {e['description'][:100]}")
    if r.get("correctness_issues"):
        for e in r["correctness_issues"]:
            print(f"  ! {e['description'][:100]}")
    if r.get("quality_issues"):
        for e in r["quality_issues"]:
            print(f"  ~ {e['description'][:100]}")
    print(f"  耗时: {elapsed:.1f}s")


def eval_all(parallel: bool = True, simple: bool = False):
    from pathlib import Path
    files = sorted(Path("data/samples").glob("*.txt"))
    print(f"共 {len(files)} 个电路\n")

    if not parallel:
        t0 = time.time()
        for i, f in enumerate(files):
            desc = f.read_text(encoding="utf-8")
            r = _eval_single(desc)
            if simple:
                ok = "✅" if r["overall_score"] >= 70 else ("⚠️" if r["overall_score"] >= 30 else "❌")
                print(f"  [{i+1}/{len(files)}] {ok} {f.name:35s} {r['overall_score']:3d}/100  {r['summary'][:50]}")
            else:
                print(f"[{i+1}/{len(files)}] {f.name}: {r['overall_score']}/100 — {r['summary'][:60]}")
        print(f"\n总耗时: {time.time() - t0:.1f}s")
    else:
        # Parallel evaluation — N circuits simultaneously
        file_data = [(f.name, f.read_text(encoding="utf-8")) for f in files]
        t0 = time.time()

        print(f"并行评估中（{len(files)} 路同时进行）...\n")
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(files), 8)) as ex:
            futures = {ex.submit(_eval_single, desc): name for name, desc in file_data}
            done = 0
            for fut in as_completed(futures):
                name = futures[fut]
                done += 1
                try:
                    r = fut.result()
                    results[name] = r
                    ok = "✅" if r["overall_score"] >= 70 else ("⚠️" if r["overall_score"] >= 30 else "❌")
                    print(f"  [{done}/{len(files)}] {ok} {name:35s} {r['overall_score']:3d}/100  {r['summary'][:50]}")
                except Exception as e:
                    print(f"  [{done}/{len(files)}] ❌ {name}: {e}")

        elapsed = time.time() - t0
        print(f"\n总耗时: {elapsed:.1f}s ({elapsed/len(files):.1f}s/个)")

        # Print details for problem circuits
        print("\n" + "=" * 60)
        print("详情（有问题的电路）")
        print("=" * 60)
        for name, r in sorted(results.items()):
            if r["overall_score"] < 80:
                print(f"\n[{name}] {r['overall_score']}/100")
                if r.get("fatal_errors"):
                    print(f"  致命: {'; '.join(e['description'][:80] for e in r['fatal_errors'])}")
                if r.get("correctness_issues"):
                    print(f"  问题: {'; '.join(e['description'][:80] for e in r['correctness_issues'])}")


if __name__ == "__main__":
    use_sim = "--sim" in sys.argv
    simple = "--fast" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        eval_all(parallel=True, simple=simple)
    else:
        eval_one(args[0], use_sim=use_sim)
