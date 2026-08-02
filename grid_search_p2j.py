"""Grid-search PORT_JUNCTION_RADIUS to find the FP/FN sweet spot on 5 representative images."""
import sys, os, json, math, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import evaluate, parse_gt
from src.vision import unified_pipeline as up
from src.vision.unified_pipeline import process_image, DEFAULT_CONFIG

BENCHMARK = Path("benchmark")
DETECTIONS = BENCHMARK / "detections"
FIXED = BENCHMARK / "fixed"
RESULT = BENCHMARK / "result"

# 5 diverse images: simple, medium, complex, dense junctions, digital
SAMPLE = ["C-14_D1_P1", "C32_D1_P1", "C161_D1_P1", "C2_D2_P2", "C10_D1_P3"]
RADII = [120, 150, 180, 200, 220, 250, 300]

print("=" * 80)
print(f"Grid search: P2J threshold across {len(SAMPLE)} images")
print(f"P2J radii: {RADII}")
print("=" * 80)

results = {}
for pjr in RADII:
    up.PORT_JUNCTION_RADIUS = pjr
    up.PORT_JUNCTION_FALLBACK = max(40, pjr - 20)
    
    totals = {"PC": 0.0, "FP": 0.0, "FN": 0.0, "GA": 0.0, "CNA": 0.0, "n": 0}
    
    for stem in SAMPLE:
        img_path = str(BENCHMARK / f"{stem}.jpg")
        if not Path(img_path).exists():
            img_path = str(BENCHMARK / f"{stem}.jpeg")
        if not Path(img_path).exists():
            continue
        
        # Load detection JSON
        det_path = FIXED / f"{stem}.json"
        if not det_path.exists():
            det_path = DETECTIONS / f"{stem}.json"
        det_comps = json.loads(det_path.read_text(encoding="utf-8")).get("components", [])
        
        # Load GT
        gt_path = RESULT / f"{stem}_gt.txt"
        if not gt_path.exists():
            continue
        gt_groups = parse_gt(gt_path)
        
        # Run pipeline
        config = {"use_ccl": False, "use_skeleton": True, "use_nn_filter": True,
                   "use_sobel": True, "use_close_port": True,
                   "use_force_connect": False, "use_los": True, "skip_llm": True}
        try:
            result = process_image(img_path, config=config)
        except Exception as e:
            print(f"  {stem} ERROR: {e}")
            continue
        
        # Evaluate
        metrics = evaluate(result, gt_groups, det_comps)
        if metrics["port_correct_rate"] is None:
            continue
        
        totals["PC"] += metrics["port_correct_rate"]
        totals["FP"] += metrics["fp_rate"]
        totals["FN"] += metrics["fn_rate"]
        totals["GA"] += metrics["group_accuracy"]
        totals["CNA"] += metrics["comp_neighbor_accuracy"]
        totals["n"] += 1
    
    if totals["n"]:
        for k in ["PC", "FP", "FN", "GA", "CNA"]:
            totals[k] /= totals["n"]
        results[pjr] = totals
        print(f"\nP2J={pjr:>4d}  PC={totals['PC']:.3f}  FP={totals['FP']:.3f}  FN={totals['FN']:.3f}  GA={totals['GA']:.3f}  CNA={totals['CNA']:.3f}  (n={totals['n']})")

# Find best (lowest FP*FN product = best balance)
print("\n" + "=" * 80)
print("Best balance (min FP×FN):")
best = None
for pjr, r in results.items():
    score = r["FP"] * r["FN"]
    if best is None or score < best[1]:
        best = (pjr, score, r)
if best:
    pjr, score, r = best
    print(f"  P2J={pjr}  PC={r['PC']:.3f}  FP={r['FP']:.3f}  FN={r['FN']:.3f}  FP×FN={score:.4f}")

print("\nBest PC:")
best_pc = max(results.items(), key=lambda x: x[1]["PC"])
print(f"  P2J={best_pc[0]}  PC={best_pc[1]['PC']:.3f}  FP={best_pc[1]['FP']:.3f}  FN={best_pc[1]['FN']:.3f}")

print("\nDone.")
