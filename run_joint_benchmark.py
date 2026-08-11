"""Run the publication-oriented LLM-free joint vision benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vision.joint_benchmark import run_joint_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    parser.add_argument("--cghd-root", type=Path, required=True)
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        help="Verified prior joint-benchmark directory whose prediction JSONs are imported read-only",
    )
    args = parser.parse_args()
    summary = run_joint_benchmark(
        benchmark_dir=args.benchmark,
        cghd_root=args.cghd_root,
        output_dir=args.output,
        ocr_model_path=args.ocr_model,
        render=not args.no_render,
        resume=args.resume,
        expected_case_count=args.expected_count,
        prediction_cache_dir=args.prediction_cache,
    )
    print("\nJoint benchmark complete")
    print(f"Images: {summary['image_count']}")
    print(f"Failures: {summary['failed_images']}")
    print(f"Render failures: {summary['render_failed_images']}")
    print(f"Summary: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
