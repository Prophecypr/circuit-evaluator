"""Validated local runner for wiring-edge error attribution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class AnalysisCase:
    stem: str
    image_path: Path
    gt_path: Path
    detection_path: Path
    prediction_path: Path
    trace_path: Path


def prepare_output(output_dir: str | Path, resume: bool = False) -> Path:
    """Create a dedicated output directory without overwriting user data."""
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Refusing to overwrite non-directory output: {output}")
        if any(output.iterdir()) and not resume:
            raise FileExistsError(
                f"Refusing to overwrite non-empty attribution output directory: {output}"
            )
    else:
        output.mkdir(parents=True)
    return output


def _stems(paths, suffix_to_remove="") -> set[str]:
    stems = set()
    for path in paths:
        stem = path.stem
        if suffix_to_remove and stem.endswith(suffix_to_remove):
            stem = stem[: -len(suffix_to_remove)]
        stems.add(stem)
    return stems


def _one_image(benchmark_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in benchmark_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem == stem
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{stem}: expected one image, found {len(matches)}")
    return matches[0]


def _require_file(path: Path, label: str, stem: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{stem}: missing {label}: {path}")
    return path


def validate_inputs(
    run_dir: str | Path,
    benchmark_dir: str | Path = "benchmark",
    expected_count: int = 50,
) -> tuple[dict, tuple[AnalysisCase, ...]]:
    """Validate a complete strict-jj cached run and join every file by stem."""
    run_dir = Path(run_dir)
    benchmark_dir = Path(benchmark_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    image_count = metadata.get("image_count")
    if image_count != expected_count:
        raise RuntimeError(f"expected {expected_count} images, found {image_count}")
    if metadata.get("failure_count") != 0:
        raise RuntimeError("input run contains failures")
    strict_config = metadata.get("configs", {}).get("strict_jj")
    if not isinstance(strict_config, dict):
        raise RuntimeError("input run has no strict_jj configuration")
    if strict_config.get("skip_llm") is not True:
        raise RuntimeError("strict_jj must set skip_llm=true")
    if strict_config.get("skip_ocr") is not True:
        raise RuntimeError("strict_jj must set skip_ocr=true")
    if strict_config.get("use_strict_jj") is not True:
        raise RuntimeError("strict_jj must set use_strict_jj=true")
    if metadata.get("final_42_image_test_used") is not False:
        raise RuntimeError("sealed final 42-image test set was used")

    prediction_dir = run_dir / "predictions" / "strict_jj"
    trace_dir = run_dir / "wiring_traces_strict_jj"
    gt_dir = benchmark_dir / "result"
    prediction_stems = _stems(prediction_dir.glob("*.json"))
    trace_stems = _stems(trace_dir.glob("*.json"))
    gt_stems = _stems(gt_dir.glob("*_gt.txt"), "_gt")
    if len(prediction_stems) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} strict_jj predictions, found {len(prediction_stems)}"
        )
    missing_traces = sorted(prediction_stems - trace_stems)
    if missing_traces:
        raise FileNotFoundError(f"missing trace for {missing_traces[0]}")
    extra_traces = sorted(trace_stems - prediction_stems)
    if extra_traces:
        raise RuntimeError(f"trace stems do not match predictions: {extra_traces}")
    if gt_stems != prediction_stems:
        missing_gt = sorted(prediction_stems - gt_stems)
        extra_gt = sorted(gt_stems - prediction_stems)
        raise RuntimeError(f"GT stems do not match predictions: missing={missing_gt}, extra={extra_gt}")

    cases = []
    for stem in sorted(prediction_stems):
        fixed_path = benchmark_dir / "fixed" / f"{stem}.json"
        detection_path = (
            fixed_path
            if fixed_path.is_file()
            else benchmark_dir / "detections" / f"{stem}.json"
        )
        cases.append(
            AnalysisCase(
                stem=stem,
                image_path=_one_image(benchmark_dir, stem),
                gt_path=_require_file(gt_dir / f"{stem}_gt.txt", "GT", stem),
                detection_path=_require_file(detection_path, "detection", stem),
                prediction_path=_require_file(
                    prediction_dir / f"{stem}.json", "prediction", stem
                ),
                trace_path=_require_file(trace_dir / f"{stem}.json", "trace", stem),
            )
        )
    return metadata, tuple(cases)
