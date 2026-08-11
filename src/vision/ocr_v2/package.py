"""Build and verify a leakage-safe Kaggle OCR training package."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

from .config import load_config
from .dataset_builder import (
    assert_no_leakage,
    collect_file_hashes,
    extract_cghd_rows,
    extract_digitize_rows,
    read_manifest,
    sha256_bytes,
    sha256_file,
    split_cghd_rows,
    write_manifest,
)
from .metrics import parse_value


SECRET_FILENAMES = {
    ".env",
    "kaggle.json",
    "credentials.json",
    "secrets.json",
    "token.txt",
}
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv", ".ipynb", ".yaml", ".yml"}
SECRET_PATTERNS = (
    re.compile(rb"(?:OPENAI|ANTHROPIC|DEEPSEEK|KAGGLE)_API_KEY\s*[=:]\s*[^\s\"']+", re.I),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
)


class PackageSafetyError(ValueError):
    pass


def _check_member(name: str, data: bytes, forbidden_hashes: set[str]) -> None:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise PackageSafetyError(f"unsafe ZIP member: {name}")
    if pure.name.casefold() in SECRET_FILENAMES:
        raise PackageSafetyError(f"secret filename found: {name}")
    if sha256_bytes(data) in forbidden_hashes:
        raise PackageSafetyError(f"final-test hash found in package member: {name}")
    if pure.suffix.casefold() in TEXT_SUFFIXES:
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise PackageSafetyError(f"secret pattern found in package member: {name}")


def inspect_package_tree(root: str | Path, forbidden_hashes: set[str]) -> dict[str, Any]:
    package_root = Path(root)
    file_count = 0
    byte_count = 0
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        _check_member(path.relative_to(package_root).as_posix(), data, forbidden_hashes)
        file_count += 1
        byte_count += len(data)
    return {"file_count": file_count, "byte_count": byte_count}


def verify_package_zip(
    zip_path: str | Path, forbidden_hashes: set[str]
) -> dict[str, Any]:
    file_count = 0
    byte_count = 0
    with ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            data = archive.read(info)
            _check_member(info.filename, data, forbidden_hashes)
            file_count += 1
            byte_count += len(data)
    return {"file_count": file_count, "byte_count": byte_count}


def validate_kaggle_notebook(path: str | Path) -> None:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Kaggle notebook must use nbformat 4")
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    required = (
        "torch.cuda.is_available",
        "src.vision.ocr_v2.train",
        "src.vision.ocr_v2.evaluate",
        "metrics_summary",
        "ocr_crnn_hand_v2_results.zip",
    )
    missing = [fragment for fragment in required if fragment not in code]
    if missing:
        raise ValueError(f"Kaggle notebook is missing Run All steps: {missing}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(code.encode("utf-8")):
            raise PackageSafetyError("secret pattern found in Kaggle notebook")


def _copy_code_and_guides(repo_root: Path, package_root: Path) -> None:
    source_dir = repo_root / "src" / "vision" / "ocr_v2"
    target_dir = package_root / "src" / "vision" / "ocr_v2"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*.py"), key=lambda path: path.name):
        shutil.copy2(source, target_dir / source.name)
    for relative in (
        Path("configs/ocr_crnn_hand_v2.json"),
        Path("notebooks/crnn_v2_kaggle.ipynb"),
        Path("docs/ocr_crnn_v2_kaggle_guide.md"),
    ):
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def _write_inventory(package_root: Path) -> None:
    entries = []
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name not in {"file_inventory.csv", "SHA256SUMS"}:
            entries.append(
                {
                    "path": path.relative_to(package_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with (package_root / "file_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size", "sha256"))
        writer.writeheader()
        writer.writerows(entries)
    (package_root / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )


def verify_checksum_file(package_root: str | Path) -> dict[str, int]:
    root = Path(package_root)
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / PurePosixPath(relative)
        if not path.is_file():
            raise PackageSafetyError(f"checksum member is missing: {relative}")
        if sha256_file(path) != expected:
            raise PackageSafetyError(f"checksum mismatch: {relative}")
        checked += 1
    return {"checked": checked}


def archive_package(package_root: str | Path, zip_path: str | Path) -> Path:
    """Create a deterministic-member-order ZIP without overwriting an old artifact."""
    root = Path(package_root)
    destination = Path(zip_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing ZIP: {destination}")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return destination


def _dataset_report(
    train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, Any]:
    train_units = Counter(parse_value(row["normalized_label"])[1] or "unitless" for row in train_rows)
    val_units = Counter(parse_value(row["normalized_label"])[1] or "unitless" for row in val_rows)
    return {
        "dataset_version": "ocr-crnn-hand-v2-data-1",
        "seed": config["data"]["seed"],
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_cghd_samples": sum(row["source_kind"] == "cghd" for row in train_rows),
        "train_digitize_samples": sum(row["source_kind"] == "digitize_hcd" for row in train_rows),
        "validation_drafters": config["data"]["validation_drafters"],
        "train_units": dict(sorted(train_units.items())),
        "val_units": dict(sorted(val_units.items())),
        "normalization_version": config["model"]["normalization_version"],
        "chars": config["model"]["chars"],
    }


def build_kaggle_package(
    *,
    config_path: str | Path,
    cghd_root: str | Path,
    digitize_root: str | Path,
    final_test_roots: Iterable[str | Path],
    output_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete data/code ZIP and verify it before returning."""
    config = load_config(config_path)
    repository = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    package_root = output / "package"
    if package_root.exists() and any(package_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing package directory: {package_root}"
        )

    forbidden_hashes = collect_file_hashes(final_test_roots)
    forbidden_fragments = [Path(root).name for root in final_test_roots]
    with tempfile.TemporaryDirectory(prefix="ocr_v2_build_", dir=output) as temporary:
        staging = Path(temporary) / "package"
        data_dir = staging / "data"
        data_dir.mkdir(parents=True)
        cghd_rows = extract_cghd_rows(cghd_root, data_dir)
        cghd_train, val_rows = split_cghd_rows(
            cghd_rows, set(config["data"]["validation_drafters"])
        )
        digitize_rows = extract_digitize_rows(digitize_root, data_dir)
        train_rows = sorted(cghd_train + digitize_rows, key=lambda row: row["sample_id"])
        val_rows = sorted(val_rows, key=lambda row: row["sample_id"])
        if not train_rows or not val_rows:
            raise ValueError(
                f"empty OCR partition: train={len(train_rows)}, val={len(val_rows)}"
            )
        all_rows = train_rows + val_rows
        assert_no_leakage(
            all_rows,
            forbidden_hashes=forbidden_hashes,
            forbidden_path_fragments=forbidden_fragments,
        )
        write_manifest(data_dir / config["data"]["train_manifest"], train_rows)
        write_manifest(data_dir / config["data"]["val_manifest"], val_rows)
        report = _dataset_report(train_rows, val_rows, config)
        (data_dir / "dataset_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (data_dir / "split.json").write_text(
            json.dumps(
                {
                    "split_id": "writers-v1",
                    "validation_drafters": config["data"]["validation_drafters"],
                    "seed": config["data"]["seed"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _copy_code_and_guides(repository, staging)
        validate_kaggle_notebook(staging / "notebooks" / "crnn_v2_kaggle.ipynb")
        (staging / "README.md").write_text(
            "# CRNN-v2 Kaggle package\n\n"
            "Open `notebooks/crnn_v2_kaggle.ipynb` on Kaggle and run all cells. "
            "See `docs/ocr_crnn_v2_kaggle_guide.md` for Chinese instructions.\n",
            encoding="utf-8",
        )
        _write_inventory(staging)
        inspect_package_tree(staging, forbidden_hashes)
        shutil.move(str(staging), str(package_root))

    zip_path = output / "ocr_crnn_hand_v2_kaggle.zip"
    archive_package(package_root, zip_path)
    tree_report = inspect_package_tree(package_root, forbidden_hashes)
    zip_report = verify_package_zip(zip_path, forbidden_hashes)
    checksum_report = verify_checksum_file(package_root)
    return {
        **report,
        "package_root": str(package_root),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "tree_file_count": tree_report["file_count"],
        "zip_file_count": zip_report["file_count"],
        "checksums_verified": checksum_report["checked"],
        "forbidden_hash_count": len(forbidden_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--zip", required=True, type=Path)
    verify.add_argument("--final-test-root", action="append", default=[], type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--cghd-root", required=True, type=Path)
    build.add_argument("--digitize-root", required=True, type=Path)
    build.add_argument("--final-test-root", action="append", default=[], type=Path)
    build.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_package_zip(
            args.zip, collect_file_hashes(args.final_test_root)
        )
    else:
        report = build_kaggle_package(
            config_path=args.config,
            cghd_root=args.cghd_root,
            digitize_root=args.digitize_root,
            final_test_roots=args.final_test_root,
            output_dir=args.output,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
