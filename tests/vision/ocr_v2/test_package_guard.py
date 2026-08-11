from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest

from src.vision.ocr_v2.dataset_builder import sha256_file
from src.vision.ocr_v2.package import (
    PackageSafetyError,
    archive_package,
    inspect_package_tree,
    validate_kaggle_notebook,
    verify_package_zip,
)


def test_package_rejects_secret_names(tmp_path):
    (tmp_path / "kaggle.json").write_text("secret", encoding="utf-8")
    with pytest.raises(PackageSafetyError, match="secret filename"):
        inspect_package_tree(tmp_path, forbidden_hashes=set())


def test_package_rejects_api_key_patterns_in_text(tmp_path):
    fake_secret = "OPENAI_API_KEY=" + "sk-" + "proj-" + "a" * 32
    (tmp_path / "notes.txt").write_text(
        fake_secret,
        encoding="utf-8",
    )
    with pytest.raises(PackageSafetyError, match="secret pattern"):
        inspect_package_tree(tmp_path, forbidden_hashes=set())


def test_package_rejects_final_test_hash(tmp_path):
    image = tmp_path / "copied.jpg"
    image.write_bytes(b"final-test-image")
    with pytest.raises(PackageSafetyError, match="final-test hash"):
        inspect_package_tree(tmp_path, forbidden_hashes={sha256_file(image)})


def test_zip_verifier_accepts_inventory_and_rejects_traversal(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "README.md").write_text("safe", encoding="utf-8")
    safe_zip = tmp_path / "safe.zip"
    archive_package(package, safe_zip)
    report = verify_package_zip(safe_zip, forbidden_hashes=set())
    assert report["file_count"] == 1

    bad_zip = tmp_path / "bad.zip"
    with ZipFile(bad_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(PackageSafetyError, match="unsafe ZIP member"):
        verify_package_zip(bad_zip, forbidden_hashes=set())


def test_kaggle_notebook_has_complete_run_all_contract():
    validate_kaggle_notebook(Path("notebooks/crnn_v2_kaggle.ipynb"))


def test_repository_wrapper_can_import_package_when_run_as_a_script():
    completed = subprocess.run(
        [sys.executable, "scripts/build_ocr_kaggle_package.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "build" in completed.stdout
