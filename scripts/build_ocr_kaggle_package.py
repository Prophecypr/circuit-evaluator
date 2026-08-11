"""Repository-root wrapper for the CRNN-v2 Kaggle package builder."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.vision.ocr_v2.package import main


if __name__ == "__main__":
    main()
