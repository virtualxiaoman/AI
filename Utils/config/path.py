from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS_DIR = PROJECT_ROOT / "Datasets"
MODELS_DIR = PROJECT_ROOT / "Models"
TestPictures = DATASETS_DIR / "TestPictures"

if __name__ == "__main__":
    print(PROJECT_ROOT, DATASETS_DIR, MODELS_DIR, TestPictures)
