from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RAW_CSV_PATH = DATA_DIR / "2025_07_24_el_school.csv"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_dataset.csv"
ROLLING_CSV_PATH = DATA_DIR / "prepared_rolling_dataset.csv"

MODEL_PATH = ARTIFACTS_DIR / "catboost_model.cbm"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
