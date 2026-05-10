"""
Пути проекта — делегируют всё в центральный Settings.

Модуль сохранён для обратной совместимости.
"""
from app.core.config import get_settings

_cfg = get_settings()

ROOT_DIR = _cfg.root_dir
DATA_DIR = _cfg.data_path
ARTIFACTS_DIR = _cfg.artifacts_path

RAW_CSV_PATH = _cfg.raw_csv_path
MODEL_PATH = _cfg.model_path
METRICS_PATH = _cfg.metrics_path
