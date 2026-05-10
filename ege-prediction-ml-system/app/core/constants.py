"""
Константы приложения — делегируют всё в центральный Settings.

Модуль сохранён для обратной совместимости: старый код,
импортирующий CAT_FEATURES / USED_FEATURES / TARGET_COL,
продолжит работать без изменений.
"""
from app.core.config import get_settings

_cfg = get_settings()

SUBJECT_NAMES = [
    "Обществознание",
    "История",
    "Литература",
    "Русский",
    "Английский язык",
    "Математика",
    "Физика",
    "Биология",
    "Химия",
    "Информатика",
    "Русский ОГЭ",
    "Математика ОГЭ",
    "Обществознание ОГЭ",
    "Биология ОГЭ",
]

CAT_FEATURES: list[str] = _cfg.cat_features_list
USED_FEATURES: list[str] = _cfg.used_features_list
TARGET_COL: str = _cfg.target_col
