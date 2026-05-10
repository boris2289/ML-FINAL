"""
Пайплайн подготовки данных.

Поток: CSV-файл → raw_data (БД) → clean → cleaned_data (БД)
       → aggregate → prepared_data (БД) → обучение.

Все параметры читаются из центрального Settings (.env).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.db.repository import (
    load_prepared_data,
    load_raw_data,
    save_cleaned_data,
    save_prepared_data,
    save_raw_data,
)


# ──────────────────────────────────────────────
# Загрузка сырого CSV (первый шаг)
# ──────────────────────────────────────────────
def load_raw_csv(path: Path | None = None) -> pd.DataFrame:
    cfg = get_settings()
    path = path or cfg.raw_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Исходный CSV не найден: {path}. "
            f"Положите файл {cfg.raw_csv_filename} в папку {cfg.data_dir}/."
        )
    return pd.read_csv(path, low_memory=False)


# ──────────────────────────────────────────────
# Этап 1: очистка и предобработка
# ──────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_settings()

    keep_cols = [
        "student_id", "student_target", "student_class",
        "course_type", "course_package_type", "subject_name",
        "course_student_active", "course_student_ege_result",
        "homework_done_respectful", "homework_done_mark",
        "test_part", "test_done_mark", "lesson_date",
        "student_city", "course_name",
        "homework_done_mark_probe", "clan_name",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Числовые поля, которые могут приехать из CSV как строки
    for col in [
        "course_student_ege_result",
        "student_target",
        "student_class",
        "course_student_active",
        "homework_done_respectful",
        "homework_done_mark",
        "homework_done_mark_probe",
        "test_part",
        "test_done_mark",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Таргет: балл ЕГЭ в допустимом диапазоне
    df = df[df["course_student_ege_result"].notna()]
    df = df[df["course_student_ege_result"].between(0, cfg.ege_max_score)]
    df = df[df["course_student_ege_result"] >= cfg.ege_min_score]
    df["course_student_ege_result"] = df["course_student_ege_result"].astype(float)

    # Ожидаемый балл студента 0–100
    df = df[df["student_target"].notna()]
    df = df[(df["student_target"] >= 0.0) & (df["student_target"] <= 100.0)]
    df["student_target"] = df["student_target"].astype(str)

    # Класс: из конфига
    df["student_class"] = df["student_class"].fillna(0).astype(int)
    df = df[df["student_class"].isin(cfg.allowed_classes_list)]
    df["student_class"] = df["student_class"].astype(str)

    # Сплит частей теста
    if "test_part" in df.columns and "test_done_mark" in df.columns:
        df["test_part_one"] = df.apply(
            lambda r: r["test_done_mark"] if r["test_part"] == 1.0 else None, axis=1,
        )
        df["test_part_two"] = df.apply(
            lambda r: r["test_done_mark"] if r["test_part"] == 2.0 else None, axis=1,
        )
        df = df.drop(["test_part", "test_done_mark"], axis=1)

    # Типы
    categorical_columns = [
        "student_target", "student_class", "course_type",
        "course_package_type", "subject_name", "student_city", "course_name",
    ]
    numerical_columns = [
        "student_id", "course_student_active", "course_student_ege_result",
        "homework_done_respectful", "homework_done_mark",
        "test_part_one", "test_part_two",
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lesson_date" in df.columns:
        df["lesson_date"] = pd.to_datetime(df["lesson_date"], errors="coerce")

    return df


# ──────────────────────────────────────────────
# Этап 2: лаги, дельты, rolling, экстремумы
# ──────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_settings()
    df = df.copy()
    window = cfg.rolling_window

    # Лаги
    df["homework_lag_1"] = df.groupby("student_id")["homework_done_mark"].shift(1)
    df["homework_lag_2"] = df.groupby("student_id")["homework_done_mark"].shift(2)
    df["test1_lag_1"] = df.groupby("student_id")["test_part_one"].shift(1)
    df["test1_lag_2"] = df.groupby("student_id")["test_part_one"].shift(2)
    df["test2_lag_1"] = df.groupby("student_id")["test_part_two"].shift(1)
    df["test2_lag_2"] = df.groupby("student_id")["test_part_two"].shift(2)

    # Разности
    df["homework_diff"] = df["homework_done_mark"] - df["homework_lag_1"]
    df["test1_diff"] = df["test_part_one"] - df["test1_lag_1"]
    df["test2_diff"] = df["test_part_two"] - df["test2_lag_1"]

    # Rolling (окно из конфига)
    for col, prefix in [
        ("homework_done_mark", "homework"),
        ("test_part_one", "test1"),
        ("test_part_two", "test2"),
    ]:
        df[f"{prefix}_rolling_mean_{window}"] = (
            df.groupby("student_id")[col].transform(lambda x: x.rolling(window).mean())
        )
        df[f"{prefix}_rolling_std_{window}"] = (
            df.groupby("student_id")[col].transform(lambda x: x.rolling(window).std())
        )

    # Экстремумы
    for col, prefix in [
        ("homework_done_mark", "homework"),
        ("test_part_one", "test1"),
        ("test_part_two", "test2"),
    ]:
        df[f"{prefix}_max"] = df.groupby("student_id")[col].transform("max")
        df[f"{prefix}_min"] = df.groupby("student_id")[col].transform("min")

    return df


def aggregate_by_week(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_settings()
    window = cfg.rolling_window
    df = df.copy()
    df.fillna(-1, inplace=True)
    df["lesson_date"] = pd.to_datetime(df["lesson_date"])
    df["month"] = df["lesson_date"].dt.to_period("W").astype(str)

    group_cols = ["student_id", "course_type", "course_package_type", "subject_name", "month"]
    agg_dict = {
        "homework_done_mark": "mean",
        "test_part_one": "mean",
        "test_part_two": "mean",
        "homework_lag_1": "last",
        "homework_lag_2": "last",
        "test1_lag_1": "last",
        "test2_lag_1": "last",
        "homework_diff": "last",
        "test1_diff": "last",
        "test2_diff": "last",
        f"homework_rolling_mean_{window}": "last",
        f"homework_rolling_std_{window}": "last",
        f"test1_rolling_mean_{window}": "last",
        f"test2_rolling_std_{window}": "last",
        "homework_max": "last",
        "homework_min": "last",
        "test1_max": "last",
        "test1_min": "last",
        "test2_max": "last",
        "test2_min": "last",
        "student_class": "first",
        "student_target": "first",
        "course_student_ege_result": "first",
        "course_name": "first",
    }
    # Оставляем только колонки, присутствующие в DataFrame
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Клиппинг значений
    for col in ["test_part_one", "test_part_two", "homework_done_mark"]:
        if col in agg_df.columns:
            agg_df.loc[~agg_df[col].between(0, 100), col] = 100

    return agg_df


def remove_outliers(col1: str, data: pd.DataFrame) -> pd.DataFrame:
    target = get_settings().target_col
    cols = [target, col1]
    bounds = {}
    for col in cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 2 * iqr, q3 + 2 * iqr)

    condition = (
        (data[target] >= bounds[target][0])
        & (data[target] <= bounds[target][1])
        & (data[col1] >= bounds[col1][0])
        & (data[col1] <= bounds[col1][1])
    )
    return data[condition]


# ──────────────────────────────────────────────
# Полный пайплайн подготовки данных (через БД)
# ──────────────────────────────────────────────
def prepare_full_dataset(
    raw_path: Path | None = None,
    subject_filter: str | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    """
    CSV → raw_data (БД) → clean → cleaned_data (БД)
        → aggregate → prepared_data (БД).

    Возвращает подготовленный DataFrame, готовый для обучения.
    """
    cfg = get_settings()
    experiment = experiment or cfg.experiment_name
    subject_filter = subject_filter or cfg.subject_filter_value

    # 1. Читаем CSV
    raw_df = load_raw_csv(raw_path)

    # 2. Сохраняем сырые данные в БД
    print(f"[data] Сохраняю сырые данные в raw_data (experiment={experiment})...")
    save_raw_data(raw_df, experiment=experiment)

    # 3. Очистка + фичи
    print("[data] Очистка данных...")
    cleaned_df = clean_dataset(raw_df)
    cleaned_df = add_lag_features(cleaned_df)

    # 4. Сохраняем очищенные данные в БД
    print(f"[data] Сохраняю очищенные данные в cleaned_data (experiment={experiment})...")
    save_cleaned_data(cleaned_df, experiment=experiment)

    # 5. Агрегация + выбросы
    print("[data] Агрегация и удаление выбросов...")
    agg_df = aggregate_by_week(cleaned_df)
    for col in ["homework_done_mark", "test_part_one", "test_part_two"]:
        if col in agg_df.columns:
            agg_df = remove_outliers(col, agg_df)

    if subject_filter:
        agg_df = agg_df[agg_df["subject_name"] == subject_filter]

    # 6. Сохраняем подготовленные данные в БД
    print(f"[data] Сохраняю подготовленные данные в prepared_data (experiment={experiment})...")
    save_prepared_data(agg_df, experiment=experiment)

    print(f"[data] Готово: {len(agg_df)} строк в prepared_data.")
    return agg_df


def load_prepared_dataset(
    experiment: str | None = None,
    subject_filter: str | None = None,
) -> pd.DataFrame:
    """Загрузить подготовленный датасет из БД."""
    cfg = get_settings()
    experiment = experiment or cfg.experiment_name
    subject_filter = subject_filter or cfg.subject_filter_value

    df = load_prepared_data(
        experiment=experiment,
        subject_filter=subject_filter,
    )
    if df.empty:
        raise RuntimeError(
            f"Подготовленные данные не найдены в БД для experiment='{experiment}'. "
            "Запустите сначала этап подготовки данных (prepare_full_dataset)."
        )
    return df
