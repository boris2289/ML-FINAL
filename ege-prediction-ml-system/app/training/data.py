from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.paths import CLEANED_CSV_PATH, DATA_DIR, RAW_CSV_PATH, ROLLING_CSV_PATH


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Исходный CSV не найден: {path}. "
            "Положите файл 2025_07_24_el_school.csv в папку data/."
        )
    return pd.read_csv(path, low_memory=False)


# ──────────────────────────────────────────────
# Этап 1: очистка и предобработка
# ──────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "student_id", "student_target", "student_class",
        "course_type", "course_package_type", "subject_name",
        "course_student_active", "course_student_ege_result",
        "homework_done_respectful", "homework_done_mark",
        "test_part", "test_done_mark", "lesson_date",
        "student_city", "course_name",
        "homework_done_mark_probe", "clan_name",
    ]
    df = df[keep_cols].copy()

    # Таргет: балл ЕГЭ от 30 до 100
    df = df[df["course_student_ege_result"].between(0, 100)]
    df = df[df["course_student_ege_result"] >= 30]
    df["course_student_ege_result"] = df["course_student_ege_result"].astype(float)

    # Ожидаемый балл студента 0–100
    df = df[(df["student_target"] >= 0.0) & (df["student_target"] <= 100.0)]
    df["student_target"] = df["student_target"].astype(str)

    # Класс: 9, 10, 11
    df["student_class"] = df["student_class"].fillna(0).astype(int)
    df = df[df["student_class"].isin((9, 10, 11))]
    df["student_class"] = df["student_class"].astype(str)

    # Сплит частей теста
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
    df[categorical_columns] = df[categorical_columns].astype(str)
    df[numerical_columns] = df[numerical_columns].astype(float)
    df["lesson_date"] = pd.to_datetime(df["lesson_date"])

    return df


# ──────────────────────────────────────────────
# Этап 2: лаги, дельты, rolling, экстремумы
# ──────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    # Rolling
    for col, prefix in [
        ("homework_done_mark", "homework"),
        ("test_part_one", "test1"),
        ("test_part_two", "test2"),
    ]:
        df[f"{prefix}_rolling_mean_3"] = (
            df.groupby("student_id")[col].transform(lambda x: x.rolling(3).mean())
        )
        df[f"{prefix}_rolling_std_3"] = (
            df.groupby("student_id")[col].transform(lambda x: x.rolling(3).std())
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
    df = df.copy()
    df.fillna(-1, inplace=True)
    df["lesson_date"] = pd.to_datetime(df["lesson_date"])
    df["month"] = df["lesson_date"].dt.to_period("W").astype(str)

    group_cols = ["student_id", "course_type", "course_package_type", "subject_name", "month"]
    agg_df = df.groupby(group_cols).agg({
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
        "homework_rolling_mean_3": "last",
        "homework_rolling_std_3": "last",
        "test1_rolling_mean_3": "last",
        "test2_rolling_std_3": "last",
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
    }).reset_index()

    # Клиппинг значений
    for col in ["test_part_one", "test_part_two", "homework_done_mark"]:
        agg_df.loc[~agg_df[col].between(0, 100), col] = 100

    return agg_df


def remove_outliers(col1: str, data: pd.DataFrame) -> pd.DataFrame:
    cols = ["course_student_ege_result", col1]
    bounds = {}
    for col in cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 2 * iqr, q3 + 2 * iqr)

    condition = (
        (data["course_student_ege_result"] >= bounds["course_student_ege_result"][0])
        & (data["course_student_ege_result"] <= bounds["course_student_ege_result"][1])
        & (data[col1] >= bounds[col1][0])
        & (data[col1] <= bounds[col1][1])
    )
    return data[condition]


# ──────────────────────────────────────────────
# Полный пайплайн подготовки данных
# ──────────────────────────────────────────────
def prepare_full_dataset(
    raw_path: Path | None = None,
    subject_filter: str | None = None,
) -> pd.DataFrame:
    """Загрузить, очистить, создать фичи, агрегировать и убрать выбросы."""
    ensure_data_dir()

    df = load_raw_csv(raw_path)
    df = clean_dataset(df)
    df = add_lag_features(df)
    agg_df = aggregate_by_week(df)

    # Удаление выбросов
    for col in ["homework_done_mark", "test_part_one", "test_part_two"]:
        agg_df = remove_outliers(col, agg_df)

    if subject_filter:
        agg_df = agg_df[agg_df["subject_name"] == subject_filter]

    # Сохраняем промежуточные файлы
    df.to_csv(CLEANED_CSV_PATH, index=False)
    agg_df.to_csv(ROLLING_CSV_PATH, index=False)

    return agg_df


def load_prepared_dataset(path: Path | None = None) -> pd.DataFrame:
    """Загрузить уже подготовленный датасет."""
    path = path or ROLLING_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Подготовленный датасет не найден: {path}. "
            "Запустите сначала этап подготовки данных."
        )
    df = pd.read_csv(path)
    df["student_target"] = df["student_target"].astype(str)
    df["student_class"] = df["student_class"].astype(int).astype(str)
    return df
