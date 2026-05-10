"""
Репозиторий для работы с PostgreSQL.

Таблицы:
  raw_data      — сырой CSV
  cleaned_data  — после очистки + лаги/rolling
  prepared_data — агрегированный, без выбросов (для обучения)
  students_input / predictions — инференс-пайплайн
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor, execute_values

from app.core.config import get_settings
from app.db.config import PostgresSettings, settings as _default_settings

# Колонки для inference-таблицы students_input (legacy)
INPUT_COLUMNS = [
    "student_target", "student_class", "course_name", "subject_name",
    "homework_done_mark", "test_part_one", "test_part_two",
    "homework_lag_1", "homework_lag_2", "test1_lag_1", "test2_lag_1",
    "homework_diff", "test1_diff", "test2_diff",
    "homework_rolling_mean_3", "homework_rolling_std_3",
    "test1_rolling_mean_3", "test2_rolling_std_3",
    "homework_max", "homework_min", "test1_max", "test1_min",
    "test2_max", "test2_min",
]


# ── Подключение ──────────────────────────────────────────────
def get_connection(db_settings: PostgresSettings | None = None):
    db_settings = db_settings or _default_settings
    return psycopg2.connect(db_settings.dsn)


def initialize_schema(db_settings: PostgresSettings | None = None) -> None:
    """Выполняет init_tables.sql — создаёт все таблицы."""
    sql_path = get_settings().root_dir / "sql" / "init_tables.sql"
    ddl = sql_path.read_text(encoding="utf-8")
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


# ══════════════════════════════════════════════════════════════
#  Универсальный DataFrame ↔ таблица
# ══════════════════════════════════════════════════════════════
def _dataframe_to_table(
    df: pd.DataFrame,
    table: str,
    *,
    extra_cols: dict[str, Any] | None = None,
    clear_filter: str | None = None,
    db_settings: PostgresSettings | None = None,
) -> int:
    """Записать DataFrame в таблицу."""
    if df.empty:
        return 0

    work = df.copy()
    extra_cols = extra_cols or {}
    for col_name, col_val in extra_cols.items():
        work[col_name] = col_val

    columns = list(work.columns)
    rows = [tuple(row[col] for col in columns) for _, row in work.iterrows()]

    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            if clear_filter:
                cur.execute(f"DELETE FROM {table} WHERE {clear_filter};")
            col_names = ", ".join(columns)
            query = f"INSERT INTO {table} ({col_names}) VALUES %s"
            execute_values(cur, query, rows, page_size=1000)
        conn.commit()

    return len(rows)


def _table_to_dataframe(
    table: str,
    *,
    where: str | None = None,
    params: tuple | None = None,
    db_settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    """Прочитать таблицу (или её часть) в DataFrame."""
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    with get_connection(db_settings) as conn:
        return pd.read_sql(query, conn, params=params)


# ══════════════════════════════════════════════════════════════
#  RAW DATA
# ══════════════════════════════════════════════════════════════
def save_raw_data(
    df: pd.DataFrame,
    experiment: str = "default",
    clear_existing: bool = True,
    db_settings: PostgresSettings | None = None,
) -> int:
    clear_filter = f"experiment = '{experiment}'" if clear_existing else None
    table_cols = [
        "student_id", "student_target", "student_class",
        "course_type", "course_package_type", "subject_name",
        "course_student_active", "course_student_ege_result",
        "homework_done_respectful", "homework_done_mark",
        "test_part", "test_done_mark", "lesson_date",
        "student_city", "course_name",
        "homework_done_mark_probe", "clan_name",
    ]
    existing = [c for c in table_cols if c in df.columns]
    return _dataframe_to_table(
        df[existing],
        "raw_data",
        extra_cols={"experiment": experiment},
        clear_filter=clear_filter,
        db_settings=db_settings,
    )


def load_raw_data(
    experiment: str = "default",
    db_settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    return _table_to_dataframe(
        "raw_data",
        where="experiment = %s",
        params=(experiment,),
        db_settings=db_settings,
    )


# ══════════════════════════════════════════════════════════════
#  CLEANED DATA
# ══════════════════════════════════════════════════════════════
def save_cleaned_data(
    df: pd.DataFrame,
    experiment: str = "default",
    clear_existing: bool = True,
    db_settings: PostgresSettings | None = None,
) -> int:
    clear_filter = f"experiment = '{experiment}'" if clear_existing else None
    drop_cols = {"id", "experiment", "created_at", "loaded_at"}
    cols = [c for c in df.columns if c not in drop_cols]
    return _dataframe_to_table(
        df[cols],
        "cleaned_data",
        extra_cols={"experiment": experiment},
        clear_filter=clear_filter,
        db_settings=db_settings,
    )


def load_cleaned_data(
    experiment: str = "default",
    db_settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    return _table_to_dataframe(
        "cleaned_data",
        where="experiment = %s",
        params=(experiment,),
        db_settings=db_settings,
    )


# ══════════════════════════════════════════════════════════════
#  PREPARED DATA  (для обучения)
# ══════════════════════════════════════════════════════════════
def save_prepared_data(
    df: pd.DataFrame,
    experiment: str = "default",
    clear_existing: bool = True,
    db_settings: PostgresSettings | None = None,
) -> int:
    clear_filter = f"experiment = '{experiment}'" if clear_existing else None
    drop_cols = {"id", "experiment", "created_at", "loaded_at"}
    cols = [c for c in df.columns if c not in drop_cols]
    return _dataframe_to_table(
        df[cols],
        "prepared_data",
        extra_cols={"experiment": experiment},
        clear_filter=clear_filter,
        db_settings=db_settings,
    )


def load_prepared_data(
    experiment: str = "default",
    subject_filter: str | None = None,
    db_settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    where = "experiment = %s"
    params: list = [experiment]
    if subject_filter:
        where += " AND subject_name = %s"
        params.append(subject_filter)
    df = _table_to_dataframe(
        "prepared_data",
        where=where,
        params=tuple(params),
        db_settings=db_settings,
    )
    if not df.empty:
        df["student_target"] = df["student_target"].astype(str)
        if "student_class" in df.columns:
            df["student_class"] = df["student_class"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════
#  STUDENTS INPUT / PREDICTIONS  (inference pipeline)
# ══════════════════════════════════════════════════════════════
def seed_input_data_from_dataframe(
    df: pd.DataFrame,
    limit: int | None = None,
    clear_existing: bool = False,
    db_settings: PostgresSettings | None = None,
) -> int:
    work_df = df.head(limit).copy() if limit else df.copy()
    ege_col = "course_student_ege_result" if "course_student_ege_result" in work_df.columns else None

    rows: list[tuple[Any, ...]] = []
    for _, row in work_df.iterrows():
        values = [row.get(col, -1) for col in INPUT_COLUMNS]
        ege_val = float(row[ege_col]) if ege_col and pd.notna(row.get(ege_col)) else None
        rows.append((*values, ege_val))

    if not rows:
        return 0

    columns = INPUT_COLUMNS + ["source_ege_result"]

    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            if clear_existing:
                cur.execute("TRUNCATE TABLE predictions RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE students_input RESTART IDENTITY CASCADE;")

            col_names = ", ".join(columns)
            query = f"INSERT INTO students_input ({col_names}) VALUES %s"
            execute_values(cur, query, rows, page_size=500)
        conn.commit()

    return len(rows)


def fetch_input_rows_without_predictions(
    limit: int = 100,
    db_settings: PostgresSettings | None = None,
) -> list[dict[str, Any]]:
    col_select = ", ".join([f"s.{col}" for col in INPUT_COLUMNS])
    query = f"""
        SELECT s.id, s.source_ege_result, {col_select}
        FROM students_input s
        LEFT JOIN predictions p ON p.input_data_id = s.id
        WHERE p.input_data_id IS NULL
        ORDER BY s.id
        LIMIT %s
    """
    with get_connection(db_settings) as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def insert_predictions(
    predictions: list[dict[str, Any]],
    db_settings: PostgresSettings | None = None,
) -> int:
    if not predictions:
        return 0

    rows = [
        (
            item["input_data_id"],
            item["predicted_ege_score"],
            item.get("prediction_timestamp", datetime.now(timezone.utc)),
            item.get("model_version"),
        )
        for item in predictions
    ]

    query = """
        INSERT INTO predictions (
            input_data_id,
            predicted_ege_score,
            prediction_timestamp,
            model_version
        ) VALUES %s
        ON CONFLICT (input_data_id) DO UPDATE SET
            predicted_ege_score = EXCLUDED.predicted_ege_score,
            prediction_timestamp = EXCLUDED.prediction_timestamp,
            model_version = EXCLUDED.model_version
    """
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, page_size=500)
        conn.commit()
    return len(rows)


def get_table_counts(db_settings: PostgresSettings | None = None) -> dict[str, int]:
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            counts = {}
            for table in ("raw_data", "cleaned_data", "prepared_data", "students_input", "predictions"):
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                counts[table] = cur.fetchone()[0]
    return counts


def fetch_recent_predictions(
    limit: int = 20,
    db_settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    query = """
        SELECT
            p.id,
            p.input_data_id,
            s.student_target,
            s.student_class,
            s.subject_name,
            s.source_ege_result,
            p.predicted_ege_score,
            p.model_version,
            p.prediction_timestamp
        FROM predictions p
        INNER JOIN students_input s ON s.id = p.input_data_id
        ORDER BY p.prediction_timestamp DESC, p.id DESC
        LIMIT %s
    """
    with get_connection(db_settings) as conn:
        return pd.read_sql(query, conn, params=(limit,))
