from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor, execute_values

from app.db.config import PostgresSettings, settings

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


def get_connection(db_settings: PostgresSettings | None = None):
    db_settings = db_settings or settings
    return psycopg2.connect(db_settings.dsn)


def initialize_schema(db_settings: PostgresSettings | None = None) -> None:
    ddl = build_schema_sql()
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def build_schema_sql() -> str:
    return """
CREATE TABLE IF NOT EXISTS students_input (
    id BIGSERIAL PRIMARY KEY,
    student_target TEXT NOT NULL,
    student_class TEXT NOT NULL,
    course_name TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    homework_done_mark DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_one DOUBLE PRECISION NOT NULL DEFAULT 0,
    test_part_two DOUBLE PRECISION NOT NULL DEFAULT 0,
    homework_lag_1 DOUBLE PRECISION DEFAULT -1,
    homework_lag_2 DOUBLE PRECISION DEFAULT -1,
    test1_lag_1 DOUBLE PRECISION DEFAULT -1,
    test2_lag_1 DOUBLE PRECISION DEFAULT -1,
    homework_diff DOUBLE PRECISION DEFAULT 0,
    test1_diff DOUBLE PRECISION DEFAULT 0,
    test2_diff DOUBLE PRECISION DEFAULT 0,
    homework_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    homework_rolling_std_3 DOUBLE PRECISION DEFAULT -1,
    test1_rolling_mean_3 DOUBLE PRECISION DEFAULT -1,
    test2_rolling_std_3 DOUBLE PRECISION DEFAULT -1,
    homework_max DOUBLE PRECISION DEFAULT 0,
    homework_min DOUBLE PRECISION DEFAULT 0,
    test1_max DOUBLE PRECISION DEFAULT 0,
    test1_min DOUBLE PRECISION DEFAULT 0,
    test2_max DOUBLE PRECISION DEFAULT 0,
    test2_min DOUBLE PRECISION DEFAULT 0,
    source_ege_result DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_data_id BIGINT NOT NULL REFERENCES students_input(id) ON DELETE CASCADE,
    predicted_ege_score DOUBLE PRECISION NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version TEXT,
    UNIQUE (input_data_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_input_id ON predictions(input_data_id);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(prediction_timestamp);
"""


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

            placeholders = ", ".join(["%s"] * len(columns))
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
            cur.execute("SELECT COUNT(*) FROM students_input;")
            input_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM predictions;")
            prediction_count = cur.fetchone()[0]
    return {"students_input": input_count, "predictions": prediction_count}


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
