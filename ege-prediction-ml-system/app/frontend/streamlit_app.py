from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from app.batch.pipeline import run_batch_prediction
from app.db.config import settings
from app.db.repository import (
    fetch_recent_predictions,
    get_table_counts,
    initialize_schema,
    seed_input_data_from_dataframe,
)
from app.training.data import load_prepared_dataset

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ROLLING_CSV_PATH = Path(os.getenv("ROLLING_CSV_PATH", "data/prepared_rolling_dataset.csv"))
DEFAULT_SEED_LIMIT = int(os.getenv("DEFAULT_SEED_LIMIT", "100"))


@st.cache_data
def load_rolling_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["student_target"] = df["student_target"].astype(str)
    df["student_class"] = df["student_class"].astype(int).astype(str)
    return df


def call_prediction_api(features: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=features,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="EGE Prediction System", layout="wide")
st.title("Прогнозирование баллов ЕГЭ")
st.write("Streamlit UI для API-инференса и batch-предсказаний через PostgreSQL.")

with st.sidebar:
    st.subheader("API")
    st.code(API_BASE_URL)
    if st.button("Проверить API"):
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=10).json()
            st.success(str(health))
        except Exception as exc:
            st.error(f"API недоступен: {exc}")

    st.subheader("PostgreSQL")
    st.code(settings.jdbc_url)
    st.caption(f"user={settings.user}")

form_tab, csv_tab, json_tab, db_tab = st.tabs(
    ["Ввод данных", "Из CSV", "JSON", "PostgreSQL batch pipeline"]
)

# ─── Вкладка: Ручной ввод ─────────────────────────
with form_tab:
    st.write("Введите данные студента для получения прогноза балла ЕГЭ.")

    col1, col2 = st.columns(2)
    with col1:
        student_target = st.text_input("Ожидаемый балл (student_target)", value="90.0")
        student_class = st.selectbox("Класс", ["9", "10", "11"], index=2)
        subject_name = st.selectbox("Предмет", [
            "Обществознание", "История", "Литература", "Русский",
            "Английский язык", "Математика", "Физика", "Биология",
            "Химия", "Информатика",
        ])
        course_name = st.text_input("Название курса", value="Годовой 2к25 стандарт")

    with col2:
        homework_done_mark = st.number_input("Средний балл ДЗ", 0.0, 100.0, 70.0)
        test_part_one = st.number_input("Средний балл теста (ч.1)", 0.0, 100.0, 60.0)
        test_part_two = st.number_input("Средний балл теста (ч.2)", 0.0, 100.0, 40.0)

    if st.button("Предсказать балл ЕГЭ"):
        try:
            payload = {
                "student_target": student_target,
                "student_class": student_class,
                "course_name": course_name,
                "subject_name": subject_name,
                "homework_done_mark": homework_done_mark,
                "test_part_one": test_part_one,
                "test_part_two": test_part_two,
            }
            result = call_prediction_api(payload)
            st.success(f"Прогноз балла ЕГЭ: {result['predicted_ege_score']}")
            st.json(result)
        except Exception as exc:
            st.error(f"Ошибка предсказания: {exc}")

# ─── Вкладка: Из CSV ──────────────────────────────
with csv_tab:
    df = load_rolling_csv(str(ROLLING_CSV_PATH))
    if df is None:
        st.info(f"Файл {ROLLING_CSV_PATH.name} не найден. Запустите этап подготовки данных.")
    else:
        if "sample_row_index" not in st.session_state:
            st.session_state.sample_row_index = 0

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Случайная строка"):
                st.session_state.sample_row_index = int(np.random.randint(0, len(df)))
        with col2:
            selected_index = st.number_input(
                "Индекс строки",
                min_value=0,
                max_value=max(len(df) - 1, 0),
                value=int(st.session_state.sample_row_index),
                step=1,
            )
        st.session_state.sample_row_index = int(selected_index)

        row = df.iloc[int(st.session_state.sample_row_index)]
        true_ege = row.get("course_student_ege_result", "N/A")
        st.write(f"Истинный балл ЕГЭ: **{true_ege}**")
        st.dataframe(row.to_frame().T, use_container_width=True)

        if st.button("Предсказать выбранную строку"):
            try:
                features = row.to_dict()
                result = call_prediction_api(features)
                st.success(f"Прогноз: {result['predicted_ege_score']}")
                st.write(f"Истинный балл: {true_ege}")
                st.json(result)
            except Exception as exc:
                st.error(f"Ошибка предсказания: {exc}")

# ─── Вкладка: JSON ────────────────────────────────
with json_tab:
    st.write("Вставьте JSON с данными студента.")
    sample_payload = {
        "student_target": "90.0",
        "student_class": "11",
        "course_name": "Годовой 2к25 стандарт",
        "subject_name": "Обществознание",
        "homework_done_mark": 70.0,
        "test_part_one": 60.0,
        "test_part_two": 40.0,
    }
    json_text = st.text_area(
        "JSON payload",
        value=json.dumps(sample_payload, ensure_ascii=False, indent=2),
        height=320,
    )

    if st.button("Предсказать из JSON"):
        try:
            payload = json.loads(json_text)
            result = call_prediction_api(payload)
            st.success(f"Прогноз балла ЕГЭ: {result['predicted_ege_score']}")
            st.json(result)
        except json.JSONDecodeError:
            st.error("Некорректный формат JSON.")
        except Exception as exc:
            st.error(f"Ошибка предсказания: {exc}")

# ─── Вкладка: PostgreSQL batch pipeline ───────────
with db_tab:
    st.subheader("Batch prediction pipeline (PostgreSQL)")
    st.caption(
        "Пайплайн: чтение students_input из БД, прогон модели, запись predictions обратно."
    )

    left, right = st.columns([1, 1])
    with left:
        seed_limit = st.number_input(
            "Строк для вставки", min_value=1, max_value=10000, value=DEFAULT_SEED_LIMIT,
        )
        clear_existing = st.checkbox("Очистить перед вставкой", value=False)

        if st.button("1) Инициализировать таблицы"):
            try:
                initialize_schema()
                st.success("Таблицы students_input и predictions готовы.")
            except Exception as exc:
                st.error(f"Ошибка инициализации: {exc}")

        if st.button("2) Вставить данные из CSV"):
            try:
                initialize_schema()
                rolling_df = load_prepared_dataset()
                inserted = seed_input_data_from_dataframe(
                    rolling_df, limit=int(seed_limit), clear_existing=clear_existing,
                )
                st.success(f"Вставлено {inserted} строк в students_input.")
            except Exception as exc:
                st.error(f"Ошибка вставки: {exc}")

        batch_limit = st.number_input(
            "Размер батча", min_value=1, max_value=10000, value=100,
        )
        model_version = st.text_input("Версия модели", value="local-catboost-v1")

        if st.button("3) Запустить batch prediction"):
            try:
                result = run_batch_prediction(
                    limit=int(batch_limit), model_version=model_version,
                )
                st.success("Batch prediction завершён.")
                st.json(result)
            except Exception as exc:
                st.error(f"Ошибка batch prediction: {exc}")

    with right:
        if st.button("Обновить статистику"):
            st.cache_data.clear()
        try:
            counts = get_table_counts()
            st.metric("Строк в students_input", counts["students_input"])
            st.metric("Строк в predictions", counts["predictions"])
            recent = fetch_recent_predictions(limit=20)
            if not recent.empty:
                st.dataframe(recent, use_container_width=True)
            else:
                st.info("Предсказаний пока нет.")
        except Exception as exc:
            st.warning(f"Не удалось прочитать статистику: {exc}")

    st.markdown("""
**Команды**

Одиночный batch:
```bash
python -m app.batch.run_batch_prediction --limit 100 --model-version local-catboost-v1
```

Непрерывный планировщик (каждые 5 минут):
```bash
BATCH_INTERVAL_SECONDS=300 python -m app.batch.scheduler
```
""")
