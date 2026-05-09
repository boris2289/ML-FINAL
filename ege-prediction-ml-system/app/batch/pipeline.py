from __future__ import annotations

from datetime import datetime, timezone

from app.api.services import PredictorService
from app.core.constants import USED_FEATURES
from app.db.repository import (
    INPUT_COLUMNS,
    fetch_input_rows_without_predictions,
    insert_predictions,
)


def run_batch_prediction(limit: int = 100, model_version: str | None = None) -> dict:
    predictor = PredictorService()
    if not predictor.is_ready:
        raise RuntimeError("Артефакты модели не готовы. Сначала обучите модель.")

    rows = fetch_input_rows_without_predictions(limit=limit)
    if not rows:
        return {
            "rows_read": 0,
            "rows_written": 0,
            "message": "Нет новых строк в students_input.",
        }

    prediction_rows = []
    for row in rows:
        features = {col: row[col] for col in INPUT_COLUMNS}
        result = predictor.predict(features)
        prediction_rows.append({
            "input_data_id": row["id"],
            "predicted_ege_score": result["predicted_ege_score"],
            "prediction_timestamp": datetime.now(timezone.utc),
            "model_version": model_version,
        })

    rows_written = insert_predictions(prediction_rows)
    return {
        "rows_read": len(rows),
        "rows_written": rows_written,
        "message": "Пакетное предсказание выполнено успешно.",
    }
