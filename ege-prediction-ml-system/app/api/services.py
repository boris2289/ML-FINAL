from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor

from app.core.constants import USED_FEATURES
from app.core.paths import MODEL_PATH


class ModelNotReadyError(RuntimeError):
    pass


class PredictorService:
    def __init__(self) -> None:
        self.model: CatBoostRegressor | None = None
        self.reload()

    def reload(self) -> None:
        if not MODEL_PATH.exists():
            self.model = None
            return
        self.model = CatBoostRegressor()
        self.model.load_model(str(MODEL_PATH))

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, features: dict) -> dict:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Артефакты модели не найдены. Сначала обучите модель, "
                "чтобы создать artifacts/catboost_model.cbm."
            )

        row = pd.DataFrame([features])[USED_FEATURES]
        prediction = float(self.model.predict(row)[0])
        return {"predicted_ege_score": round(prediction, 2)}
