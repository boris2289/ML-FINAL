from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.core.constants import CAT_FEATURES, TARGET_COL, USED_FEATURES
from app.core.paths import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH


@dataclass
class TrainingResult:
    mae: float
    r2: float
    model_path: str
    metrics_path: str


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    iterations: int = 100,
    depth: int = 4,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 7,
    random_strength: float = 12,
    random_state: int = 54,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        cat_features=CAT_FEATURES,
        loss_function="MAE",
        random_seed=random_state,
        verbose=100,
    )
    model.fit(
        x_train, y_train,
        eval_set=(x_test, y_test),
        use_best_model=True,
        verbose=100,
    )
    return model


def evaluate_model(
    model: CatBoostRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    y_pred = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return mae, r2


def save_artifacts(
    model: CatBoostRegressor,
    mae: float,
    r2: float,
) -> TrainingResult:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(
        json.dumps({"mae": mae, "r2": r2}, indent=2),
        encoding="utf-8",
    )

    return TrainingResult(
        mae=mae,
        r2=r2,
        model_path=str(MODEL_PATH),
        metrics_path=str(METRICS_PATH),
    )


def run_training_pipeline(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 54,
    **model_kwargs,
) -> tuple[CatBoostRegressor, TrainingResult]:
    X = df[USED_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    model = train_catboost(X_train, y_train, X_test, y_test, **model_kwargs)
    mae, r2 = evaluate_model(model, X_test, y_test)
    result = save_artifacts(model, mae, r2)

    print(f"MAE (средняя абсолютная ошибка): {mae:.2f}")
    print(f"R2 (доля объяснённой дисперсии): {r2:.2f}")

    return model, result
