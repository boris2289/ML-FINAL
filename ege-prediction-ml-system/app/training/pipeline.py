"""
Тренировочный пайплайн CatBoost.

Все гиперпараметры берутся из Settings (.env) по умолчанию,
но могут быть перекрыты аргументами функций.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.core.config import get_settings


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
    iterations: int | None = None,
    depth: int | None = None,
    learning_rate: float | None = None,
    l2_leaf_reg: float | None = None,
    random_strength: float | None = None,
    random_state: int | None = None,
) -> CatBoostRegressor:
    cfg = get_settings()

    model = CatBoostRegressor(
        iterations=iterations or cfg.iterations,
        depth=depth or cfg.depth,
        learning_rate=learning_rate or cfg.learning_rate,
        l2_leaf_reg=l2_leaf_reg or cfg.l2_leaf_reg,
        random_strength=random_strength or cfg.random_strength,
        cat_features=cfg.cat_features_list,
        loss_function=cfg.loss_function,
        random_seed=random_state or cfg.random_state,
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
    cfg = get_settings()
    cfg.artifacts_path.mkdir(parents=True, exist_ok=True)

    model.save_model(str(cfg.model_path))
    cfg.metrics_path.write_text(
        json.dumps({"mae": mae, "r2": r2}, indent=2),
        encoding="utf-8",
    )

    return TrainingResult(
        mae=mae,
        r2=r2,
        model_path=str(cfg.model_path),
        metrics_path=str(cfg.metrics_path),
    )


def run_training_pipeline(
    df: pd.DataFrame,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
    **model_kwargs,
) -> tuple[CatBoostRegressor, TrainingResult]:
    cfg = get_settings()

    X = df[cfg.used_features_list]
    y = df[cfg.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size or cfg.test_size,
        random_state=random_state or cfg.random_state,
    )

    model = train_catboost(X_train, y_train, X_test, y_test, **model_kwargs)
    mae, r2 = evaluate_model(model, X_test, y_test)
    result = save_artifacts(model, mae, r2)

    print(f"MAE (средняя абсолютная ошибка): {mae:.2f}")
    print(f"R2 (доля объяснённой дисперсии): {r2:.2f}")

    return model, result
