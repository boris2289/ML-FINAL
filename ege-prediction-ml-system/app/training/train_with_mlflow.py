"""
Точка входа для обучения с MLflow.

Все параметры берутся из .env через Settings.
CLI-аргументы (--iterations и т.д.) перекрывают .env,
но если их не передать — используются значения из .env.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.catboost
from sklearn.model_selection import train_test_split

from app.core.config import get_settings
from app.training.data import load_prepared_dataset, prepare_full_dataset
from app.training.pipeline import evaluate_model, save_artifacts, train_catboost


def run_training(
    *,
    iterations: int | None = None,
    depth: int | None = None,
    learning_rate: float | None = None,
    random_state: int | None = None,
    subject_filter: str | None = None,
    experiment_name: str | None = None,
    model_name: str | None = None,
    register_model: bool | None = None,
    tracking_uri: str | None = None,
    skip_prepare: bool = False,
) -> dict:
    cfg = get_settings()

    # Значения: CLI-аргумент → .env → дефолт
    iterations = iterations or cfg.iterations
    depth = depth or cfg.depth
    learning_rate = learning_rate or cfg.learning_rate
    random_state = random_state or cfg.random_state
    experiment_name = experiment_name or cfg.experiment_name
    model_name = model_name or cfg.model_name
    register_model = register_model if register_model is not None else cfg.register_model
    tracking_uri = tracking_uri or cfg.mlflow_tracking_uri
    subject_filter = subject_filter or cfg.subject_filter_value

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Подготовка данных
    if skip_prepare:
        df = load_prepared_dataset(experiment=experiment_name)
    else:
        df = prepare_full_dataset(
            subject_filter=subject_filter,
            experiment=experiment_name,
        )

    X = df[cfg.used_features_list]
    y = df[cfg.target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=random_state,
    )

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params({
            "model_type": "CatBoostRegressor",
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": cfg.l2_leaf_reg,
            "random_strength": cfg.random_strength,
            "loss_function": cfg.loss_function,
            "random_state": random_state,
            "subject_filter": subject_filter or "all",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "test_size": cfg.test_size,
            "rolling_window": cfg.rolling_window,
            "experiment": experiment_name,
        })

        model = train_catboost(
            X_train, y_train, X_test, y_test,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=cfg.l2_leaf_reg,
            random_strength=cfg.random_strength,
            random_state=random_state,
        )
        mae, r2 = evaluate_model(model, X_test, y_test)
        artifact_info = save_artifacts(model, mae, r2)

        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.log_artifact(artifact_info.model_path, artifact_path="model_artifacts")
        mlflow.log_artifact(artifact_info.metrics_path, artifact_path="metrics")

        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
        )

        model_uri = f"runs:/{run.info.run_id}/model"

        model_version = None
        if register_model:
            registration = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )
            model_version = registration.version

        result = {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "model_name": model_name,
            "model_version": model_version,
            "logged_model_uri": model_uri,
            "mae": mae,
            "r2": r2,
            "artifacts": {
                "model_path": str(artifact_info.model_path),
                "metrics_path": str(artifact_info.metrics_path),
            },
        }

    return result


def build_parser() -> argparse.ArgumentParser:
    cfg = get_settings()
    parser = argparse.ArgumentParser(
        description="Train EGE prediction model and log to MLflow. "
                    "All defaults come from .env.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--subject-filter", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--register-model",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        default=False,
        help="Skip data preparation, read from prepared_data table in DB.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        subject_filter=args.subject_filter,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        register_model=args.register_model,
        tracking_uri=args.tracking_uri,
        skip_prepare=args.skip_prepare,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
