from __future__ import annotations

import argparse
import json
import os

import mlflow
import mlflow.catboost

from app.training.data import load_prepared_dataset, prepare_full_dataset
from app.training.pipeline import evaluate_model, run_training_pipeline, save_artifacts, train_catboost
from app.core.constants import USED_FEATURES, TARGET_COL

from sklearn.model_selection import train_test_split


def str_to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def run_training(
    *,
    iterations: int = 3000,
    depth: int = 12,
    learning_rate: float = 0.03,
    random_state: int = 54,
    subject_filter: str | None = None,
    experiment_name: str = "ege-prediction-experiment",
    model_name: str = "ege-catboost-regressor",
    register_model: bool = True,
    tracking_uri: str | None = None,
    skip_prepare: bool = False,
) -> dict:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Подготовка данных
    if skip_prepare:
        df = load_prepared_dataset()
        if subject_filter:
            df = df[df["subject_name"] == subject_filter]
    else:
        df = prepare_full_dataset(subject_filter=subject_filter)

    X = df[USED_FEATURES]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    with mlflow.start_run(run_name="catboost-ege-prediction") as run:
        mlflow.log_params({
            "model_type": "CatBoostRegressor",
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "subject_filter": subject_filter or "all",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
        })

        model = train_catboost(
            X_train, y_train, X_test, y_test,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
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
    parser = argparse.ArgumentParser(description="Train EGE prediction model and log to MLflow.")
    parser.add_argument("--iterations", type=int, default=int(os.getenv("ITERATIONS", "3000")))
    parser.add_argument("--depth", type=int, default=int(os.getenv("DEPTH", "12")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.03")))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "54")))
    parser.add_argument("--subject-filter", default=os.getenv("SUBJECT_FILTER"))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "ege-prediction-experiment"),
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "ege-catboost-regressor"),
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        default=str_to_bool(os.getenv("REGISTER_MODEL", "true")),
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        default=str_to_bool(os.getenv("SKIP_PREPARE", "false")),
        help="Skip data preparation, use existing prepared_rolling_dataset.csv.",
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