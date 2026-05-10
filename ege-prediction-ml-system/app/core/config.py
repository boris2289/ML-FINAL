"""
Единый конфиг приложения.

Все параметры читаются из .env файла в корне проекта.
Новый эксперимент = новый .env → перезапуск.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_root() -> Path:
    """Корень проекта — папка с .env или два уровня выше core/."""
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


ROOT_DIR = _find_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Эксперимент ───────────────────────────
    experiment_name: str = "ege-prediction-experiment"
    run_name: str = "catboost-ege-prediction"
    model_name: str = "ege-catboost-regressor"
    register_model: bool = True

    # ── Данные ────────────────────────────────
    raw_csv_filename: str = "2025_07_24_el_school.csv"
    subject_filter: str = ""
    ege_min_score: int = 30
    ege_max_score: int = 100
    test_size: float = 0.2

    # Списковые поля хранятся как строки (CSV),
    # потому что pydantic-settings пытается JSON-парсить List[T]
    # до того как сработает валидатор.
    allowed_classes: str = "9,10,11"

    # ── Гиперпараметры CatBoost ───────────────
    iterations: int = 3000
    depth: int = 12
    learning_rate: float = 0.03
    l2_leaf_reg: float = 7
    random_strength: float = 12
    random_state: int = 54
    loss_function: str = "MAE"

    # ── Фичи (CSV-строки) ────────────────────
    cat_features: str = "student_target,student_class,course_name,subject_name"

    used_features: str = (
        "student_target,student_class,course_name,subject_name,"
        "homework_done_mark,test_part_one,test_part_two,"
        "homework_lag_1,homework_lag_2,test1_lag_1,test2_lag_1,"
        "homework_diff,test1_diff,test2_diff,"
        "homework_rolling_mean_3,homework_rolling_std_3,"
        "test1_rolling_mean_3,test2_rolling_std_3,"
        "homework_max,homework_min,test1_max,test1_min,"
        "test2_max,test2_min"
    )

    target_col: str = "course_student_ege_result"

    # ── Rolling / Lag ─────────────────────────
    lag_periods: str = "1,2"
    rolling_window: int = 3

    # ── PostgreSQL ────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ege_predictions"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # ── MLflow ────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5001"

    # ── API / Frontend ────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    streamlit_port: int = 8501

    # ── Batch ─────────────────────────────────
    batch_interval_seconds: int = 300
    batch_limit: int = 100
    batch_model_version: str = "docker-catboost-v1"
    default_seed_limit: int = 100

    # ── Пути ──────────────────────────────────
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # ══════════════════════════════════════════
    #  Парсеры CSV → списки (свойства)
    # ══════════════════════════════════════════
    @property
    def allowed_classes_list(self) -> list[int]:
        return [int(s.strip()) for s in self.allowed_classes.split(",") if s.strip()]

    @property
    def cat_features_list(self) -> list[str]:
        return [s.strip() for s in self.cat_features.split(",") if s.strip()]

    @property
    def used_features_list(self) -> list[str]:
        return [s.strip() for s in self.used_features.split(",") if s.strip()]

    @property
    def lag_periods_list(self) -> list[int]:
        return [int(s.strip()) for s in self.lag_periods.split(",") if s.strip()]

    # ══════════════════════════════════════════
    #  Вычисляемые пути
    # ══════════════════════════════════════════
    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def data_path(self) -> Path:
        return ROOT_DIR / self.data_dir

    @property
    def artifacts_path(self) -> Path:
        return ROOT_DIR / self.artifacts_dir

    @property
    def raw_csv_path(self) -> Path:
        return self.data_path / self.raw_csv_filename

    @property
    def model_path(self) -> Path:
        return self.artifacts_path / "catboost_model.cbm"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_path / "metrics.json"

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    @property
    def pg_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def subject_filter_value(self) -> str | None:
        return self.subject_filter if self.subject_filter else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
