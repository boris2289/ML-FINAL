"""
Загрузка CSV в БД (полный пайплайн: raw → cleaned → prepared)
и/или seed students_input из prepared_data.

Все настройки из .env.
"""
from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.db.repository import initialize_schema, seed_input_data_from_dataframe
from app.training.data import load_prepared_dataset, prepare_full_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load CSV into DB and optionally seed students_input.",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Run full data preparation pipeline (CSV → raw → cleaned → prepared in DB).",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Seed students_input from prepared_data in DB.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of rows to insert into students_input (default from .env).",
    )
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="Truncate students_input and predictions before inserting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_settings()

    initialize_schema()
    print("Таблицы инициализированы.")

    if args.prepare:
        print("Запуск полного пайплайна подготовки данных...")
        df = prepare_full_dataset(experiment=cfg.experiment_name)
        print(f"Подготовлено {len(df)} строк → prepared_data.")

    if args.seed:
        limit = args.limit or cfg.default_seed_limit
        print(f"Seed students_input (limit={limit})...")
        df = load_prepared_dataset(experiment=cfg.experiment_name)
        inserted = seed_input_data_from_dataframe(
            df=df,
            limit=limit,
            clear_existing=args.clear_existing,
        )
        print(f"Inserted {inserted} rows into students_input.")

    if not args.prepare and not args.seed:
        print("Используйте --prepare и/или --seed. См. --help.")


if __name__ == "__main__":
    main()
