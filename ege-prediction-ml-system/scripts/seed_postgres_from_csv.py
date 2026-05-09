from __future__ import annotations

import argparse

from app.db.repository import initialize_schema, seed_input_data_from_dataframe
from app.training.data import load_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed PostgreSQL students_input from prepared_rolling_dataset.csv.",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Number of rows to insert.",
    )
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="Truncate students_input and predictions before inserting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_schema()
    df = load_prepared_dataset()
    inserted = seed_input_data_from_dataframe(
        df=df,
        limit=args.limit,
        clear_existing=args.clear_existing,
    )
    print(f"Inserted {inserted} rows into students_input.")


if __name__ == "__main__":
    main()
