from __future__ import annotations

import argparse
import json

from app.batch.pipeline import run_batch_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch prediction against PostgreSQL students_input.",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of rows to process in one run.",
    )
    parser.add_argument(
        "--model-version", type=str, default="local-catboost-v1",
        help="Model version label to store.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_batch_prediction(limit=args.limit, model_version=args.model_version)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
