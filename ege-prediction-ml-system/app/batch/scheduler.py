from __future__ import annotations

import os
import time

from app.batch.pipeline import run_batch_prediction


INTERVAL_SECONDS = int(os.getenv("BATCH_INTERVAL_SECONDS", "300"))
MODEL_VERSION = os.getenv("BATCH_MODEL_VERSION", "scheduled-catboost-v1")
BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "100"))


def main() -> None:
    print(
        f"Starting batch scheduler with interval={INTERVAL_SECONDS}s, "
        f"limit={BATCH_LIMIT}, model_version={MODEL_VERSION}"
    )
    while True:
        try:
            result = run_batch_prediction(limit=BATCH_LIMIT, model_version=MODEL_VERSION)
            print(result)
        except Exception as exc:
            print(f"Batch scheduler iteration failed: {exc}")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
