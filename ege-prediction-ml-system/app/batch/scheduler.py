from __future__ import annotations

import time

from app.batch.pipeline import run_batch_prediction
from app.core.config import get_settings


def main() -> None:
    cfg = get_settings()
    interval = cfg.batch_interval_seconds
    limit = cfg.batch_limit
    model_version = cfg.batch_model_version

    print(
        f"Starting batch scheduler with interval={interval}s, "
        f"limit={limit}, model_version={model_version}"
    )
    while True:
        try:
            result = run_batch_prediction(limit=limit, model_version=model_version)
            print(result)
        except Exception as exc:
            print(f"Batch scheduler iteration failed: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
