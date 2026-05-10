#!/usr/bin/env bash
set -euo pipefail

# Все параметры берутся из .env автоматически.
# CLI-аргументы (если переданы) перекрывают .env.

python -m app.training.train_with_mlflow "$@"
