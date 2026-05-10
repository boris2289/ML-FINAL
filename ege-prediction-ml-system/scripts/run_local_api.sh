#!/usr/bin/env bash
set -euo pipefail

# Все параметры из .env
uvicorn app.api.main:app --host 127.0.0.1 --port "${API_PORT:-8000}" --reload
