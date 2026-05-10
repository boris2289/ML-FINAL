#!/usr/bin/env bash
set -euo pipefail

# Все параметры из .env
streamlit run app/frontend/streamlit_app.py --server.port "${STREAMLIT_PORT:-8501}"
