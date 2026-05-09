#!/usr/bin/env bash
set -euo pipefail

export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
streamlit run app/frontend/streamlit_app.py
