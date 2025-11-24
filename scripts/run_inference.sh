#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || echo "Activate your venv first"
streamlit run src/inference/streamlit_app.py "$@"