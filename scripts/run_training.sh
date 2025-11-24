#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || echo "Activate your venv first"
python -m src.training.train "$@"