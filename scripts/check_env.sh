#!/usr/bin/env bash
set -e
echo "Python: $(python --version)"
echo "pip: $(pip --version)"
echo "Platform: $(uname -a)"
echo "Checking torch import..."
python - <<'PY'
try:
    import torch
    print('Torch version:', torch.__version__)
    print('Device count CUDA:', torch.cuda.device_count())
    print('Has MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
except Exception as e:
    print('Torch import failed:', e)
PY
echo "Checking mediapipe import..."
python - <<'PY'
try:
    import mediapipe as mp
    print('Mediapipe imported successfully')
except Exception as e:
    print('Mediapipe import failed:', e)
PY
echo "Environment check complete." 