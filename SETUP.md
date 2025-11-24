## Environment & Installation Guide (macOS)

### 1. Recommended Python Version

Use Python 3.10 or 3.11 for full feature (mediapipe). Python 3.13 lacks mediapipe wheel; project will fall back to zero landmarks.

To install Python 3.11 via pyenv (optional):

```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
```

Then create environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch (CPU/MPS build)

Apple Silicon / Intel (no CUDA) latest supported (Python 3.13 needs newer torch):

```bash
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If version resolution fails:

```bash
pip install --no-cache-dir torch torchvision torchaudio
```

Or use Conda (recommended for some configurations):

```bash
conda create -n menthheath python=3.10 -y
conda activate menthheath
conda install pytorch torchvision torchaudio -c pytorch -y
```

### 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

If `pyaudio` fails:

```bash
brew install portaudio
pip install pyaudio
```

### 4. Verify Environment

```bash
bash scripts/check_env.sh
```

### 5. First Synthetic Training Run

```bash
python src/training/train.py
```

### Common Installation Errors

| Symptom                     | Cause                  | Resolution                                                 |
| --------------------------- | ---------------------- | ---------------------------------------------------------- |
| Torch wheel not found       | Outdated pip           | `pip install --upgrade pip` then retry                     |
| Mediapipe protobuf mismatch | protobuf>=4 installed  | `pip install 'protobuf<4' --force-reinstall`               |
| pyaudio build failure       | Missing portaudio      | `brew install portaudio` then `pip install pyaudio`        |
| SSL error downloading model | Corporate proxy / cert | Set `REQUESTS_CA_BUNDLE` or use offline fallback tokenizer |

### Offline Mode

If huggingface model download blocked, fallback tokenizer is used automatically (see `text_preprocess.py`). Provide local HF model with:

```bash
transformers-cli download bert-base-uncased
```

### Next Steps

Replace synthetic data with a CSV manifest: `data/train_manifest.csv` containing header:

```
text,audio_path,video_path,label
Feeling okay today,data/audio/sample1.wav,data/video/sample1.mp4,Normal
```

Then modify training to load it (to be integrated).
