# Multimodal Mental Health Detection (Text + Audio + Video)

An end-to-end prototype for classifying mental health related states (Normal, Anxiety, Stress, Depression) using three modalities: user text, short audio clip, and facial video frame/stream. Designed for experimentation and extension – not a diagnostic or medical device.

## Key Components

| Layer         | Description                                                                            |
| ------------- | -------------------------------------------------------------------------------------- |
| Text Encoder  | BERT tokenizer + embedding -> LSTM pooling                                             |
| Audio Encoder | MFCC + pitch + energy + jitter + shimmer statistics -> LSTM & statistical pooling      |
| Video Encoder | Frame resized + lightweight CNN (MobileNetV2 style) + (fallback) zero facial landmarks |
| Fusion        | Multi-head attention over projected modality embeddings -> fused sequence              |
| Classifier    | BiLSTM over fused sequence -> logits (4-class softmax)                                 |
| Inference UIs | Streamlit live app, Flask REST API, realtime script                                    |

## Current Status

Implemented: synthetic training demo, checkpoint saving/loading, MP3 audio support (via pydub conversion), live webcam frame capture, heuristic facial expression overlay (simple cascade + label), session history + JSON report generation, manual and auto live prediction. MediaPipe landmarks are currently replaced with zeros on Python 3.13 (wheel unavailable). Code paths prepared for future landmark integration.

## Pipeline Overview

1. Data acquisition (place raw or prepared files under `data/`).
2. Preprocessing: text tokenization, audio feature extraction (librosa), video frame sampling + (fallback) landmarks.
3. Unimodal encoders produce modality embeddings.
4. Attention fusion combines modality embeddings into fused sequence.
5. Classifier predicts class probabilities.
6. Metrics (accuracy, precision, recall, macro F1) tracked during training; plots saved.
7. Inference via Streamlit (upload or live), Flask API, or realtime script.

## Folder Structure

```
menthheath/
  README.md
  requirements.txt
  src/
    data/
      dataloaders/
        daic_woz_loader.py
        iemocap_loader.py
        mosei_loader.py
    preprocessing/
      text_preprocess.py
      audio_preprocess.py
      video_preprocess.py
    models/
      text_model.py
      audio_model.py
      video_model.py
      fusion/
        attention_fusion.py
      classifier.py
    training/
      train.py
      evaluate.py
      metrics.py
    inference/
      realtime.py
      api.py
      streamlit_app.py
    utils/
      config.py
      logging_utils.py
      visualization.py
  scripts/
    download_datasets.sh
    run_training.sh
    run_inference.sh
  tests/
    test_forward.py
  diagrams/
    architecture.md
```

## Installation

Python 3.13 environment suggested. PyTorch wheels for Mac CPU/MPS: versions earlier than 2.6 are not published for 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # excludes torch trio
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

Optional (improve live performance):

```bash
brew install portaudio   # if using microphone (pyaudio)
pip install pyaudio watchdog
```

If MediaPipe 3.13 wheels become available:

```bash
pip install mediapipe
```

## Dataset Preparation (Summary)

- DAIC-WOZ: Register & download; place audio (.wav), transcripts (.txt), and video frames (extracted) under `data/daic_woz/`.
- IEMOCAP: Structured sessions; parse dialogue text & audio.
- MOSEI: Use CMU SDK or preprocessed features; map to required mental health labels (custom mapping script to be added).

Detailed steps are documented in `scripts/download_datasets.sh` and inline comments in each loader.

## Quick Synthetic Training Demo

Generates synthetic samples to validate the pipeline and produce a checkpoint in `outputs/`:

```bash
source .venv/bin/activate
python -m src.training.train
```

Configurable parameters live in `src/utils/config.py` (epochs, batch size, feature dims, checkpoint paths). Training automatically saves metrics plots (loss & F1) when complete.

## Using Real Datasets

Dataset loaders are scaffolded; you must supply actual dataset files (not included). Place audio/text/video under a consistent manifest referencing file paths.

Suggested CSV manifest columns:

```
id,text,audio_path,video_path,label
```

Adapt `train.py` to read real samples (see synthetic dataset stub for structure).

## Streamlit Live Inference

Launch UI:

```bash
source .venv/bin/activate
streamlit run src/inference/streamlit_app.py
```

Modes:

- Upload Prediction: provide text, audio (wav/mp3), video (mp4) then click "Run Upload Prediction".
- Live Mode: enable webcam stream (WebRTC), enter text; use Manual Live Predict or toggle auto-update (if implemented) to refresh predictions.
- Expression Overlay: heuristic facial expression label drawn on frame (non-landmark, just cascade + simple rules).
- Session Report: after some predictions, click Generate/Download Report to save JSON with history & final probabilities.

Tips:

- First prediction may be slower (model warm-up).
- If MP3 upload fails, ensure `pydub` installed and `ffmpeg` accessible (optional). For pure Python fallback, small MP3s should still convert.

## REST API (Flask)

Start server:

```bash
source .venv/bin/activate
python src/inference/api.py
```

Health check:

```bash
curl http://localhost:8080/health
```

Prediction endpoint expects existing file paths accessible to the server:

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"I feel stressed","audio_path":"/absolute/path/sample.wav","video_path":"/absolute/path/sample.mp4"}'
```

Response:

```json
{
  "probabilities": {
    "Normal": 0.2,
    "Anxiety": 0.35,
    "Stress": 0.3,
    "Depression": 0.15
  },
  "prediction": "Anxiety"
}
```

## Paper → Code Mapping

Reference document: `diagrams/architecture.md` maps conceptual modules (feature extraction, fusion attention, classifier) to implemented files in `src/models/*` and `src/preprocessing/*`.

## Metrics & Visualization

Outputs:

- Checkpoints: `outputs/ckpt_epoch_X.pt` and `outputs/ckpt_final.pt`
- Plots: `outputs/train_loss.png`, `outputs/train_f1.png`
- Session JSON reports (when downloaded): user-specified filename in browser

Extend `src/utils/visualization.py` (placeholder) or add confusion matrix generation using stored predictions after real dataset training.

## Recommendations Layer

File `streamlit_app.py` includes a simple heuristic `recommend(probs)` mapping top class to guidance text (non-clinical). Replace with evidence-based suggestions only if you have verified protocols.

## Troubleshooting

| Issue                      | Cause                                | Fix                                                              |
| -------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| ModuleNotFoundError: 'src' | Path not injected early              | Ensure path injection appears before imports (already patched)   |
| Torch version not found    | Python 3.13 + old requested versions | Use torch 2.9.1 + torchvision 0.24.1 + torchaudio 2.9.1          |
| MP3 rejected               | pydub not installed                  | `pip install pydub` and optionally `brew install ffmpeg`         |
| No live prediction         | Frame/text missing                   | Capture webcam frame and enter min length text (>5 chars)        |
| Report empty               | No predictions yet                   | Generate after at least one prediction (history creates entries) |
| Mediapipe unavailable      | Wheel not published                  | Landmarks replaced with zeros; retry when wheel released         |

## Extending / Next Steps

- Integrate real datasets & balanced sampling.
- Replace heuristic expression with landmark-driven emotion classifier.
- Add microphone streaming buffer for continuous audio capture.
- Persist session histories to local storage or database (e.g., SQLite).
- Implement evaluation script for confusion matrix & per-class F1.
- Model optimization (quantization / distillation) for faster live inference.

## Ethical / Disclaimer

This repository is for research and educational prototyping. It does NOT provide medical advice. For any mental health concerns, seek qualified professional assistance. Do not deploy outputs directly in clinical settings without rigorous validation, ethical review, and compliance procedures.

## License

No explicit license text included. Add one (e.g., MIT) before sharing publicly.

## Performance Note

Any stated accuracy targets (e.g., 88%) are illustrative and depend heavily on dataset quality, label mapping, and hyperparameter tuning. Expect variation; treat synthetic demo results only as functional smoke tests.
