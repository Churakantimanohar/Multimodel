# Architecture Mapping (Paper → Code)

| Paper Component            | Implementation                               | File(s)                                                              |
| -------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| Text sentiment transformer | BERT + LSTM head                             | `src/models/text_model.py`                                           |
| Audio MFCC + prosody       | MFCC + pitch/energy + jitter/shimmer -> LSTM | `src/preprocessing/audio_preprocess.py`, `src/models/audio_model.py` |
| Facial expressions         | MediaPipe landmarks + MobileNetV2 CNN        | `src/preprocessing/video_preprocess.py`, `src/models/video_model.py` |
| Feature Extraction         | Tokenizer, MFCC, landmarks                   | Preprocessing files                                                  |
| Fusion strategy            | Multihead attention over modality embeddings | `src/models/fusion/attention_fusion.py`                              |
| Classification             | BiLSTM over fused modality sequence          | `src/models/classifier.py`                                           |
| Metrics                    | Accuracy, Precision, Recall, F1              | `src/training/metrics.py`                                            |
| Training Loop              | Integrated multimodal batch pipeline         | `src/training/train.py`                                              |
| Real-time inference        | Webcam/audio/text interface                  | `src/inference/realtime.py`, `src/inference/streamlit_app.py`        |
| API Deployment             | Flask endpoint                               | `src/inference/api.py`                                               |

## Data Flow

```
Raw (text, audio, video) -> preprocessing -> modality encoders -> attention fusion -> classifier -> prediction
```

## Notes

- Replace dummy dataset with real loaders once data paths and label mappings are available.
- Extend attention fusion to cross-modal (Q from text, K/V from audio+video) if needed for performance.
