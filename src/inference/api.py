from flask import Flask, request, jsonify
import torch
import cv2
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import get_config
from src.models.text_model import TextEncoder
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder
from src.models.fusion.attention_fusion import MultiModalAttentionFusion
from src.models.classifier import FusionClassifier
from src.inference.checkpoint_utils import load_latest
from src.preprocessing.text_preprocess import preprocess_text_batch
from src.preprocessing.audio_preprocess import extract_audio_features
from src.preprocessing.video_preprocess import process_video

cfg = get_config()
app = Flask(__name__)

text_enc = TextEncoder().to(cfg.device)
audio_enc = AudioEncoder().to(cfg.device)
video_enc = VideoEncoder().to(cfg.device)
fusion = MultiModalAttentionFusion(dim_text=256, dim_audio=256, dim_video=256, fusion_dim=cfg.fusion_hidden_dim).to(cfg.device)
classifier = FusionClassifier(seq_dim=cfg.fusion_hidden_dim, num_classes=cfg.num_classes).to(cfg.device)

# Attempt to load latest checkpoint
ckpt_path, loaded = load_latest({
    'text_enc': text_enc,
    'audio_enc': audio_enc,
    'video_enc': video_enc,
    'fusion': fusion,
    'classifier': classifier
})
if ckpt_path:
    print(f"[API] Loaded checkpoint {ckpt_path} modules={loaded}")
else:
    print("[API] No checkpoint found; using randomly initialized weights")

LABELS = ["Normal","Anxiety","Stress","Depression"]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    audio_path = data.get('audio_path')
    video_path = data.get('video_path')

    tokenized = preprocess_text_batch([text])
    text_vec = text_enc(tokenized['input_ids'].to(cfg.device), tokenized['attention_mask'].to(cfg.device))

    feats = extract_audio_features(audio_path)
    audio_vec = audio_enc(
        torch.tensor(feats['mfcc'], dtype=torch.float32).to(cfg.device),
        feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer']
    )

    video_feats = process_video(video_path)
    # Placeholder frames for CNN: using first landmark frame as blank image
    frame_tensor = torch.zeros(1,3,cfg.video_frame_size,cfg.video_frame_size).to(cfg.device)
    landmarks = torch.tensor(video_feats['landmarks'], dtype=torch.float32).to(cfg.device)
    video_vec = video_enc(frame_tensor, landmarks)

    fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
    logits = classifier(fused_seq)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    result = {label: float(probs[i]) for i,label in enumerate(LABELS)}
    return jsonify({'probabilities': result, 'prediction': LABELS[int(probs.argmax())]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
