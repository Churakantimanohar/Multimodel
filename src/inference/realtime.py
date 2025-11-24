import torch
try:
    import pyaudio  # optional dependency; may require install
    _HAS_PYAUDIO = True
except Exception:
    _HAS_PYAUDIO = False
import cv2
import os, sys
from transformers import AutoTokenizer

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
from src.preprocessing.video_preprocess import extract_face_landmarks

# NOTE: Real-time pipeline skeleton; audio capture uses PyAudio; video via OpenCV webcam.

cfg = get_config()

def load_models(device):
    text_enc = TextEncoder().to(device)
    audio_enc = AudioEncoder().to(device)
    video_enc = VideoEncoder().to(device)
    fusion = MultiModalAttentionFusion(dim_text=256, dim_audio=256, dim_video=256, fusion_dim=cfg.fusion_hidden_dim).to(device)
    classifier = FusionClassifier(seq_dim=cfg.fusion_hidden_dim, num_classes=cfg.num_classes).to(device)
    ckpt_path, loaded = load_latest({
        'text_enc': text_enc,
        'audio_enc': audio_enc,
        'video_enc': video_enc,
        'fusion': fusion,
        'classifier': classifier
    })
    if ckpt_path:
        print(f"[Realtime] Loaded checkpoint {ckpt_path} modules={loaded}")
    else:
        print('[Realtime] No checkpoint found; using random weights')
    return text_enc, audio_enc, video_enc, fusion, classifier


def predict(text, audio_path, frame, cached_models=None):
    device = cfg.device
    if cached_models is None:
        cached_models = load_models(device)
    text_enc, audio_enc, video_enc, fusion, classifier = cached_models
    tokenized = preprocess_text_batch([text])
    text_vec = text_enc(tokenized['input_ids'].to(device), tokenized['attention_mask'].to(device))

    if audio_path and os.path.exists(audio_path):
        feats = extract_audio_features(audio_path)
        audio_vec = audio_enc(
            torch.tensor(feats['mfcc'], dtype=torch.float32).to(device),
            feats['pitch'], feats['energy'], feats['jitter'], feats['shimmer']
        )
    else:
        # Fallback zero vector if audio missing
        audio_vec = torch.zeros_like(text_vec)

    landmarks = extract_face_landmarks(frame)  # (1,468,3)
    frame_tensor = torch.tensor(frame).permute(2,0,1).unsqueeze(0).float()/255.0
    video_vec = video_enc(frame_tensor.to(device), torch.tensor(landmarks, dtype=torch.float32).to(device))

    fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
    logits = classifier(fused_seq)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs


def run_live(text: str = "live session", audio_seconds: int = 3):
    device = cfg.device
    models = load_models(device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Webcam not accessible')
        return
    temp_wav = 'temp_live_audio.wav'
    if _HAS_PYAUDIO:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    else:
        stream = None
    import wave, time
    frames_audio = []
    start_time = time.time()
    print("Press 'q' to quit live capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame grab failed')
            break
        cv2.imshow('Live Video', frame)
        # Collect audio
        if stream and (time.time() - start_time) < audio_seconds:
            data = stream.read(1024, exception_on_overflow=False)
            frames_audio.append(data)
        # Every N frames perform prediction
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            # Write audio temp
            if stream:
                wf = wave.open(temp_wav, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames_audio))
                wf.close()
                audio_path = temp_wav
            else:
                audio_path = None
            probs = predict(text, audio_path, frame, cached_models=models)
            print('Probabilities:', probs)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if stream:
        stream.stop_stream(); stream.close(); pa.terminate()
    if os.path.exists(temp_wav):
        os.remove(temp_wav)


if __name__ == '__main__':
    run_live()
