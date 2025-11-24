import librosa
import numpy as np
from typing import Dict
import os
from tempfile import NamedTemporaryFile
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False
from ..utils.config import get_config

_cfg = get_config()


def _load_audio(path: str):
    ext = os.path.splitext(path)[1].lower()
    # Librosa can often handle mp3; if it fails and pydub available, convert.
    try:
        y, sr = librosa.load(path, sr=_cfg.audio_sample_rate)
        return y, sr
    except Exception:
        if ext == '.mp3' and _HAS_PYDUB:
            with NamedTemporaryFile(suffix='.wav') as tf:
                audio = AudioSegment.from_file(path, format='mp3')
                audio.export(tf.name, format='wav')
                y, sr = librosa.load(tf.name, sr=_cfg.audio_sample_rate)
                return y, sr
        raise

def extract_audio_features(path: str) -> Dict[str, np.ndarray]:
    y, sr = _load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_cfg.n_mfcc)
    pitch = librosa.yin(y, fmin=50, fmax=500)
    energy = librosa.feature.rms(y=y)
    # Simple jitter/shimmer proxies (not clinical): frame-to-frame diff statistics
    frame_amp = librosa.feature.rms(y=y).flatten()
    jitter = np.mean(np.abs(np.diff(pitch)))
    shimmer = np.mean(np.abs(np.diff(frame_amp)))
    return {
        'mfcc': mfcc.T,  # time x n_mfcc
        'pitch': pitch.reshape(-1,1),
        'energy': energy.T,
        'jitter': np.array([[jitter]]),
        'shimmer': np.array([[shimmer]])
    }
