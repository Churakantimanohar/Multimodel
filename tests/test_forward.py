import os
import sys
import torch

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.text_model import TextEncoder
from src.models.audio_model import AudioEncoder
from src.models.video_model import VideoEncoder
from src.models.fusion.attention_fusion import MultiModalAttentionFusion
from src.models.classifier import FusionClassifier
from src.utils.config import get_config

cfg = get_config()

def test_forward_pass():
    text_enc = TextEncoder()
    audio_enc = AudioEncoder()
    video_enc = VideoEncoder()
    fusion = MultiModalAttentionFusion(256,256,256,fusion_dim=256)
    classifier = FusionClassifier(seq_dim=256, num_classes=cfg.num_classes)

    input_ids = torch.randint(0,100,(2,cfg.max_seq_len))
    attention_mask = torch.ones_like(input_ids)
    text_vec = text_enc(input_ids, attention_mask)

    mfcc = torch.randn(100, cfg.n_mfcc)
    pitch = torch.randn(50)
    energy = torch.randn(50)
    jitter = torch.randn(1)
    shimmer = torch.randn(1)
    audio_vec = audio_enc(mfcc, pitch, energy, jitter, shimmer)
    audio_vec = audio_vec.repeat(2,1)

    frames = torch.randn(2,3,cfg.video_frame_size,cfg.video_frame_size)
    landmarks = torch.randn(2,32,468,3)[:,0]  # simplify
    video_vec = video_enc(frames, landmarks)

    fused_pooled, fused_seq = fusion(text_vec, audio_vec, video_vec)
    logits = classifier(fused_seq)
    assert logits.shape == (2, cfg.num_classes)

if __name__ == '__main__':
    test_forward_pass(); print('Forward test passed')
