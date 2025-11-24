import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    text_model_name: str = "bert-base-uncased"
    max_seq_len: int = 128
    audio_sample_rate: int = 16000
    n_mfcc: int = 40
    video_frame_size: int = 224
    batch_size: int = 8
    num_epochs: int = 2  # reduced for quick demo run
    lr: float = 2e-5
    weight_decay: float = 1e-4
    fusion_hidden_dim: int = 256
    classifier_hidden_dim: int = 128
    num_classes: int = 4
    device: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def get_config() -> TrainingConfig:
    return TrainingConfig()
