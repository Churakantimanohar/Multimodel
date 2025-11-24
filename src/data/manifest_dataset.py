import os
import csv
import torch
from torch.utils.data import Dataset
from src.preprocessing.text_preprocess import clean_text
from src.preprocessing.audio_preprocess import extract_audio_features
from src.preprocessing.video_preprocess import process_video
from src.utils.config import get_config

cfg = get_config()
LABEL_MAP = {"Normal":0, "Anxiety":1, "Stress":2, "Depression":3}

class ManifestMultimodalDataset(Dataset):
    def __init__(self, manifest_path: str, max_samples: int = None):
        self.manifest_path = manifest_path
        self.root = os.path.dirname(manifest_path)
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get('label','').strip()
                if lbl not in LABEL_MAP:
                    continue
                item = {
                    'text': clean_text(row.get('text','')),
                    'audio_path': os.path.join(self.root, row['audio_path']) if row.get('audio_path') else None,
                    'video_path': os.path.join(self.root, row['video_path']) if row.get('video_path') else None,
                    'label': LABEL_MAP[lbl]
                }
                self.samples.append(item)
                if max_samples and len(self.samples) >= max_samples:
                    break
        if not self.samples:
            raise RuntimeError('Manifest dataset empty or labels unsupported.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Text handled later via tokenizer; return raw string
        audio_feats = None
        if s['audio_path'] and os.path.exists(s['audio_path']):
            try:
                audio_feats = extract_audio_features(s['audio_path'])
            except Exception:
                audio_feats = None
        video_feats = None
        if s['video_path'] and os.path.exists(s['video_path']):
            try:
                video_feats = process_video(s['video_path'], max_frames=32)
            except Exception:
                video_feats = None
        return {
            'text': s['text'],
            'audio_feats': audio_feats,
            'video_feats': video_feats,
            'label': s['label']
        }
