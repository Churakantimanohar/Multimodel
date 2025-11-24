import csv
import os
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"Normal":0, "Anxiety":1, "Stress":2, "Depression":3}

class ManifestMultimodalDataset(Dataset):
    """Generic dataset reading a CSV manifest with columns:
    text,audio_path,video_path,label
    Paths can be relative to manifest directory.
    """
    def __init__(self, manifest_path: str, limit: int = None):
        self.manifest_path = manifest_path
        self.root_dir = os.path.dirname(manifest_path)
        self.samples: List[Dict[str, Any]] = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' not in row or 'label' not in row:
                    raise ValueError('Manifest must contain text and label columns')
                audio_path = row.get('audio_path','').strip()
                video_path = row.get('video_path','').strip()
                sample = {
                    'text': row['text'],
                    'audio_path': os.path.join(self.root_dir, audio_path) if audio_path else None,
                    'video_path': os.path.join(self.root_dir, video_path) if video_path else None,
                    'label_str': row['label']
                }
                if sample['label_str'] not in LABEL_MAP:
                    continue  # skip unknown labels
                self.samples.append(sample)
                if limit and len(self.samples) >= limit:
                    break
        if not self.samples:
            raise RuntimeError('No samples loaded from manifest. Check file paths and labels.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s