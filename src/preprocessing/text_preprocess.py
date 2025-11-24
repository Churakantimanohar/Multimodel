from transformers import AutoTokenizer
from typing import List, Dict
from ..utils.config import get_config

_cfg = get_config()
try:
    _tokenizer = AutoTokenizer.from_pretrained(_cfg.text_model_name)
except Exception:
    # Fallback simple whitespace tokenizer if HF download fails (offline mode)
    class _FallbackTokenizer:
        def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors=None):
            import torch
            input_ids = []
            attention = []
            for t in texts:
                tokens = t.lower().split()
                ids = [min(len(tok),100) for tok in tokens][:max_length]
                pad_len = max_length - len(ids)
                ids = ids + [0]*pad_len
                att = [1]*(max_length-pad_len) + [0]*pad_len
                input_ids.append(ids)
                attention.append(att)
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention, dtype=torch.long)
            }
    _tokenizer = _FallbackTokenizer()


def tokenize_texts(texts: List[str]):
    return _tokenizer(texts, padding=True, truncation=True, max_length=_cfg.max_seq_len, return_tensors='pt')


def clean_text(t: str) -> str:
    return t.strip()


def preprocess_text_batch(batch: List[str]) -> Dict:
    cleaned = [clean_text(x) for x in batch]
    return tokenize_texts(cleaned)
