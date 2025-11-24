import torch
import torch.nn as nn
from transformers import AutoModel
from ..utils.config import get_config

_cfg = get_config()

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = _cfg.text_model_name, hidden_dim: int = 256):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(self.transformer.config.hidden_size, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence)
        pooled = torch.mean(lstm_out, dim=1)
        return pooled  # (batch, hidden_dim)
