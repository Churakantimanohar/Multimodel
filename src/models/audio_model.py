import torch
import torch.nn as nn
from ..utils.config import get_config

_cfg = get_config()

class AudioEncoder(nn.Module):
    def __init__(self, mfcc_dim: int = _cfg.n_mfcc, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(mfcc_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim*2 + 4, hidden_dim*2)

    def forward(self, mfcc, pitch, energy, jitter, shimmer):
        # mfcc shape: (B,T,n_mfcc) or (T,n_mfcc)
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)
        B = mfcc.size(0)
        lstm_out, _ = self.lstm(mfcc)  # (B,T,2H)
        pooled = torch.mean(lstm_out, dim=1)  # (B,2H)
        def feat_stat(arr):
            t = torch.tensor(arr, dtype=torch.float32)
            if t.dim() == 1:
                return torch.mean(t)
            return torch.mean(t.view(-1))
        stats = []
        for arr in [pitch, energy, jitter, shimmer]:
            if isinstance(arr, torch.Tensor):
                stats.append(arr.float().mean())
            else:
                stats.append(feat_stat(arr))
        aux = torch.stack(stats)  # (4,)
        aux = aux.unsqueeze(0).repeat(B,1)  # (B,4)
        combined = torch.cat([pooled, aux], dim=1)
        return self.proj(combined)
