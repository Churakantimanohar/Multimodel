import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, seq_dim: int = 256, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.bilstm = nn.LSTM(seq_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, fused_sequence):
        # fused_sequence: (B, N_modalities, seq_dim)
        lstm_out, _ = self.bilstm(fused_sequence)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.out(self.dropout(pooled))
        return logits
