import torch
import torch.nn as nn
from torchvision import models

class VideoEncoder(nn.Module):
    def __init__(self, landmark_dim: int = 468*3, cnn_out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.cnn = nn.Sequential(*list(base.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cnn_proj = nn.Linear(1280, cnn_out_dim)
        self.landmark_proj = nn.Linear(landmark_dim, hidden_dim)
        self.final_proj = nn.Linear(cnn_out_dim + hidden_dim, hidden_dim)

    def forward(self, frames_tensor, landmarks):
        # frames_tensor: (B, C, H, W)
        feats = self.cnn(frames_tensor)
        pooled = self.pool(feats).flatten(1)
        cnn_embed = self.cnn_proj(pooled)  # (B, cnn_out_dim)

        # landmarks possible shapes:
        # (B, T, 468, 3) -> temporal sequence
        # (B, 468, 3) -> single frame
        if landmarks.dim() == 4:
            B, T, P, C = landmarks.shape
            lm = landmarks.view(B, T, -1)  # (B,T,468*3)
            lm_mean = torch.mean(lm, dim=1)  # (B,468*3)
        elif landmarks.dim() == 3:
            # Could be (B,468,3) or (T,468,3) when B=1 in frames
            if landmarks.size(0) == frames_tensor.size(0):
                B, P, C = landmarks.shape
                lm_mean = landmarks.view(B, -1)
            else:
                # Treat first dim as temporal, aggregate
                T, P, C = landmarks.shape
                lm_mean = landmarks.view(T, -1).mean(dim=0, keepdim=True)  # (1,468*3)
        else:
            raise ValueError(f"Unexpected landmarks shape {landmarks.shape}")

        landmark_embed = self.landmark_proj(lm_mean)  # (B, hidden_dim)
        # Repeat landmark embed if batch mismatch
        if landmark_embed.size(0) != cnn_embed.size(0):
            if landmark_embed.size(0) == 1:
                landmark_embed = landmark_embed.repeat(cnn_embed.size(0), 1)
            else:
                raise RuntimeError("Batch size mismatch between frames and landmarks")
        combined = torch.cat([cnn_embed, landmark_embed], dim=1)
        return self.final_proj(combined)
