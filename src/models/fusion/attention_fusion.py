import torch
import torch.nn as nn

class MultiModalAttentionFusion(nn.Module):
    def __init__(self, dim_text: int, dim_audio: int, dim_video: int, fusion_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.text_proj = nn.Linear(dim_text, fusion_dim)
        self.audio_proj = nn.Linear(dim_audio, fusion_dim)
        self.video_proj = nn.Linear(dim_video, fusion_dim)
        self.attn = nn.MultiheadAttention(fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_vec, audio_vec, video_vec):
        # Each vec: (B, D)
        B = text_vec.size(0)
        stacked = torch.stack([
            self.text_proj(text_vec),
            self.audio_proj(audio_vec),
            self.video_proj(video_vec)
        ], dim=1)  # (B, 3, fusion_dim)
        attn_out, _ = self.attn(stacked, stacked, stacked)
        fused = self.norm(attn_out + stacked)
        pooled = torch.mean(fused, dim=1)  # (B, fusion_dim)
        return pooled, fused  # fused sequence for classifier
