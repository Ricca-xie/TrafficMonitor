import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class MultiHeadAttention(BaseFeaturesExtractor):
    def __init__(self,
                 input_dim: int = 2,
                 d_model: int = 64,
                 dropout: float = 0.1,
                 nhead: int = 4,
                 dim_feedforward: int = 128,
                 num_layers: int = 2,
                 seq_len: int = 20):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.proj_kv = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model+d_model, d_model)


    def forward(self, ac_attr, rel_vecs, brk_spot, cov_cnt):
        other = torch.stack([ac_attr, brk_spot, cov_cnt], dim=1)
        kv = self.proj_kv(other)
        attn_out, attn_w = self.cross_attn(query=rel_vecs, key=kv, values=kv)

        fused = torch.cat([rel_vecs, attn_out], dim=1)
        fused = self.proj_out(fused)

        return fused