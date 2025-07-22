import torch
import torch.nn as nn

class TransformerExtractor(nn.Module):
    def __init__(self,
                 input_dim=2,      # 每辆车的 (x,y)
                 d_model=64,       # Transformer 内部维度
                 nhead=4,
                 num_layers=2,
                 seq_len=20):
        super().__init__()
        # 线性映射到 d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        # 简单的位置编码（可替换为 nn.Embedding）
        self.pos_enc = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)

        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        #
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 最终池化（取序列第一个位置或做 mean-pool）
        self.pool = lambda x: x.mean(dim=1)  # [B, seq, d_model] -> [B, d_model]

    def forward(self, relative_vecs):
        # relative_vecs: [B, seq_len, 2]
        x = self.input_proj(relative_vecs)             # -> [B, seq_len, d_model]
        x = x + self.pos_enc.unsqueeze(0)              # 加上位置编码
        x = self.transformer(x.permute(1,0,2))         # Transformer 要求 [seq, B, d_model]
        x = x.permute(1,0,2)                           # -> [B, seq, d_model]
        z_rel = self.pool(x)                           # -> [B, d_model]
        return z_rel
