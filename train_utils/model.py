import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from train_utils.traffic_transformer import TransformerExtractor

class CustomModelWithTrans(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        super().__init__(observation_space, features_dim)
        self.hidden_dim = 32
        # (1) 历史轨迹编码
        self.attr_net = nn.Sequential(
            nn.Linear(observation_space['ac_attr'], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        # (2) Transformer 提取相对位置特征
        self.trans_extractor = TransformerExtractor(
            input_dim=2, d_model=64, nhead=4, num_layers=2, seq_len=20
        )
        # (3) 标量特征编码
        self.scalar_net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )

        fused_dim = 64 + 64 + 32

        self.policy_net = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        self.value_net  = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        # 解包
        ac_attr = obs['ac_attr']        # [B, ac_dim]
        rel_vecs = obs['relative_vecs']  # [B, 20, 2]
        scalars = torch.stack([
                  obs['cover_counts'],
                  obs['bound_dist'],
                  obs['break_spot'],
                  obs['no_vehicles'].float()
                ], dim=-1)        # [B,4]

        # 三路并行
        z_attr = self.attr_net(ac_attr)            # [B,64]
        z_rel  = self.trans_extractor(rel_vecs)    # [B,64]
        z_sca  = self.scalar_net(scalars)          # [B,32]

        # 融合 & 输出
        fused = torch.cat([z_attr, z_rel, z_sca], dim=-1)
        logits = self.policy_net(fused)
        value  = self.value_net(fused).squeeze(-1)
        return logits, value
