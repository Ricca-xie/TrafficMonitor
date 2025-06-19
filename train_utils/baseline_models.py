'''
@Author: Ricca
@Date: 2024-07-16
@Description: Custom Model
@LastEditTime:
'''
import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class CustomModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int ):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        ac_attr_dim = observation_space["ac_attr"].shape[0]
        veh_pos_dim = observation_space["relative_vecs"].shape[0]
        bound_dim = observation_space["bound_dist"].shape[0]
        # veh_info_dim = observation_space["cover_counts"].shape[0]
        action_dim = observation_space["action_dir"].shape[0]

        self.hidden_dim = 32
        self.linear_encoder_ac = nn.Sequential(
            nn.Linear(ac_attr_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_veh = nn.Sequential(
            nn.Linear(veh_pos_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_bound = nn.Sequential(
            nn.Linear(bound_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_veh_info = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear_encoder_action = nn.Sequential(
            nn.Linear(action_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(32+32+32+32+32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations):
        ac_attr = observations["ac_attr"]
        veh_pos_dim = observations["relative_vecs"]
        veh_covered = observations["cover_counts"]
        bound = observations["bound_dist"]
        action_dir = observations["action_dir"]

        # for k, v in observations.items():
        #     print(k,"shape",v.shape)

        ac_feat = self.linear_encoder_ac(ac_attr)
        veh_feat = self.linear_encoder_veh(veh_pos_dim)
        bound_feat = self.linear_encoder_bound(bound)
        action_feat = self.linear_encoder_action(action_dir)

        batch_size = veh_covered.shape[0]
        info_feat = veh_covered.view(batch_size, 1, 1)
        veh_encoder = self.linear_encoder_veh_info(info_feat)
        veh_encoder = veh_encoder.mean(dim=1)

        all_feature_output = self.output(torch.cat([ac_feat, veh_feat, bound_feat, veh_encoder, action_feat], dim=1))
        # all_feature_output = self.output(torch.cat([ac_feat, bound_feat, action_feat], dim=1))

        # print('ac_feat', ac_feat.mean().item(),'veh_feat', veh_feat.mean().item(),'info_feat', info_feat.mean().item())
        return all_feature_output