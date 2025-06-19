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
        # super(CustomModel, self).__init__(observation_space, features_dim)
        ac_attr_dim = observation_space["ac_attr"].shape[0]
        # ac_shape = observation_space["ac_attr"].shape
        # passen_shape = observation_space["passen_attr"].shape
        # mask_shape = observation_space["passen_mask"].shape
        # sinr_shape = observation_space["sinr_attr"].shape
        # uncertainty_shape = observation_space["uncertainty_attr"].shape
        #
        # self.hidden_dim = 32
        # self.linear_encoder_ac = nn.Sequential(
        #     nn.Linear(ac_shape[-1], self.hidden_dim),
        #     nn.ReLU(),
        # )
        # self.linear_encoder_passen = nn.Sequential(
        #     nn.Linear(passen_shape[0]*passen_shape[-1], self.hidden_dim),
        #     nn.ReLU(),
        # )
        # self.linear_encoder_mask = nn.Sequential(
        #     nn.Linear(mask_shape[-1], self.hidden_dim),
        #     nn.ReLU(),
        # )
        # self.linear_encoder_sinr = nn.Sequential(
        #     nn.Linear(sinr_shape[0]*sinr_shape[-1], self.hidden_dim),
        #     nn.ReLU(),
        # )
        # # uncertainty_outdim = 1500
        # self.linear_encoder_uncertainty = nn.Sequential(
        #     nn.Linear(uncertainty_shape[0] * uncertainty_shape[-1], self.hidden_dim),
        #     nn.ReLU(),
        # )
        #
        # input_dim = int(
        #     6
        # )

        self.output = nn.Sequential(
            nn.Linear(ac_attr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )


    def forward(self, observations):
        ac_attr = self.output(observations["ac_attr"])
        batch_size = ac_attr.size(0)
        all_feature_output = self.output(ac_attr.reshape(batch_size,-1))

        return all_feature_output