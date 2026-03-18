import gymnasium as gym
import torch
from torch import nn

from lit_sb3.core.preprocessing import get_flattened_obs_dim
from lit_sb3.modules.feature_extractors.base import BaseFeaturesExtractor


class FlattenExtractor(BaseFeaturesExtractor):
    """Feature extract that flatten the input.

    Used as a placeholder when feature extraction is not needed.

    Args:
        observation_space: The observation space of the environment
    """

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)
