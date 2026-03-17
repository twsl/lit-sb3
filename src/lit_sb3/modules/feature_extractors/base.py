import gymnasium as gym
from torch import nn


class BaseFeaturesExtractor(nn.Module):
    """Base class that represents a features extractor.

    Args:
        observation_space: The observation space of the environment
        features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        """The number of features that the extractor outputs."""
        return self._features_dim
