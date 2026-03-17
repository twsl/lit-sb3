"""Policies: abstract base class and concrete implementations."""

from abc import ABC, abstractmethod
import collections
import copy
from functools import partial
from typing import Any, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from lit_sb3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from lit_sb3.common.types import PyTorchObs, Schedule
from lit_sb3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from lit_sb3.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from lit_sb3.modules import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from lit_sb3.policies.actor_critic import ActorCriticPolicy
from lit_sb3.policies.base import BaseModel, BasePolicy

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


class ContinuousCritic(BaseModel):
    """Critic network(s) for DDPG/SAC/TD3.

    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    Args:
        observation_space: Observation space
        action_space: Action space
        net_arch: Network architecture
        features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        features_dim: Number of features
        activation_fn: Activation function
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        n_critics: Number of critic networks to create.
        share_features_extractor: Whether the features extractor is shared or not
            between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Only predict the Q-value using the first network.

        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))
