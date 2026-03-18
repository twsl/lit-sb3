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

from lit_sb3.core.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from lit_sb3.core.types import PyTorchObs, Schedule
from lit_sb3.core.utils import get_device, is_vectorized_observation, obs_as_tensor
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


class ActorCriticCnnPolicy(ActorCriticPolicy):
    """CNN policy class for actor-critic algorithms (has both policy and value prediction).

    Used by A2C, PPO and the likes.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        net_arch: The specification of the policy and value networks.
        activation_fn: Activation function
        ortho_init: Whether to use or not orthogonal initialization
        use_sde: Whether to use State Dependent Exploration or not
        log_std_init: Initial value for the log standard deviation
        full_std: Whether to use (n_features x n_actions) parameters
            for the std instead of only (n_features,) when using gSDE
        use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        squash_output: Whether to squash the output using a tanh function,
            this allows to ensure boundaries when using gSDE.
        features_extractor_class: Features extractor to use.
        features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        share_features_extractor: If True, the features extractor is shared between the policy and value networks.
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
