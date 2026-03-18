from typing import Any

from gymnasium import spaces
import torch
from torch import nn

from lit_sb3.core.types import Schedule
from lit_sb3.modules import BaseFeaturesExtractor, NatureCNN
from lit_sb3.policies.maskable.actor_critic import MaskableActorCriticPolicy


class MaskableActorCriticCnnPolicy(MaskableActorCriticPolicy):
    """CNN policy class for actor-critic algorithms (has both policy and value prediction).

    Used by A2C, PPO and the likes.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        net_arch: The specification of the policy and value networks.
        activation_fn: Activation function
        ortho_init: Whether to use or not orthogonal initialization
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
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
