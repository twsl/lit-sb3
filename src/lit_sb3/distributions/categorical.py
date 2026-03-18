"""Categorical distribution for discrete action spaces."""

from typing import TypeVar

import torch
from torch import nn
from torch.distributions import Categorical

from lit_sb3.distributions.base import Distribution

SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")


class CategoricalDistribution(Distribution):
    """Categorical distribution for discrete actions.

    Args:
        action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layer that represents the distribution.

        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        Args:
            latent_dim: Dimension of the last layer
                of the policy network (before the action layer)
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self: SelfCategoricalDistribution, action_logits: torch.Tensor
    ) -> SelfCategoricalDistribution:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        return self.distribution.log_prob(actions)  # ty:ignore[unresolved-attribute]

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()  # ty:ignore[unresolved-attribute]

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()  # ty:ignore[unresolved-attribute]

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)  # ty:ignore[invalid-argument-type, unresolved-attribute]

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
