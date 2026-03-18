"""Bernoulli distribution for MultiBinary action spaces."""

from typing import TypeVar

import torch
from torch import nn
from torch.distributions import Bernoulli

from lit_sb3.distributions.base import Distribution

SelfBernoulliDistribution = TypeVar("SelfBernoulliDistribution", bound="BernoulliDistribution")


class BernoulliDistribution(Distribution):
    """Bernoulli distribution for MultiBinary action spaces.

    Args:
        action_dims: Number of binary actions
    """

    def __init__(self, action_dims: int) -> None:
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layer that represents the distribution.

        it will be the logits of the Bernoulli distribution.

        Args:
            latent_dim: Dimension of the last layer
                of the policy network (before the action layer)
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self: SelfBernoulliDistribution, action_logits: torch.Tensor) -> SelfBernoulliDistribution:
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        return self.distribution.log_prob(actions).sum(dim=1)  # ty:ignore[unresolved-attribute]

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=1)  # ty:ignore[unresolved-attribute]

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()  # ty:ignore[unresolved-attribute]

    def mode(self) -> torch.Tensor:
        return torch.round(self.distribution.probs)  # ty:ignore[invalid-argument-type, unresolved-attribute]

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
