"""MultiCategorical distribution for multi-discrete action spaces."""

from typing import TypeVar

import torch
from torch import nn
from torch.distributions import Categorical

from lit_sb3.distributions.base import Distribution

SelfMultiCategoricalDistribution = TypeVar("SelfMultiCategoricalDistribution", bound="MultiCategoricalDistribution")


class MultiCategoricalDistribution(Distribution):
    """MultiCategorical distribution for multi discrete actions.

    Args:
        action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: list[int]):
        super().__init__()
        self.action_dims = action_dims
        self.distribution: list[Categorical] = []

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layer that represents the distribution.

        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        Args:
            latent_dim: Dimension of the last layer
                of the policy network (before the action layer)
        """
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(
        self: SelfMultiCategoricalDistribution, action_logits: torch.Tensor
    ) -> SelfMultiCategoricalDistribution:
        self.distribution = [
            Categorical(logits=split) for split in torch.split(action_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self.distribution, torch.unbind(actions, dim=1), strict=False)
            ],
            dim=1,
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)  # ty:ignore[invalid-argument-type]

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
