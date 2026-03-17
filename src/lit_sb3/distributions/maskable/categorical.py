"""Maskable categorical distribution for discrete action spaces."""

from typing import TypeVar

import torch
from torch import nn

from lit_sb3.distributions.maskable.base import MaskableDistribution, MaybeMasks
from lit_sb3.distributions.maskable.maskable_categorical import MaskableCategorical

SelfMaskableCategoricalDistribution = TypeVar(
    "SelfMaskableCategoricalDistribution", bound="MaskableCategoricalDistribution"
)


class MaskableCategoricalDistribution(MaskableDistribution):
    """Categorical distribution for discrete actions. Supports invalid action masking.

    Args:
        action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: MaskableCategorical | None = None
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
        self: SelfMaskableCategoricalDistribution, action_logits: torch.Tensor
    ) -> SelfMaskableCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return torch.argmax(self.distribution.probs, dim=1)  # ty:ignore[invalid-argument-type]

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: MaybeMasks) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        self.distribution.apply_masking(masks)
