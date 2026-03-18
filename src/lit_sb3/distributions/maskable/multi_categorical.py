"""Maskable multi-categorical distribution for multi-discrete action spaces."""

from typing import TypeVar

import torch
from torch import nn

from lit_sb3.distributions.maskable.base import MaskableDistribution, MaybeMasks
from lit_sb3.distributions.maskable.maskable_categorical import MaskableCategorical

SelfMaskableMultiCategoricalDistribution = TypeVar(
    "SelfMaskableMultiCategoricalDistribution", bound="MaskableMultiCategoricalDistribution"
)


class MaskableMultiCategoricalDistribution(MaskableDistribution):
    """MultiCategorical distribution for multi discrete actions. Supports invalid action masking.

    Args:
        action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: list[int]) -> None:
        super().__init__()
        self.distributions: list[MaskableCategorical] = []
        self.action_dims = action_dims

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
        self: SelfMaskableMultiCategoricalDistribution, action_logits: torch.Tensor
    ) -> SelfMaskableMultiCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))

        self.distributions = [
            MaskableCategorical(logits=split) for split in torch.split(reshaped_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        assert len(self.distributions) > 0, "Must set distribution parameters"

        # Restructure shape to align with each categorical
        actions = actions.view(-1, len(self.action_dims))

        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self.distributions, torch.unbind(actions, dim=1), strict=False)
            ],
            dim=1,
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return torch.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return torch.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> torch.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)  # ty:ignore[invalid-argument-type]

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: MaybeMasks) -> None:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        split_masks = [None] * len(self.distributions)
        if masks is not None:
            masks_tensor = torch.as_tensor(masks)
            # Restructure shape to align with logits
            masks_tensor = masks_tensor.view(-1, sum(self.action_dims))
            # Then split columnwise for each discrete action
            split_masks = torch.split(masks_tensor, list(self.action_dims), dim=1)

        for distribution, mask in zip(self.distributions, split_masks, strict=False):
            distribution.apply_masking(mask)
