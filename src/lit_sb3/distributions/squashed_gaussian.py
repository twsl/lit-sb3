"""Squashed (tanh) diagonal Gaussian distribution."""

from typing import TypeVar

import torch

from lit_sb3.distributions.gaussian import DiagGaussianDistribution
from lit_sb3.distributions.tanh_bijector import TanhBijector

SelfSquashedDiagGaussianDistribution = TypeVar(
    "SelfSquashedDiagGaussianDistribution", bound="SquashedDiagGaussianDistribution"
)


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    Args:
        action_dim: Dimension of the action space.
        epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions: torch.Tensor | None = None

    def proba_distribution(
        self: SelfSquashedDiagGaussianDistribution, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: torch.Tensor, gaussian_actions: torch.Tensor | None = None) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor | None:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob
