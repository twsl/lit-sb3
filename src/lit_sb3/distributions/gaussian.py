"""Diagonal Gaussian distribution for continuous actions."""

from typing import TypeVar

import torch
from torch import nn
from torch.distributions import Normal

from lit_sb3.distributions.base import Distribution, sum_independent_dims

SelfDiagGaussianDistribution = TypeVar("SelfDiagGaussianDistribution", bound="DiagGaussianDistribution")


class DiagGaussianDistribution(Distribution):
    """Gaussian distribution with diagonal covariance matrix, for continuous actions.

    Args:
        action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        """Create the layers and parameter that represent the distribution.

        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values).

        Args:
            latent_dim: Dimension of the last layer of the policy (before the action layer)
            log_std_init: Initial value for the log standard deviation
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(
        self: SelfDiagGaussianDistribution, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> SelfDiagGaussianDistribution:
        """Create the distribution given its parameters (mean, std).

        Args:
            mean_actions: Mean of the action distribution.
            log_std: Log standard deviation of the action distribution.
        """
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        """Get the log probabilities of actions according to the distribution.

        Note that you must first call the ``proba_distribution()`` method.

        Args:
            actions: Actions taken.
        """
        log_prob = self.distribution.log_prob(actions)  # ty:ignore[unresolved-attribute]
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor | None:
        return sum_independent_dims(self.distribution.entropy())  # ty:ignore[unresolved-attribute]

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()  # ty:ignore[unresolved-attribute]

    def mode(self) -> torch.Tensor:
        return self.distribution.mean  # ty:ignore[unresolved-attribute]

    def actions_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the log probability of taking an action.

        given the distribution parameters.

        Args:
            mean_actions: Mean of the action distribution.
            log_std: Log standard deviation of the action distribution.
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
