"""Abstract base class for probability distributions."""

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from torch import nn

SelfDistribution = TypeVar("SelfDistribution", bound="Distribution")


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self) -> None:
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> nn.Module | tuple[nn.Module, nn.Parameter]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes.
        """

    @abstractmethod
    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        """Set parameters of the distribution.

        Returns:
            self
        """

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the log likelihood.

        Args:
            x: the taken action

        Returns:
            The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> torch.Tensor | None:
        """Returns Shannon's entropy of the probability.

        Returns:
            the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Returns a sample from the probability distribution.

        Returns:
            the stochastic action
        """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """Returns the most likely action (deterministic output).

        from the probability distribution.

        Returns:
            the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Return actions according to the probability distribution.

        Args:
            deterministic: Whether to use deterministic actions.
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        """Returns samples from the probability distribution.

        given its parameters.

        Returns:
            actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns samples and the associated log probabilities.

        from the probability distribution given its parameters.

        Returns:
            actions and log prob
        """


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Continuous actions are usually considered to be independent,.

    so we can sum components of the ``log_prob`` or the entropy.

    Args:
        tensor: shape: (n_batch, n_actions) or (n_batch,)

    Returns:
        shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    tensor = tensor.sum(dim=1) if len(tensor.shape) > 1 else tensor.sum()
    return tensor
