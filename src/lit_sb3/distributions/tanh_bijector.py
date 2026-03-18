"""Tanh bijective transformation."""

import torch


class TanhBijector:
    """Bijective transformation of a probability distribution.

    using a squashing function (tanh).

    Args:
        epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """Inverse of Tanh.

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """Inverse tanh.

        Args:
            y: The output value.
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)
