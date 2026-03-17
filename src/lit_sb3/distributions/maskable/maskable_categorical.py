"""Modified PyTorch Categorical distribution with action masking support."""

import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

from lit_sb3.distributions.maskable.base import MaybeMasks


class MaskableCategorical(Categorical):
    """Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    Args:
        probs: Tensor containing finite non-negative values, which will be renormalized
            to sum to 1 along the last dimension.
        logits: Tensor of unnormalized log probabilities.
        validate_args: Whether or not to validate that arguments to methods like lob_prob()
            and icdf() match the distribution's shape, support, etc.
        masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set to a
            large negative value, resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
        masks: MaybeMasks = None,
    ):
        self.masks: torch.Tensor | None = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: MaybeMasks) -> None:
        """Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        Args:
            masks: An optional boolean ndarray of compatible shape with the distribution.
                If True, the corresponding choice's logit value is preserved. If False, it is set
                to a large negative value, resulting in near 0 probability. If masks is None, any
                previously applied masking is removed, and the original logits are restored.
        """
        if masks is not None:
            device = self.logits.device  # ty:ignore[unresolved-attribute]
            self.masks = torch.as_tensor(masks, dtype=torch.bool, device=device).reshape(self.logits.shape)  # ty:ignore[unresolved-attribute]
            huge_neg = torch.tensor(-1e8, dtype=self.logits.dtype, device=device)  # ty:ignore[unresolved-attribute]

            logits = torch.where(self.masks, self._original_logits, huge_neg)  # ty:ignore[no-matching-overload]
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)  # ty:ignore[invalid-argument-type]

        # self.probs may already be cached, so we must force an update
        self.probs = logits_to_probs(self.logits)  # ty:ignore[invalid-argument-type, invalid-assignment]

    def entropy(self) -> torch.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device  # ty:ignore[unresolved-attribute]
        p_log_p = self.logits * self.probs  # ty:ignore[unsupported-operator]
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0, device=device))
        return -p_log_p.sum(-1)
