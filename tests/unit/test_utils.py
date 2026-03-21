"""Unit tests for utility functions.

Adapted from HenryJia/lightning-baselines3 tests/common/test_utils.py.
"""

from gymnasium import spaces
import pytest

from lit_sb3.core.preprocessing import get_action_dim, get_obs_shape
from lit_sb3.utils.tensors import zip_strict


def test_get_action_dim():
    s1 = spaces.Box(shape=[3, 4], low=0, high=1)
    s2 = spaces.Discrete(5)
    s3 = spaces.MultiDiscrete([5, 2, 2])
    s4 = spaces.MultiBinary(7)

    assert get_action_dim(s1) == 12
    assert get_action_dim(s2) == 1
    assert get_action_dim(s3) == 3
    assert get_action_dim(s4) == 7


def test_get_obs_shape():
    s1 = spaces.Box(shape=[3, 4], low=0, high=1)
    s2 = spaces.Discrete(5)
    s3 = spaces.MultiDiscrete([5, 2, 2])
    s4 = spaces.MultiBinary(7)

    assert get_obs_shape(s1) == (3, 4)
    assert get_obs_shape(s2) == (1,)
    assert get_obs_shape(s3) == (3,)
    assert get_obs_shape(s4) == (7,)


def test_zip_strict():
    """Test zip_strict raises ValueError when iterables have different lengths."""
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9, 0]

    for _ in zip_strict(a, b):
        pass

    with pytest.raises(ValueError):
        for _ in zip_strict(b, c):
            pass
