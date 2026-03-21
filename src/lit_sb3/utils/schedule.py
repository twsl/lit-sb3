"""Learning rate and clip-range schedule helpers."""

import warnings

from lit_sb3.core.types import Schedule


class FloatSchedule:
    """Wrapper that ensures the output of a Schedule is cast to float.

    Can wrap either a constant value or an existing callable Schedule.

    Args:
        value_schedule: Constant value or callable schedule
            (e.g. LinearSchedule, ConstantSchedule)
    """

    def __init__(self, value_schedule: Schedule | float) -> None:
        if isinstance(value_schedule, FloatSchedule):
            self.value_schedule: Schedule = value_schedule.value_schedule
        elif isinstance(value_schedule, (float, int)):
            self.value_schedule = ConstantSchedule(float(value_schedule))
        else:
            assert callable(value_schedule), (
                f"The learning rate schedule must be a float or a callable, not {value_schedule}"
            )
            self.value_schedule = value_schedule

    def __call__(self, progress_remaining: float) -> float:
        # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
        # Some types are have odd behaviors when part of a Schedule, like numpy floats
        return float(self.value_schedule(progress_remaining))

    def __repr__(self) -> str:
        return f"FloatSchedule({self.value_schedule})"


class LinearSchedule:
    """LinearSchedule interpolates linearly between start and end.

    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``  where end is reached e.g 0.1
            then end is reached after 10% of the complete training process.
    """

    def __init__(self, start: float, end: float, end_fraction: float) -> None:
        self.start = start
        self.end = end
        self.end_fraction = end_fraction

    def __call__(self, progress_remaining: float) -> float:
        if (1 - progress_remaining) > self.end_fraction:
            return self.end
        else:
            return self.start + (1 - progress_remaining) * (self.end - self.start) / self.end_fraction

    def __repr__(self) -> str:
        return f"LinearSchedule(start={self.start}, end={self.end}, end_fraction={self.end_fraction})"


class ConstantSchedule:
    """Constant schedule that always returns the same value.

    Useful for fixed learning rates or clip ranges.

    Args:
        val: constant value
    """

    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, _: float) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"ConstantSchedule(val={self.val})"


class SimpleLinearSchedule:
    """Linear schedule from *initial_value* down to 0 as training progresses.

    Args:
        initial_value: Starting value (must be positive).
    """

    def __init__(self, initial_value: float | str) -> None:
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value: float | str) -> SimpleLinearSchedule:
    """Return a :class:`SimpleLinearSchedule` for *initial_value*."""
    return SimpleLinearSchedule(initial_value)


# ===== Deprecated schedule functions ====
# only kept for backward compatibility when unpickling old models, use FloatSchedule
# and other classes like `LinearSchedule() instead


def get_schedule_fn(value_schedule: Schedule | float) -> Schedule:
    """Transform (if needed) learning rate and clip range (for PPO).

    to callable.

    Args:
        value_schedule: Constant value of schedule function

    Returns:
        Schedule function (can return constant value)
    """
    warnings.warn("get_schedule_fn() is deprecated, please use FloatSchedule() instead", stacklevel=2)
    if isinstance(value_schedule, (float, int)):
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return lambda progress_remaining: float(value_schedule(progress_remaining))


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """Create a function that interpolates linearly between start and end.

    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.

    Returns:
        Linear schedule function.
    """
    warnings.warn("get_linear_fn() is deprecated, please use LinearSchedule() instead", stacklevel=2)

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float) -> Schedule:
    """Create a function that returns a constant.

    It is useful for learning rate schedule (to avoid code duplication).

    Args:
        val: constant value

    Returns:
        Constant schedule function.
    """
    warnings.warn("constant_fn() is deprecated, please use ConstantSchedule() instead", stacklevel=2)

    def func(_: float) -> float:
        return val

    return func


# ==== End of deprecated schedule functions ====
