from typing import Any

from .base import EventCallback
from .convert import ConvertCallback


class EveryNTimesteps(EventCallback):
    """Trigger a callback every ``n_steps`` timesteps.

    Args:
        n_steps: Number of timesteps between two trigger.
        callback: Callback that will be called
            when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: "EventCallback"):
        super().__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class LogEveryNTimesteps(EveryNTimesteps):
    """Log data every ``n_steps`` timesteps.

    Args:
        n_steps: Number of timesteps between two trigger.
    """

    def __init__(self, n_steps: int):
        super().__init__(n_steps, callback=ConvertCallback(self._log_data))

    def _log_data(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
        self.model.dump_logs()
        return True
