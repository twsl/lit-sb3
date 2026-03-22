"""RawStatisticsCallback — logs raw per-episode data through SummaryWriter."""

from __future__ import annotations

from lit_sb3.callbacks.base import BaseCallback


class RawStatisticsCallback(BaseCallback):
    """Callback for logging raw per-episode returns and lengths to TensorBoard.

    Requires TensorBoard logging to be active.  The callback bypasses
    Lightning's default log-throttling so that every episode is recorded.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._timesteps_counter = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            writer = self.summary_writer
            if "episode" in info and writer is not None:
                self._timesteps_counter += info["episode"]["l"]
                writer.add_scalar("raw/rollouts/episodic_return", info["episode"]["r"], self._timesteps_counter)
                writer.add_scalar("raw/rollouts/episodic_length", info["episode"]["l"], self._timesteps_counter)
                writer.flush()
        return True
