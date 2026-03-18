"""RawStatisticsCallback — logs raw per-episode data to TensorBoard."""

from __future__ import annotations

from typing import Any

from lit_sb3.callbacks.base import BaseCallback


class RawStatisticsCallback(BaseCallback):
    """Callback for logging raw per-episode returns and lengths to TensorBoard.

    Requires TensorBoard logging to be active.  The callback bypasses
    Lightning's default log-throttling so that every episode is recorded.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._timesteps_counter = 0
        self._tensorboard_writer: Any = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        try:
            from lit_sb3.utils.logger import TensorBoardOutputFormat

            for out_format in self.logger.output_formats:
                if isinstance(out_format, TensorBoardOutputFormat):
                    self._tensorboard_writer = out_format
        except ImportError:
            pass

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and self._tensorboard_writer is not None:
                logger_dict = {
                    "raw/rollouts/episodic_return": info["episode"]["r"],
                    "raw/rollouts/episodic_length": info["episode"]["l"],
                }
                exclude_dict = dict.fromkeys(logger_dict)
                self._timesteps_counter += info["episode"]["l"]
                self._tensorboard_writer.write(logger_dict, exclude_dict, self._timesteps_counter)
        return True
