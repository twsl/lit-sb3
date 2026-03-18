from collections.abc import Callable
from typing import Any

from .base import BaseCallback


class ConvertCallback(BaseCallback):
    """Convert functional callback (old-style) to object.

    Args:
        callback:
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Callable[[dict[str, Any], dict[str, Any]], bool] | None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True
