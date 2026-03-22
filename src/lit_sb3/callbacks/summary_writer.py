"""Callback exposing TensorBoard SummaryWriter from Lightning TensorBoardLogger."""

from __future__ import annotations

from typing import Any

import lightning as L  # noqa: N812
from lightning.pytorch.loggers import TensorBoardLogger


class SummaryWriterCallback(L.Callback):
    """Resolve and expose the active TensorBoard SummaryWriter."""

    def __init__(self) -> None:
        super().__init__()
        self.writer: Any = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.writer = self._resolve_writer(trainer)

    def on_validate_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.writer is None:
            self.writer = self._resolve_writer(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.writer is None:
            self.writer = self._resolve_writer(trainer)

    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.writer is None:
            self.writer = self._resolve_writer(trainer)

    @staticmethod
    def _resolve_writer(trainer: L.Trainer) -> Any:
        for logger in getattr(trainer, "loggers", []):
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        return None

    @classmethod
    def get_writer(cls, trainer: L.Trainer) -> Any:
        for callback in trainer.callbacks:
            if isinstance(callback, cls):
                return callback.writer
        return None
