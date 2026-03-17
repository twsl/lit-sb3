from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import lightning as L  # noqa: N812

from lit_sb3.utils.logger import Logger
from lit_sb3.vec_env import VecEnv

if TYPE_CHECKING:
    from lit_sb3.algorithms.base import BaseAlgorithm


class BaseCallback(L.Callback, ABC):
    """Base class for callback.

    Inherits from ``lightning.Callback`` so instances can be passed directly
    to a ``lightning.Trainer`` via its ``callbacks`` argument.  When training
    runs through :meth:`BaseAlgorithm.learn` the SB3 lifecycle methods
    (``on_training_start`` / ``on_training_end``) are called manually before
    ``Trainer.fit``; the Lightning shims below detect that and skip to avoid
    double-calls.  When a callback is passed directly to a ``Trainer`` without
    going through ``learn``, the shims invoke the SB3 lifecycle instead.

    Args:
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    model: "BaseAlgorithm"

    def __init__(self, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]
        # Guard flags to prevent double-calls when used via learn()
        self._training_started = False
        self._training_ended = False

    @property
    def training_env(self) -> VecEnv:
        training_env = self.model.get_env()
        assert training_env is not None, (
            "`model.get_env()` returned None, you must initialize the model with an environment to use callbacks"
        )
        return training_env

    @property
    def logger(self) -> Logger:
        return self.model.logger

    # Type hint as string to avoid circular import
    def init_callback(self, model: "BaseAlgorithm") -> None:
        """Initialize the callback by saving references to the.

        RL model and the training environment for convenience.
        """
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:  # noqa: B027
        pass

    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        self._training_started = True
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:  # noqa: B027
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """Returns.

        If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        Returns:
            If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._training_ended = True
        self._on_training_end()

    def _on_training_end(self) -> None:  # noqa: B027
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:  # noqa: B027
        pass

    def update_locals(self, locals_: dict[str, Any]) -> None:
        """Update the references to the local variables.

        Args:
            locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: dict[str, Any]) -> None:  # noqa: B027
        """Update the references to the local variables on sub callbacks.

        Args:
            locals_: the local variables during rollout collection
        """
        pass

    # ------------------------------------------------------------------
    # Lightning shims — bridge fit-level hooks to SB3 lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Lightning hook: called at the very beginning of ``Trainer.fit``.

        When invoked through :meth:`BaseAlgorithm.learn`, ``on_training_start``
        has already been called manually, so this is a no-op.  When a callback
        is passed directly to ``Trainer(callbacks=[cb])`` this initialises the
        SB3 side of the callback instead.
        """
        if self._training_started:
            return
        self.init_callback(pl_module)  # type: ignore[arg-type]
        self.on_training_start(
            {"total_timesteps": getattr(pl_module, "_total_timesteps", 0)},
            {},
        )

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Lightning hook: called at the very end of ``Trainer.fit``.

        When invoked through :meth:`BaseAlgorithm.learn`, ``on_training_end``
        has already been called by ``LightningModule.on_train_end``, so this
        is a no-op.  When a callback is passed directly to ``Trainer`` this
        tears down the SB3 side instead.
        """
        if self._training_ended:
            return
        self.on_training_end()


class EventCallback(BaseCallback):
    """Base class for triggering callback on event.

    Args:
        callback: Callback that will be called
            when an event is triggered.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: "BaseCallback | None" = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            assert self.callback is not None
            self.callback.parent = self

    def init_callback(self, model: "BaseAlgorithm") -> None:
        super().init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """Update the references to the local variables.

        Args:
            locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """Class for chaining callbacks.

    Args:
        callbacks: A list of callbacks that will be called
            sequentially.
    """

    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

            # Fix for https://github.com/DLR-RM/stable-baselines3/issues/1791
            # pass through the parent callback to all children
            callback.parent = self.parent

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """Update the references to the local variables.

        Args:
            locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)
