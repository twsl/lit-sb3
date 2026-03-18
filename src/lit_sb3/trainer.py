"""RLTrainer — a ``lightning.Trainer`` subclass for reinforcement learning.

Usage::

    from lit_sb3 import PPO, RLTrainer

    model = PPO("MlpPolicy", "CartPole-v1")
    trainer = RLTrainer(max_steps=100_000, enable_progress_bar=True)
    trainer.fit(model)

The SB3-compatible ``model.learn(total_timesteps)`` API still works and simply
delegates to ``RLTrainer.fit`` internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L  # noqa: N812

from lit_sb3.core.types import MaybeCallback

if TYPE_CHECKING:
    from lit_sb3.algorithms.base import BaseAlgorithm


class RLTrainer(L.Trainer):
    """A :class:`lightning.Trainer` subclass that drives reinforcement-learning runs.

    All standard ``lightning.Trainer`` keyword arguments are accepted and
    forwarded to the parent class, giving full access to Lightning features
    (mixed precision, custom loggers, DDP, plugins, …).

    The RL-specific arguments control the training schedule and SB3 callbacks.
    ``accelerator``, ``enable_model_summary``, ``enable_checkpointing``, and
    ``logger`` have RL-appropriate defaults but can be overridden.

    Args:
        max_steps: Total environment steps to collect (RL timesteps budget).
            Passed as ``max_steps=-1`` to the parent ``L.Trainer`` — epoch-based
            stopping is used internally, computed from the algorithm.
        rl_callback: SB3-style callback(s) fired at each RL step.
        tb_log_name: Tag used in the TensorBoard run name.
        reset_num_timesteps: If ``True``, reset the step counter before training.
        log_every_n_steps: How often (in rollout cycles / episodes) to emit SB3
            stats *and* how often Lightning logs metrics within a step.
            ``None`` uses the algorithm's default for SB3 stats while Lightning
            falls back to its own default (50 steps).
        **trainer_kwargs: Forwarded verbatim to :class:`lightning.Trainer`.
    """

    def __init__(
        self,
        max_steps: int,
        rl_callback: MaybeCallback = None,
        log_every_n_steps: int | None = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        # Lightning Trainer kwargs with RL-appropriate defaults
        accelerator: str = "auto",
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        enable_checkpointing: bool = False,
        logger: Any = False,
        **trainer_kwargs: Any,
    ) -> None:
        self._rl_total_timesteps = max_steps
        self._rl_callback = rl_callback
        self._rl_log_every_n_steps = log_every_n_steps
        self._rl_tb_log_name = tb_log_name
        self._rl_reset_num_timesteps = reset_num_timesteps
        self._rl_enable_progress_bar = enable_progress_bar

        super().__init__(
            accelerator=accelerator,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            enable_checkpointing=enable_checkpointing,
            # Disable Lightning's step-based stopping — we use epoch-based control.
            max_steps=-1,
            # Pass through to Lightning's own metric-logging frequency.
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            **trainer_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        model: BaseAlgorithm,
        *args: Any,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        """Train *model* for :attr:`_rl_total_timesteps` environment steps.

        Steps:
        1. ``model._setup_learn`` — reset counters, env, logger, wrap callbacks.
        2. ``model._prepare_fit`` — inject per-algorithm state.
        3. ``model._rl_max_epochs`` — compute ``max_epochs`` for Lightning.
        4. ``model._pre_fit_hook`` — algorithm-specific work before ``super().fit``
           (e.g. on-policy pre-collects the first rollout).
        5. ``super().fit(model)`` — standard Lightning training loop.

        Args:
            model: The RL algorithm (a ``LightningModule`` subclass).
            *args: Additional positional arguments forwarded to
                :meth:`lightning.Trainer.fit`.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`lightning.Trainer.fit`.

        Returns:
            The trained model (same object, for chaining / SB3 compatibility).
        """
        total_timesteps, rl_callback = model._setup_learn(
            self._rl_total_timesteps,
            self._rl_callback,
            self._rl_reset_num_timesteps,
            self._rl_tb_log_name,
            self._rl_enable_progress_bar,
        )

        log_every_n_steps = (
            self._rl_log_every_n_steps if self._rl_log_every_n_steps is not None else model._default_log_every_n_steps
        )

        model._prepare_fit(rl_callback, log_every_n_steps)

        # Patch max_epochs so Lightning runs the right number of rollout cycles.
        self.fit_loop.max_epochs = model._rl_max_epochs(total_timesteps)

        # Add the RL callback to the trainer's callback list.
        if rl_callback not in self.callbacks:  # ty:ignore[unresolved-attribute]
            self.callbacks.append(rl_callback)  # ty:ignore[unresolved-attribute]

        # Algorithm-specific pre-fit work (on-policy: pre-collect first rollout).
        rl_callback.on_training_start(locals(), globals())
        model._pre_fit_hook(rl_callback)

        super().fit(model, *args, **kwargs)
        # on_train_end() in the model calls rl_callback.on_training_end()
        return model
