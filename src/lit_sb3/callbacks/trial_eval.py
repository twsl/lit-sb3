"""TrialEvalCallback for Optuna hyperparameter optimisation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lit_sb3.callbacks.eval import EvalCallback
from lit_sb3.vec_env.base_vec_env import VecEnv


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting an Optuna trial.

    Evaluates the model every ``eval_freq`` steps and reports the mean reward
    to the Optuna trial.  If the trial should be pruned, training is stopped
    early.

    Args:
        eval_env: Environment used for evaluation.
        trial: Optuna trial instance.
        n_eval_episodes: Number of episodes per evaluation.
        eval_freq: Evaluate every ``eval_freq`` steps.
        deterministic: Use deterministic actions during evaluation.
        verbose: Verbosity level.
        best_model_save_path: Where to save the best model.
        log_path: Where to save evaluation logs.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: Any,  # optuna.Trial
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
