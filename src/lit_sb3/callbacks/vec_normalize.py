"""SaveVecNormalizeCallback — periodically saves VecNormalize statistics."""

from __future__ import annotations

from pathlib import Path

from lit_sb3.callbacks.base import BaseCallback


class SaveVecNormalizeCallback(BaseCallback):
    """Callback for saving a :class:`~lit_sb3.vec_env.VecNormalize` wrapper periodically.

    Args:
        save_freq: Save every ``save_freq`` steps.
        save_path: Folder where the ``VecNormalize`` pickle is saved.
        name_prefix: If set, save timestamped files instead of overwriting a
            single ``vecnormalize.pkl``.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str | Path,
        name_prefix: str | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        assert self.model is not None

        if self.n_calls % self.save_freq == 0:
            save_dir = Path(self.save_path)
            if self.name_prefix is not None:
                path = save_dir / f"{self.name_prefix}_{self.num_timesteps}_steps.pkl"
            else:
                path = save_dir / "vecnormalize.pkl"

            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(path))
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True
