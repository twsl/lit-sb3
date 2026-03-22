"""RLEnvDataModule — a ``LightningDataModule`` that provides environments instead of dataloaders.

In supervised learning, a ``LightningDataModule`` manages datasets and
dataloaders for each stage (fit, validate, test, predict).  In reinforcement
learning the "data" *is* the environment: the agent generates its own training
data by interacting with the environment.

``RLEnvDataModule`` follows the same stage-based pattern but exposes
**environments** rather than dataloaders::

    dm = RLEnvDataModule(
        train_env="CartPole-v1",
        val_env="CartPole-v1",
    )
    trainer = RLTrainer(max_timesteps=100_000)
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import lightning as L  # noqa: N812

from lit_sb3.core.types import GymEnv

if TYPE_CHECKING:
    from lit_sb3.vec_env import VecEnv


class RLEnvDataModule(L.LightningDataModule):
    """A :class:`~lightning.LightningDataModule` that holds environments for each RL stage.

    Instead of :meth:`train_dataloader`, :meth:`val_dataloader`, etc., this
    module exposes :attr:`train_env`, :attr:`val_env`, :attr:`test_env`, and
    :attr:`predict_env`.  The :class:`~lit_sb3.trainer.RLTrainer` extracts the
    appropriate environment from the datamodule at each stage.

    Args:
        train_env: Environment (or env-id string) used for training.
        val_env: Environment (or env-id string) used for validation / evaluation.
        test_env: Environment (or env-id string) used for testing / evaluation.
        predict_env: Environment (or env-id string) used for prediction.
        monitor_wrapper: Whether to wrap raw Gym envs in a ``Monitor`` wrapper.
        verbose: Verbosity level forwarded to env wrapping utilities.
    """

    _ENV_ATTRS: ClassVar[dict[str, str]] = {
        "fit": "train_env",
        "validate": "val_env",
        "test": "test_env",
        "predict": "predict_env",
    }
    _ENV_SPEC_ATTRS: ClassVar[dict[str, str]] = {
        "fit": "_train_env_spec",
        "validate": "_val_env_spec",
        "test": "_test_env_spec",
        "predict": "_predict_env_spec",
    }

    def __init__(
        self,
        train_env: GymEnv | str | None = None,
        val_env: GymEnv | str | None = None,
        test_env: GymEnv | str | None = None,
        predict_env: GymEnv | str | None = None,
        monitor_wrapper: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        # Raw specs (string ids or unwrapped envs) — resolved lazily in setup()
        self._train_env_spec = train_env
        self._val_env_spec = val_env
        self._test_env_spec = test_env
        self._predict_env_spec = predict_env
        self._monitor_wrapper = monitor_wrapper
        self._verbose = verbose

        # Resolved (wrapped) VecEnv instances — populated by setup()
        self.train_env: VecEnv | None = None
        self.val_env: VecEnv | None = None
        self.test_env: VecEnv | None = None
        self.predict_env: VecEnv | None = None

    # ------------------------------------------------------------------
    # LightningDataModule lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Resolve and wrap environments for the requested *stage*.

        Called automatically by Lightning before each stage, or can be
        called manually.
        """
        if stage in ("fit", None) and self._train_env_spec is not None and self.train_env is None:
            self.train_env = self._resolve(self._train_env_spec)
        if stage in ("fit", "validate", None) and self._val_env_spec is not None and self.val_env is None:
            self.val_env = self._resolve(self._val_env_spec)
        if stage in ("test", None) and self._test_env_spec is not None and self.test_env is None:
            self.test_env = self._resolve(self._test_env_spec)
        if stage in ("predict", None) and self._predict_env_spec is not None and self.predict_env is None:
            self.predict_env = self._resolve(self._predict_env_spec)

    def teardown(self, stage: str | None = None) -> None:
        """Release resolved environments for the requested *stage*."""
        if stage is None:
            self.close()
            return

        for attr in self._attrs_for_teardown(stage):
            env = getattr(self, attr, None)
            if env is not None:
                env.close()
                setattr(self, attr, None)

    def on_exception(self, exception: BaseException) -> None:
        """Ensure vectorized envs are closed when Lightning aborts the run."""
        del exception
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_env(self, stage: Literal["fit", "validate", "test", "predict"], *, resolve: bool = True) -> VecEnv | None:
        """Return the environment configured for *stage*.

        Args:
            stage: Lightning stage name.
            resolve: Whether to call :meth:`setup` before reading the stage env.
        """
        self._validate_stage(stage)
        if resolve:
            self.setup(stage)
        return cast("VecEnv | None", getattr(self, self._ENV_ATTRS[stage]))

    def train_dataloader(self) -> None:
        """RLEnvDataModule exposes envs, not training dataloaders."""
        raise TypeError(
            "RLEnvDataModule does not define train_dataloader(); use train_env via RLTrainer.fit(..., datamodule=...)."
        )

    def val_dataloader(self) -> None:
        """RLEnvDataModule exposes envs, not validation dataloaders."""
        raise TypeError(
            "RLEnvDataModule does not define val_dataloader(); use val_env via RLTrainer.validate(..., datamodule=...)."
        )

    def test_dataloader(self) -> None:
        """RLEnvDataModule exposes envs, not test dataloaders."""
        raise TypeError(
            "RLEnvDataModule does not define test_dataloader(); use test_env via RLTrainer.test(..., datamodule=...)."
        )

    def predict_dataloader(self) -> None:
        """RLEnvDataModule exposes envs, not predict dataloaders."""
        raise TypeError(
            "RLEnvDataModule does not define predict_dataloader(); use predict_env via RLTrainer.predict(..., datamodule=...)."
        )

    def state_dict(self) -> dict[str, Any]:
        """Persist serializable datamodule config and env specs, never live env instances."""
        return {
            "train_env_spec": self._serialize_env_spec(self._train_env_spec),
            "val_env_spec": self._serialize_env_spec(self._val_env_spec),
            "test_env_spec": self._serialize_env_spec(self._test_env_spec),
            "predict_env_spec": self._serialize_env_spec(self._predict_env_spec),
            "monitor_wrapper": self._monitor_wrapper,
            "verbose": self._verbose,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore serializable env specs and datamodule config from checkpoint state."""
        self.close()
        self._train_env_spec = self._deserialize_env_spec(state_dict.get("train_env_spec"))
        self._val_env_spec = self._deserialize_env_spec(state_dict.get("val_env_spec"))
        self._test_env_spec = self._deserialize_env_spec(state_dict.get("test_env_spec"))
        self._predict_env_spec = self._deserialize_env_spec(state_dict.get("predict_env_spec"))
        self._monitor_wrapper = bool(state_dict.get("monitor_wrapper", self._monitor_wrapper))
        self._verbose = int(state_dict.get("verbose", self._verbose))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Any,
        map_location: Any = None,
        hparams_file: Any = None,
        weights_only: bool | None = None,
        **kwargs: Any,
    ) -> RLEnvDataModule:
        """Load the datamodule and then restore env specs through :meth:`load_state_dict`."""
        loaded = super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            weights_only=weights_only,
            **kwargs,
        )
        return cast(RLEnvDataModule, loaded)

    def _resolve(self, env_spec: GymEnv | str) -> VecEnv:
        """Turn an env spec (string id or raw Gym env) into a wrapped ``VecEnv``."""
        from lit_sb3.algorithms.base import BaseAlgorithm, maybe_make_env
        from lit_sb3.vec_env import VecEnv as _VecEnv

        env: Any = maybe_make_env(env_spec, self._verbose) if isinstance(env_spec, str) else env_spec
        if not isinstance(env, _VecEnv):
            env = BaseAlgorithm._wrap_env(env, self._verbose, self._monitor_wrapper)
        return env

    def close(self) -> None:
        """Close all resolved environments and free resources."""
        for attr in ("train_env", "val_env", "test_env", "predict_env"):
            env = getattr(self, attr, None)
            if env is not None:
                env.close()
                setattr(self, attr, None)

    def __str__(self) -> str:
        """Return an env-oriented summary instead of Lightning's dataset summary."""
        lines = []
        for stage in ("fit", "validate", "test", "predict"):
            spec = getattr(self, self._ENV_SPEC_ATTRS[stage])
            env = getattr(self, self._ENV_ATTRS[stage], None)
            spec_text = self._format_env_spec(spec)
            status = "resolved" if env is not None else "unresolved"
            env_type = type(env).__name__ if env is not None else "None"
            lines.append(f"{{{stage}: spec={spec_text}, status={status}, env={env_type}}}")
        return "\n".join(lines)

    @classmethod
    def _validate_stage(cls, stage: str) -> None:
        if stage not in cls._ENV_ATTRS:
            supported = ", ".join(cls._ENV_ATTRS)
            raise ValueError(f"Unsupported stage {stage!r}. Expected one of: {supported}.")

    @classmethod
    def _attrs_for_teardown(cls, stage: str) -> tuple[str, ...]:
        cls._validate_stage(stage)
        if stage == "fit":
            return ("train_env", "val_env")
        return (cls._ENV_ATTRS[stage],)

    @staticmethod
    def _serialize_env_spec(env_spec: GymEnv | str | None) -> dict[str, Any]:
        if env_spec is None:
            return {"kind": "none", "value": None}
        if isinstance(env_spec, str):
            return {"kind": "string", "value": env_spec}
        return {"kind": "unserializable", "value": None}

    @staticmethod
    def _deserialize_env_spec(payload: Any) -> GymEnv | str | None:
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        if kind == "string":
            value = payload.get("value")
            return value if isinstance(value, str) else None
        return None

    @staticmethod
    def _format_env_spec(env_spec: GymEnv | str | None) -> str:
        if env_spec is None:
            return "None"
        if isinstance(env_spec, str):
            return env_spec
        return type(env_spec).__name__
