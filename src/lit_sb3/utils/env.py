"""Environment creation and wrapping utilities."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import gymnasium as gym

from lit_sb3.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize
from lit_sb3.vec_env.patch_gym import _patch_env
from lit_sb3.wrappers.atari.atari_wrapper import AtariWrapper
from lit_sb3.wrappers.monitor import Monitor


def unwrap_wrapper(env: gym.Env, wrapper_class: type[gym.Wrapper]) -> gym.Wrapper | None:
    """Retrieve a ``VecEnvWrapper`` object by recursively searching.

    Args:
        env: Environment to unwrap
        wrapper_class: Wrapper to look for

    Returns:
        Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: gym.Env, wrapper_class: type[gym.Wrapper]) -> bool:
    """Check if a given environment has been wrapped with a given wrapper.

    Args:
        env: Environment to check
        wrapper_class: Wrapper class to look for

    Returns:
        True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_vec_env(
    env_id: str | Callable[..., gym.Env],
    n_envs: int = 1,
    seed: int | None = None,
    start_index: int = 0,
    monitor_dir: str | Path | None = None,
    wrapper_class: Callable[[gym.Env], gym.Env] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[DummyVecEnv | SubprocVecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """Create a wrapped, monitored ``VecEnv``.

    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    Args:
        env_id: either the env ID, the env class or a callable returning an env
        n_envs: the number of environments you wish to have in parallel
        seed: the initial seed for the random number generator
        start_index: start rank index
        monitor_dir: Path to a folder where the monitor files will be saved.
            If None, no file will be written, however, the env will still be wrapped
            in a Monitor wrapper to provide additional information about training.
        wrapper_class: Additional wrapper to use on the environment.
            This can also be a function with single argument that wraps the environment in many things.
            Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
            if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
            See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
        env_kwargs: Optional keyword argument to pass to the env constructor
        vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
        vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
        monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
        wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.

    Returns:
        The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                env = _patch_env(env)

            if seed is not None:
                env.action_space.seed(seed + rank)
            monitor_path = str(Path(monitor_dir) / str(rank)) if monitor_dir is not None else None
            if monitor_path is not None and monitor_dir is not None:
                Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    vec_env.seed(seed)
    return vec_env


def make_atari_env(
    env_id: str | Callable[..., gym.Env],
    n_envs: int = 1,
    seed: int | None = None,
    start_index: int = 0,
    monitor_dir: str | Path | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[DummyVecEnv] | type[SubprocVecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """Create a wrapped, monitored VecEnv for Atari.

    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    .. note::
        By default, the ``AtariWrapper`` uses ``terminal_on_life_loss=True``, which causes
        ``env.reset()`` to perform a no-op step instead of truly resetting when the environment
        terminates due to a loss of life (but not game over). To ensure ``reset()`` always
        resets the env, pass ``wrapper_kwargs=dict(terminal_on_life_loss=False)``.

    Args:
        env_id: either the env ID, the env class or a callable returning an env
        n_envs: the number of environments you wish to have in parallel
        seed: the initial seed for the random number generator
        start_index: start rank index
        monitor_dir: Path to a folder where the monitor files will be saved.
            If None, no file will be written, however, the env will still be wrapped
            in a Monitor wrapper to provide additional information about training.
        wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
        env_kwargs: Optional keyword argument to pass to the env constructor
        vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
        vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
        monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.

    Returns:
        The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=AtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: str | Path | None = None,
    seed: int = 0,
    log_dir: str | Path | None = None,
    should_render: bool = True,
    hyperparams: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[VecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """Create a vectorised environment for evaluating a trained agent.

    Args:
        env_id: Gymnasium environment ID.
        n_envs: Number of parallel environments.
        stats_path: Folder containing saved ``VecNormalize`` stats.
        seed: Random seed.
        log_dir: Optional directory for episode reward logging.
        should_render: Add ``render_mode="human"`` when ``True``.
        hyperparams: Extra hyperparams (e.g. ``frame_stack``, ``normalize``).
        env_kwargs: Keyword arguments passed to the env constructor.
        vec_env_cls: VecEnv class to use (defaults to ``DummyVecEnv``).
        vec_env_kwargs: Keyword arguments for the VecEnv constructor.

    Returns:
        Wrapped :class:`~lit_sb3.vec_env.VecEnv`.
    """
    from lit_sb3.utils.reflection import get_wrapper_class

    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams.copy()

    if "env_wrapper" in hyperparams:
        del hyperparams["env_wrapper"]

    if vec_env_cls is None:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    env_kwargs = deepcopy(env_kwargs or {})
    if "render_mode" not in env_kwargs and should_render:
        env_kwargs["render_mode"] = "human"

    spec = gym.spec(env_id)

    def make_env(**kwargs: Any) -> gym.Env:
        return spec.make(**kwargs)

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,  # type: ignore[arg-type]
        vec_env_kwargs=vec_env_kwargs,
    )

    if "vec_env_wrapper" in hyperparams:
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)  # type: ignore[arg-type]
        assert isinstance(env, VecEnv), "vec_env_wrapper must return a VecEnv"
        del hyperparams["vec_env_wrapper"]

    if stats_path is not None:
        if hyperparams.get("normalize"):
            print("Loading running average")
            print(f"with params: {hyperparams.get('normalize_kwargs', {})}")
            path_ = Path(stats_path) / "vecnormalize.pkl"
            if Path(path_).exists():
                env = VecNormalize.load(path_, env)
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)

    return env
