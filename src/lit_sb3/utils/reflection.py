"""Dynamic class and wrapper discovery utilities."""

from collections.abc import Callable
import importlib
from typing import Any

import gymnasium as gym

from lit_sb3.callbacks import BaseCallback


def _get_module_name(dotted: str) -> str:
    return ".".join(dotted.split(".")[:-1])


def _get_class_name(dotted: str) -> str:
    return dotted.split(".")[-1]


def get_class_by_name(name: str) -> type:
    """Import and return a class given a dotted name string.

    Example::

        get_class_by_name("lit_sb3.callbacks.CheckpointCallback")
    """
    module = importlib.import_module(_get_module_name(name))
    return getattr(module, _get_class_name(name))


def get_wrapper_class(
    hyperparams: dict[str, Any],
    key: str = "env_wrapper",
) -> Callable[[gym.Env], gym.Env] | None:
    """Return a callable that applies one or more gym wrappers.

    The ``key`` entry in *hyperparams* may be a single dotted class name or a
    list.  Each entry may optionally be a ``{ClassName: kwargs_dict}`` mapping
    to pass keyword arguments to the wrapper constructor.

    Args:
        hyperparams: Hyperparameter dict that may contain ``key``.
        key: The hyperparams key to inspect (``"env_wrapper"`` or
            ``"vec_env_wrapper"``).

    Returns:
        A callable ``wrap_env(env) -> env`` or ``None`` if the key is absent.
    """
    if key not in hyperparams:
        return None

    wrapper_name = hyperparams.get(key)
    if wrapper_name is None:
        return None

    wrapper_names: list[Any] = wrapper_name if isinstance(wrapper_name, list) else [wrapper_name]

    wrapper_classes: list[type] = []
    wrapper_kwargs_list: list[dict[str, Any]] = []

    for entry in wrapper_names:
        if isinstance(entry, dict):
            assert len(entry) == 1, f"Error in YAML near {entry}: expected single-key dict."
            class_name = next(iter(entry.keys()))
            kwargs: dict[str, Any] = entry[class_name]
        else:
            class_name = entry
            kwargs = {}

        if isinstance(class_name, str):
            module = importlib.import_module(_get_module_name(class_name))
            wrapper_class = getattr(module, _get_class_name(class_name))
        elif isinstance(class_name, type):
            wrapper_class = class_name
        else:
            raise TypeError(f"Unexpected value {class_name!r} for {key}; must be a str or class.")

        wrapper_classes.append(wrapper_class)
        wrapper_kwargs_list.append(kwargs)

    def wrap_env(env: gym.Env) -> gym.Env:
        for cls, kw in zip(wrapper_classes, wrapper_kwargs_list, strict=True):
            env = cls(env, **kw)
        return env

    return wrap_env


def get_callback_list(hyperparams: dict[str, Any]) -> list[BaseCallback]:
    r"""Return a list of instantiated callbacks from *hyperparams[\"callback\"]*.

    Each entry can be:
    * a dotted class name string,
    * ``{ClassName: kwargs_dict}``, or
    * an already-instantiated :class:`~lit_sb3.callbacks.BaseCallback`.
    """
    callbacks: list[BaseCallback] = []
    if "callback" not in hyperparams:
        return callbacks

    callback_name = hyperparams.get("callback")
    if callback_name is None:
        return callbacks

    callback_names: list[Any] = callback_name if isinstance(callback_name, list) else [callback_name]

    for entry in callback_names:
        if isinstance(entry, dict):
            assert len(entry) == 1, f"Error in YAML near {entry}: expected single-key dict."
            name = next(iter(entry.keys()))
            kwargs = entry[name]
        else:
            name = entry
            kwargs = {}

        if isinstance(name, BaseCallback):
            callbacks.append(name)
        else:
            assert isinstance(name, str), f"Expected str callback name, got {type(name)}"
            callback_class = get_class_by_name(name)
            callbacks.append(callback_class(**kwargs))

    return callbacks
