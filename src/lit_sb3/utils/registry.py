"""Algorithm registry mapping string names to algorithm classes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lit_sb3.algorithms.base import BaseAlgorithm


class AlgorithmRegistry(Mapping[str, object]):
    """Resolve algorithm classes on access while exposing a mapping interface."""

    def __init__(self, targets: dict[str, str]) -> None:
        self._targets = targets
        self._cache: dict[str, type[BaseAlgorithm]] = {}

    def __getitem__(self, key: str) -> type[BaseAlgorithm]:
        if key not in self._targets:
            raise KeyError(key)
        if key not in self._cache:
            module_name, class_name = self._targets[key].rsplit(".", 1)
            module = importlib.import_module(module_name)
            self._cache[key] = getattr(module, class_name)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def __len__(self) -> int:
        return len(self._targets)


ALGOS = AlgorithmRegistry(
    {
        "a2c": "lit_sb3.algorithms.a2c.a2c.A2C",
        "ars": "lit_sb3.algorithms.ars.ars.ARS",
        "crossq": "lit_sb3.algorithms.crossq.crossq.CrossQ",
        "ddpg": "lit_sb3.algorithms.ddpg.ddpg.DDPG",
        "dqn": "lit_sb3.algorithms.dqn.dqn.DQN",
        "ppo": "lit_sb3.algorithms.ppo.ppo.PPO",
        "ppo_lstm": "lit_sb3.algorithms.ppo_recurrent.ppo_recurrent.RecurrentPPO",
        "qrdqn": "lit_sb3.algorithms.qrdqn.qrdqn.QRDQN",
        "sac": "lit_sb3.algorithms.sac.sac.SAC",
        "td3": "lit_sb3.algorithms.td3.td3.TD3",
        "tqc": "lit_sb3.algorithms.tqc.tqc.TQC",
        "trpo": "lit_sb3.algorithms.trpo.trpo.TRPO",
    }
)
