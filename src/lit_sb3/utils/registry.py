"""Algorithm registry mapping string names to algorithm classes."""

from lit_sb3.algorithms.a2c.a2c import A2C
from lit_sb3.algorithms.ars.ars import ARS
from lit_sb3.algorithms.base import BaseAlgorithm
from lit_sb3.algorithms.crossq.crossq import CrossQ
from lit_sb3.algorithms.ddpg.ddpg import DDPG
from lit_sb3.algorithms.dqn.dqn import DQN
from lit_sb3.algorithms.ppo.ppo import PPO
from lit_sb3.algorithms.ppo_recurrent.ppo_recurrent import RecurrentPPO
from lit_sb3.algorithms.qrdqn.qrdqn import QRDQN
from lit_sb3.algorithms.sac.sac import SAC
from lit_sb3.algorithms.td3.td3 import TD3
from lit_sb3.algorithms.tqc.tqc import TQC
from lit_sb3.algorithms.trpo.trpo import TRPO

ALGOS: dict[str, type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ars": ARS,
    "crossq": CrossQ,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "ppo_lstm": RecurrentPPO,
    "qrdqn": QRDQN,
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
    "trpo": TRPO,
}
