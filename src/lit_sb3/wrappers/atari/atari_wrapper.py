import gymnasium as gym
import numpy as np

from lit_sb3.wrappers.atari.clip_reward import ClipRewardEnv
from lit_sb3.wrappers.atari.episodic_life import EpisodicLifeEnv
from lit_sb3.wrappers.atari.fire_reset import FireResetEnv
from lit_sb3.wrappers.atari.max_and_skip import MaxAndSkipEnv
from lit_sb3.wrappers.atari.noop_reset import NoopResetEnv
from lit_sb3.wrappers.atari.sticky_action import StickyActionEnv
from lit_sb3.wrappers.atari.warp_frame import WarpFrame


class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Atari 2600 preprocessings.

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    Args:
        env: Environment to wrap
        noop_max: Max number of no-ops
        frame_skip: Frequency at which the agent experiences the game.
            This correspond to repeating the action ``frame_skip`` times.
        screen_size: Resize Atari frame
        terminal_on_life_loss: If True, then step() returns terminated=True whenever a life is lost.
        clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
        action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)
