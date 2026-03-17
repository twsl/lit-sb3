import gymnasium as gym
import numpy as np

from lit_sb3.common.types import AtariResetReturn, AtariStepReturn


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    Done by DeepMind for the DQN and co. since it helps value estimation.

    .. note::
        This wrapper changes the behavior of ``env.reset()``. When the environment
        terminates due to a loss of life (but not game over), calling ``reset()`` will
        perform a no-op step instead of truly resetting the environment. This can be
        confusing when evaluating or testing agents. To avoid this behavior and ensure ``reset()``
        always resets to the env, set ``terminal_on_life_loss=False`` when
        using ``make_atari_env()``.

    Args:
        env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        Args:
            kwargs: Extra keywords passed to env.reset() call

        Returns:
            the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info
