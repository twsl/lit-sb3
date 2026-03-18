import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import cv2  # ty:ignore[unresolved-import]

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """Convert to grayscale and warp frames to 84x84 (default).

    as done in the Nature paper and later work.

    Args:
        env: Environment to wrap
        width: New frame width
        height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:  # ty:ignore[invalid-method-override]
        """Returns the current observation from a frame.

        Args:
            frame: environment frame

        Returns:
            the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]
