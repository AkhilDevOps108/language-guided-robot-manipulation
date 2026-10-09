import numpy as np

from gripground.config import EnvConfig
from gripground.env.task_environment import PickPlacePyBulletEnv


def test_env_reset_and_step() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=10))
    obs, info = env.reset(seed=123)
    assert obs["rgb"].shape == (64, 64, 3)
    assert obs["state"].shape == (12,)
    assert "instruction" in info
    result = env.step(np.zeros((4,), dtype=np.float32))
    assert result.observation["rgb"].shape == (64, 64, 3)
    env.close()


def test_env_invalid_action_termination() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=10))
    env.reset(seed=123)
    result = env.step(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))
    assert result.done
    assert result.info["failure_category"] == "invalid_action"
    env.close()
