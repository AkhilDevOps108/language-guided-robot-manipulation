import numpy as np
from gymnasium.utils.env_checker import check_env

from gripground.config import EnvConfig
from gripground.data.collect_demonstrations import expert_action
from gripground.env.task_environment import PickPlacePyBulletEnv


def test_env_reset_and_step() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=10))
    obs, info = env.reset(seed=123)
    assert obs["rgb"].shape == (64, 64, 3)
    assert obs["state"].shape == (12,)
    assert "instruction" in info
    result = env.step(np.zeros((4,), dtype=np.float32))
    assert result.observation["rgb"].shape == (64, 64, 3)
    assert len(result) == 5
    env.close()


def test_env_invalid_action_termination() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=10))
    env.reset(seed=123)
    result = env.step(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))
    assert result.done
    assert result.info["failure_category"] == "invalid_action"
    env.close()


def test_expert_success_and_reproducible_reset() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=60))
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    assert np.allclose(first["cube_pos"], second["cube_pos"])
    assert np.allclose(first["target_pos"], second["target_pos"])
    obs = first
    for _ in range(60):
        action, _ = expert_action(obs)
        result = env.step(action)
        obs = result.observation
        if result.done or result.truncated:
            break
    assert result.done
    assert result.info["success"] is True
    env.close()


def test_gymnasium_checker() -> None:
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=10))
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
