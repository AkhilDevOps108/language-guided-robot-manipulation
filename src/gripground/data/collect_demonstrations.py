from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gripground.config import DatasetConfig, EnvConfig
from gripground.env.task_environment import PickPlacePyBulletEnv
from gripground.utils.logging_utils import get_logger
from gripground.utils.reproducibility import set_global_seed

LOGGER = get_logger(__name__)

INSTRUCTION_POOL = [
    "move the red cube to the green target",
    "pick the cube and place it on the goal region",
    "grasp the red block and set it down on the target",
]


def expert_action(observation: dict[str, Any], stage: int) -> tuple[np.ndarray, int]:
    ee = observation["ee_pos"]
    cube = observation["cube_pos"]
    target = observation["target_pos"]
    action = np.zeros(4, dtype=np.float32)
    if stage == 0:
        goal = cube + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        delta = goal - ee
        action[:3] = np.clip(delta / 0.03, -1.0, 1.0)
        action[3] = -1
        if np.linalg.norm(delta) < 0.03:
            stage = 1
    elif stage == 1:
        goal = cube + np.array([0.0, 0.0, 0.035], dtype=np.float32)
        delta = goal - ee
        action[:3] = np.clip(delta / 0.03, -1.0, 1.0)
        action[3] = -1
        if np.linalg.norm(delta) < 0.025:
            stage = 2
    elif stage == 2:
        action[3] = 1
        if np.linalg.norm(ee - cube) < 0.05:
            stage = 3
    elif stage == 3:
        goal = cube + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        delta = goal - ee
        action[:3] = np.clip(delta / 0.03, -1.0, 1.0)
        action[3] = 1
        if np.linalg.norm(delta) < 0.03:
            stage = 4
    elif stage == 4:
        goal = target + np.array([0.0, 0.0, 0.16], dtype=np.float32)
        delta = goal - ee
        action[:3] = np.clip(delta / 0.03, -1.0, 1.0)
        action[3] = 1
        if np.linalg.norm(delta) < 0.04:
            stage = 5
    elif stage == 5:
        goal = target + np.array([0.0, 0.0, 0.07], dtype=np.float32)
        delta = goal - ee
        action[:3] = np.clip(delta / 0.03, -1.0, 1.0)
        action[3] = 1
        if np.linalg.norm(delta) < 0.03:
            stage = 6
    else:
        action[3] = -1
        action[2] = 0.5
    return action, stage


def split_episode_ids(episode_ids: list[str], cfg: DatasetConfig) -> dict[str, list[str]]:
    n = len(episode_ids)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = episode_ids[:n_train]
    val = episode_ids[n_train : n_train + n_val]
    test = episode_ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_dataset(output_dir: Path, episodes: int, seed: int, max_steps: int) -> dict[str, Any]:
    set_global_seed(seed)
    env = PickPlacePyBulletEnv(EnvConfig(max_steps=max_steps))
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_ids: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    for idx in range(episodes):
        episode_seed = seed + idx
        instruction = INSTRUCTION_POOL[idx % len(INSTRUCTION_POOL)]
        obs, _ = env.reset(seed=episode_seed, instruction=instruction)
        images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        stage = 0
        success = False
        failure_category = "timeout"
        for _ in range(max_steps):
            act, stage = expert_action(obs, stage)
            result = env.step(act)
            images.append(obs["rgb"])
            states.append(obs["state"])
            actions.append(act)
            rewards.append(result.reward)
            dones.append(result.done or result.truncated)
            obs = result.observation
            if result.done or result.truncated:
                success = bool(result.info.get("success", False))
                failure_category = str(result.info.get("failure_category", "timeout"))
                break
        episode_id = f"episode_{idx:05d}"
        np.savez_compressed(
            episodes_dir / f"{episode_id}.npz",
            episode_id=episode_id,
            seed=np.array([episode_seed], dtype=np.int32),
            instruction=np.array(instruction, dtype=object),
            images=np.asarray(images, dtype=np.uint8),
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            success=np.array([success], dtype=np.bool_),
            failure_category=np.array(failure_category, dtype=object),
        )
        episode_ids.append(episode_id)
        summary_rows.append(
            {
                "episode_id": episode_id,
                "seed": episode_seed,
                "instruction": instruction,
                "steps": len(actions),
                "success": success,
                "failure_category": failure_category,
            }
        )
    env.close()

    cfg = DatasetConfig(dataset_dir=output_dir)
    splits = split_episode_ids(episode_ids, cfg)
    paths = sorted(episodes_dir.glob("episode_*.npz"))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": cfg.schema_version,
        "data_source": "simulated_pybullet",
        "simulator": "pybullet",
        "episode_count": len(episode_ids),
        "split_by": "episode",
        "preprocessing": {"image_height": 64, "image_width": 64, "action_clip": [-1.0, 1.0]},
        "splits": splits,
        "dataset_hash": dataset_hash(paths),
        "episodes": summary_rows,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with (reports_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PyBullet pick-and-place demonstrations")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect_dataset(args.output_dir, args.episodes, args.seed, args.max_steps)
    LOGGER.info("Collected %d episodes at %s", manifest["episode_count"], args.output_dir)


if __name__ == "__main__":
    main()

