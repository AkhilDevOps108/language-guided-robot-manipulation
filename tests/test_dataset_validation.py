import json
from pathlib import Path

import numpy as np

from gripground.data.validate_dataset import validate_dataset


def _write_dummy_episode(path: Path) -> None:
    images = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    states = np.zeros((3, 12), dtype=np.float32)
    actions = np.zeros((3, 4), dtype=np.float32)
    rewards = np.zeros((3,), dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    np.savez_compressed(
        path,
        episode_id=path.stem,
        seed=np.array([1], dtype=np.int32),
        instruction=np.array("move cube", dtype=object),
        images=images,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        success=np.array([True], dtype=np.bool_),
        failure_category=np.array("none", dtype=object),
    )


def test_validate_dataset(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True)
    _write_dummy_episode(episodes / "episode_00000.npz")
    manifest = {
        "episode_count": 1,
        "dataset_hash": "x",
        "splits": {"train": ["episode_00000"], "val": [], "test": []},
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_dataset(tmp_path)
    assert report["valid"] is True


def test_validate_rejects_invalid_actions(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir(parents=True)
    _write_dummy_episode(episodes / "episode_00000.npz")
    episode_path = episodes / "episode_00000.npz"
    with np.load(episode_path, allow_pickle=True) as data:
        values = {key: data[key] for key in data.files}
    values["actions"] = np.full((3, 4), np.nan, dtype=np.float32)
    np.savez_compressed(episode_path, **values)
    manifest = {
        "episode_count": 1,
        "dataset_hash": "x",
        "splits": {"train": ["episode_00000"], "val": [], "test": []},
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_dataset(tmp_path)
    assert report["valid"] is False
    assert any("non-finite" in error for error in report["errors"])
