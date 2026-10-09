from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def tokenize_instruction(text: str, vocab_size: int = 128, max_tokens: int = 16) -> np.ndarray:
    tokens = text.lower().split()
    ids = []
    for tok in tokens[:max_tokens]:
        ids.append((hash(tok) % (vocab_size - 1)) + 1)
    if len(ids) < max_tokens:
        ids.extend([0] * (max_tokens - len(ids)))
    return np.asarray(ids, dtype=np.int64)


@dataclass(slots=True)
class TransitionSample:
    image: torch.Tensor
    state: torch.Tensor
    instruction_tokens: torch.Tensor
    action: torch.Tensor


class DemonstrationDataset(Dataset[TransitionSample]):
    def __init__(self, transitions: list[dict[str, Any]]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> TransitionSample:
        item = self.transitions[index]
        image = torch.from_numpy(item["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        state = torch.from_numpy(item["state"].astype(np.float32))
        tokens = torch.from_numpy(item["instruction_tokens"].astype(np.int64))
        action = torch.from_numpy(item["action"].astype(np.float32))
        return TransitionSample(image=image, state=state, instruction_tokens=tokens, action=action)


def load_episode_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "episodes").glob("episode_*.npz"))


def read_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset manifest at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_transitions(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    manifest = read_manifest(dataset_dir)
    episode_ids: list[str] = manifest["splits"][split]
    transitions: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        npz_path = dataset_dir / "episodes" / f"{episode_id}.npz"
        arr = np.load(npz_path, allow_pickle=True)
        images = arr["images"]
        states = arr["states"]
        actions = arr["actions"]
        instruction = str(arr["instruction"].item())
        instruction_tokens = tokenize_instruction(instruction)
        for t in range(actions.shape[0]):
            transitions.append(
                {
                    "image": images[t],
                    "state": states[t],
                    "instruction_tokens": instruction_tokens.copy(),
                    "action": actions[t],
                    "episode_id": episode_id,
                }
            )
    return transitions

