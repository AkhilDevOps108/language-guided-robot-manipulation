from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from gripground.models.language_encoder import LanguageEncoder
from gripground.models.visual_encoder import VisualEncoder


class LanguageConditionedPolicy(nn.Module):
    def __init__(self, state_dim: int = 11, hidden_dim: int = 128, action_dim: int = 4):
        super().__init__()
        self.visual = VisualEncoder(output_dim=hidden_dim)
        self.language = LanguageEncoder(output_dim=hidden_dim // 2)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, hidden_dim // 2), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, instruction_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.visual(image)
        l = self.language(instruction_tokens)
        s = self.state_proj(state)
        return self.head(torch.cat([v, l, s], dim=1))

    @torch.no_grad()
    def predict(self, image: np.ndarray, instruction_tokens: np.ndarray, state: np.ndarray, device: torch.device) -> np.ndarray:
        self.eval()
        img_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        tok_t = torch.from_numpy(instruction_tokens.astype(np.int64)).unsqueeze(0).to(device)
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        out = self.forward(img_t, tok_t, state_t).squeeze(0).cpu().numpy()
        return out.astype(np.float32)


@dataclass(slots=True)
class LoadedPolicy:
    model: LanguageConditionedPolicy
    device: torch.device
    checkpoint_meta: dict[str, Any]


def load_policy_checkpoint(path: str, device: torch.device) -> LoadedPolicy:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = LanguageConditionedPolicy(
        state_dim=payload.get("state_dim", 11),
        hidden_dim=payload.get("hidden_dim", 128),
        action_dim=payload.get("action_dim", 4),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return LoadedPolicy(model=model, device=device, checkpoint_meta=payload)


class RandomBaselinePolicy:
    def __init__(self, action_dim: int = 4, seed: int = 123):
        self.rng = np.random.default_rng(seed)
        self.action_dim = action_dim

    def predict(self, *_: Any, **__: Any) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32)
