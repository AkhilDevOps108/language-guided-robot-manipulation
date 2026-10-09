from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]


class SimulatorAdapter(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None, instruction: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def action_space_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

