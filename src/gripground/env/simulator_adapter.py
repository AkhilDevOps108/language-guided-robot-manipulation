from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import numpy as np


class StepResult(NamedTuple):
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated


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
