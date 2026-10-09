from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np


@dataclass(slots=True)
class EpisodeResult:
    success: bool
    steps: int
    latency_ms: float
    failure_category: str


def summarize_episode_results(results: list[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    success_count = sum(1 for r in results if r.success)
    failure_count = n - success_count
    avg_steps = mean([r.steps for r in results]) if results else 0.0
    avg_latency = mean([r.latency_ms for r in results]) if results else 0.0
    failures = Counter(r.failure_category for r in results if not r.success)
    return {
        "episodes": n,
        "success_rate": success_count / n if n else 0.0,
        "failure_rate": failure_count / n if n else 0.0,
        "avg_steps": avg_steps,
        "avg_inference_latency_ms": avg_latency,
        "failure_categories": dict(failures),
    }


def action_mse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch for action_mse: {pred.shape} != {target.shape}")
    return float(np.mean((pred - target) ** 2))

