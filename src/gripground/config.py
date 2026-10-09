from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EnvConfig:
    image_width: int = 64
    image_height: int = 64
    max_steps: int = 120
    action_repeat: int = 4
    time_step: float = 1.0 / 120.0
    render: bool = False
    target_radius: float = 0.08


@dataclass(slots=True)
class DatasetConfig:
    dataset_dir: Path = Path("artifacts/data")
    schema_version: str = "1.0.0"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-5
    output_dir: Path = Path("artifacts/checkpoints")
    hidden_dim: int = 128
    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    mlflow_experiment: str = "gripground"


@dataclass(slots=True)
class EvalConfig:
    episodes: int = 30
    seed: int = 123
    reports_dir: Path = Path("reports")
