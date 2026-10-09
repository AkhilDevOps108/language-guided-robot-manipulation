from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def start_run(tracking_uri: str, experiment_name: str, run_name: str) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_dict_artifact(data: dict[str, Any], artifact_name: str) -> None:
    mlflow.log_dict(data, artifact_name)


def log_checkpoint(path: Path, artifact_subdir: str = "checkpoints") -> None:
    mlflow.log_artifact(str(path), artifact_subdir)

