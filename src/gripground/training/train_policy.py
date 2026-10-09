from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from gripground.config import TrainConfig
from gripground.data.dataset import DemonstrationDataset, load_split_transitions, read_manifest
from gripground.models.policy import LanguageConditionedPolicy
from gripground.tracking.mlflow_utils import log_checkpoint, log_dict_artifact, start_run
from gripground.utils.logging_utils import get_logger
from gripground.utils.reproducibility import set_global_seed

LOGGER = get_logger(__name__)


def collate(batch):
    images = torch.stack([b.image for b in batch], dim=0)
    states = torch.stack([b.state for b in batch], dim=0)
    tokens = torch.stack([b.instruction_tokens for b in batch], dim=0)
    actions = torch.stack([b.action for b in batch], dim=0)
    return images, states, tokens, actions


def run_epoch(model, loader, loss_fn, optimizer=None, device=torch.device("cpu"), image_jitter: float = 0.0):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()
    losses = []
    for images, states, tokens, actions in loader:
        images = images.to(device)
        if train_mode and image_jitter > 0:
            noise = torch.randn_like(images) * image_jitter
            images = torch.clamp(images + noise, 0.0, 1.0)
        states = states.to(device)
        tokens = tokens.to(device)
        actions = actions.to(device)
        preds = model(images, tokens, states)
        loss = loss_fn(preds, actions)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(path: Path, model: LanguageConditionedPolicy, cfg: TrainConfig, epoch: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict["output_dir"] = str(cfg_dict["output_dir"])
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "state_dim": 11,
        "hidden_dim": cfg.hidden_dim,
        "action_dim": 4,
        "model_state_dict": model.state_dict(),
        "train_config": cfg_dict,
        "model_type": "language_conditioned_behavior_cloning",
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train language-conditioned imitation policy")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mlflow-uri", type=str, default="file:./artifacts/mlruns")
    parser.add_argument("--mlflow-experiment", type=str, default="gripground")
    parser.add_argument("--image-jitter", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    train_transitions = load_split_transitions(args.dataset_dir, "train")
    val_transitions = load_split_transitions(args.dataset_dir, "val")
    train_loader = DataLoader(DemonstrationDataset(train_transitions), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(DemonstrationDataset(val_transitions), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    model = LanguageConditionedPolicy(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    train_history = []

    with start_run(cfg.mlflow_tracking_uri, cfg.mlflow_experiment, "train_policy"):
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.epochs,
                "hidden_dim": cfg.hidden_dim,
                "device": str(device),
                "image_jitter": args.image_jitter,
            }
        )
        try:
            for epoch in range(1, cfg.epochs + 1):
                train_loss = run_epoch(
                    model, train_loader, loss_fn, optimizer=optimizer, device=device, image_jitter=args.image_jitter
                )
                val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
                train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                LOGGER.info("Epoch %d/%d train_loss=%.5f val_loss=%.5f", epoch, cfg.epochs, train_loss, val_loss)
                latest = cfg.output_dir / "last.pt"
                save_checkpoint(latest, model, cfg, epoch, val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best = cfg.output_dir / "best.pt"
                    save_checkpoint(best, model, cfg, epoch, val_loss)
                    log_checkpoint(best)
        except KeyboardInterrupt:
            interrupted = cfg.output_dir / "interrupted.pt"
            save_checkpoint(interrupted, model, cfg, epoch=0, val_loss=float("nan"))
            log_checkpoint(interrupted)
            raise

        manifest = read_manifest(args.dataset_dir)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_identifier": manifest["dataset_hash"],
            "dataset_episodes": manifest["episode_count"],
            "model_type": "language_conditioned_behavior_cloning",
            "checkpoint_dir": str(cfg.output_dir),
            "best_checkpoint": str(cfg.output_dir / "best.pt"),
            "training_method": "supervised_behavior_cloning",
            "epochs": cfg.epochs,
            "image_jitter": args.image_jitter,
            "history": train_history,
            "best_val_loss": best_val,
            "known_limitations": [
                "Language encoder is learned from project instructions and is not a pretrained language model.",
                "Training data is simulator-generated from scripted expert trajectories.",
            ],
        }
        with Path("reports/training_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log_dict_artifact(summary, "training_summary.json")


if __name__ == "__main__":
    main()
