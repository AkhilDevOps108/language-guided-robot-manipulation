from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from gripground.config import EnvConfig, EvalConfig
from gripground.data.dataset import load_split_transitions, read_manifest, tokenize_instruction
from gripground.env.task_environment import PickPlacePyBulletEnv
from gripground.evaluation.error_analysis import save_failure_analysis
from gripground.evaluation.metrics import EpisodeResult, action_mse, summarize_episode_results
from gripground.models.policy import RandomBaselinePolicy, load_policy_checkpoint
from gripground.utils.reproducibility import set_global_seed


def run_policy_episode(env: PickPlacePyBulletEnv, policy: Any, seed: int, max_steps: int, device: torch.device | None = None) -> EpisodeResult:
    obs, _ = env.reset(seed=seed, instruction="move the red cube to the green target")
    total_latency_ms = 0.0
    final_failure = "timeout"
    success = False
    for step in range(1, max_steps + 1):
        instruction_tokens = tokenize_instruction(obs["instruction"])
        t0 = time.perf_counter()
        if hasattr(policy, "model"):
            action = policy.model.predict(obs["rgb"], instruction_tokens, obs["state"], device=policy.device)
        else:
            action = policy.predict(obs["rgb"], instruction_tokens, obs["state"])
        t1 = time.perf_counter()
        total_latency_ms += (t1 - t0) * 1000.0
        result = env.step(action)
        obs = result.observation
        if result.done or result.truncated:
            success = bool(result.info.get("success", False))
            final_failure = str(result.info.get("failure_category", "timeout"))
            return EpisodeResult(
                success=success,
                steps=step,
                latency_ms=total_latency_ms / step,
                failure_category=final_failure if not success else "none",
            )
    return EpisodeResult(success=success, steps=max_steps, latency_ms=total_latency_ms / max_steps, failure_category=final_failure)


def evaluate_action_mse(dataset_dir: Path, loaded_policy, sample_limit: int = 256) -> float | None:
    transitions = load_split_transitions(dataset_dir, "test")
    if not transitions:
        return None
    take = transitions[: min(sample_limit, len(transitions))]
    preds = []
    targets = []
    for tr in take:
        pred = loaded_policy.model.predict(tr["image"], tr["instruction_tokens"], tr["state"], loaded_policy.device)
        preds.append(pred)
        targets.append(tr["action"])
    return action_mse(np.asarray(preds), np.asarray(targets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and trained policy")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(episodes=args.episodes, seed=args.seed, reports_dir=args.reports_dir)
    set_global_seed(cfg.seed)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_policy = load_policy_checkpoint(str(args.checkpoint), device=device)
    baseline = RandomBaselinePolicy(seed=cfg.seed)

    env = PickPlacePyBulletEnv(EnvConfig(max_steps=120))
    baseline_results = []
    trained_results = []
    for i in range(cfg.episodes):
        seed = cfg.seed + i
        baseline_results.append(run_policy_episode(env, baseline, seed=seed, max_steps=120))
        trained_results.append(run_policy_episode(env, loaded_policy, seed=seed, max_steps=120))
    env.close()

    baseline_summary = summarize_episode_results(baseline_results)
    trained_summary = summarize_episode_results(trained_results)
    mse = evaluate_action_mse(args.dataset_dir, loaded_policy)
    manifest = read_manifest(args.dataset_dir)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {"episodes": cfg.episodes, "seed": cfg.seed},
        "dataset_identifier": manifest["dataset_hash"],
        "dataset_episode_count": manifest["episode_count"],
        "model_checkpoint": str(args.checkpoint),
        "training_method": "supervised_behavior_cloning",
        "evaluation_episodes": cfg.episodes,
        "baseline_policy": baseline_summary,
        "trained_policy": trained_summary,
        "action_prediction_mse_on_test_split": mse,
        "known_limitations": [
            "Action-space model is continuous and clipped in [-1, 1].",
            "Evaluation uses a single simulated task family and not physical robot deployment.",
        ],
    }
    with (args.reports_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    labels = ["Baseline", "Trained"]
    values = [baseline_summary["success_rate"], trained_summary["success_rate"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["gray", "steelblue"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("Baseline vs Trained Policy")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(args.reports_dir / "baseline_vs_trained.png")
    plt.close(fig)

    save_failure_analysis(args.reports_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

