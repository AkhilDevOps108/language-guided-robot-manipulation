from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def generate_markdown_report(reports_dir: Path) -> Path:
    dataset_summary = json.loads((reports_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    training_summary = json.loads((reports_dir / "training_summary.json").read_text(encoding="utf-8"))
    evaluation_summary = json.loads((reports_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    failure_analysis = json.loads((reports_dir / "failure_analysis.json").read_text(encoding="utf-8"))
    baseline = evaluation_summary["baseline_policy"]
    trained = evaluation_summary["trained_policy"]
    config = evaluation_summary.get("configuration", {})
    limitations = failure_analysis["observed_limitations"] + [
        item
        for item in training_summary.get("known_limitations", []) + evaluation_summary.get("known_limitations", [])
        if "language encoder" not in item.lower()
    ]
    content = f"""# GripGround Experiment Report

- Timestamp: {datetime.now(timezone.utc).isoformat()}
- Simulator: {dataset_summary.get("simulator", dataset_summary.get("data_source", "unknown"))}
- Dataset episodes: {dataset_summary["episode_count"]}
- Dataset hash: {dataset_summary["dataset_hash"]}
- Dataset split strategy: {dataset_summary.get("split_by", "not recorded")}
- Training method: behavior cloning
- Training epochs: {training_summary["epochs"]}
- Best validation loss: {training_summary["best_val_loss"]:.5f}
- Checkpoint: {training_summary["best_checkpoint"]}
- Evaluation episodes: {evaluation_summary["evaluation_episodes"]} (seed {config.get("seed", "not recorded")})

## Baseline vs Trained

- Random baseline success rate: {baseline["success_rate"]:.1%} ({round(baseline["success_rate"] * baseline["episodes"])}/{baseline["episodes"]})
- Trained success rate: {trained["success_rate"]:.1%} ({round(trained["success_rate"] * trained["episodes"])}/{trained["episodes"]})
- Trained average episode length: {trained["avg_steps"]:.2f} steps
- Trained inference latency: {trained["avg_inference_latency_ms"]:.2f} ms per action
- Held-out action prediction MSE: {evaluation_summary["action_prediction_mse_on_test_split"]:.5f}

![Baseline vs Trained](baseline_vs_trained.png)

## Failure Analysis

Most common failure categories (baseline and trained runs combined): {failure_analysis["most_common_failures"]}

- Trained-policy failures: {trained["failure_categories"]}
- Baseline failures: {baseline["failure_categories"]}
- Failed trajectory records: {evaluation_summary["failed_trajectory_count"]} (`{evaluation_summary["failed_trajectories_path"]}`)

## Known Limitations

{chr(10).join(f"- {item}" for item in dict.fromkeys(limitations))}

The result is evidence for this simulated task and seed set, not a claim of broad language or physical-robot generalization.
"""
    out = reports_dir / "experiment_report.md"
    out.write_text(content, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown report from experiment artifacts")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = generate_markdown_report(args.reports_dir)
    print(f"Generated report at {out}")


if __name__ == "__main__":
    main()
