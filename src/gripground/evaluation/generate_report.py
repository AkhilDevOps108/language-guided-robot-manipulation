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
    content = f"""# GripGround Experiment Report

- Timestamp: {datetime.now(timezone.utc).isoformat()}
- Dataset episodes: {dataset_summary["episode_count"]}
- Dataset hash: {dataset_summary["dataset_hash"]}
- Training method: behavior cloning
- Checkpoint: {training_summary["best_checkpoint"]}
- Evaluation episodes: {evaluation_summary["evaluation_episodes"]}

## Baseline vs Trained

- Baseline success rate: {evaluation_summary["baseline_policy"]["success_rate"]:.3f}
- Trained success rate: {evaluation_summary["trained_policy"]["success_rate"]:.3f}

![Baseline vs Trained](baseline_vs_trained.png)

## Failure Analysis

Most common failures: {failure_analysis["most_common_failures"]}

## Known Limitations

{chr(10).join(f"- {item}" for item in failure_analysis["observed_limitations"])}
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

