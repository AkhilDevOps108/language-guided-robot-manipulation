from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_evaluation(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    required = ("dataset_identifier", "evaluation_episodes", "configuration", "trained_policy", "baseline_policy")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    return result


def compare_safety_trials(before_path: Path, after_path: Path, output_path: Path) -> Path:
    before = _load_evaluation(before_path)
    after = _load_evaluation(after_path)
    before_rate = before["trained_policy"]["success_rate"]
    after_rate = after["trained_policy"]["success_rate"]
    same_dataset = before["dataset_identifier"] == after["dataset_identifier"]
    same_evaluation = (
        before["evaluation_episodes"] == after["evaluation_episodes"]
        and before["configuration"].get("seed") == after["configuration"].get("seed")
    )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "descriptive_not_controlled",
        "before": {
            "source": str(before_path),
            "dataset_identifier": before["dataset_identifier"],
            "evaluation_episodes": before["evaluation_episodes"],
            "seed": before["configuration"].get("seed"),
            "checkpoint_path": before.get("model_checkpoint"),
            "trained_success_rate": before_rate,
            "baseline_success_rate": before["baseline_policy"]["success_rate"],
        },
        "after": {
            "source": str(after_path),
            "dataset_identifier": after["dataset_identifier"],
            "evaluation_episodes": after["evaluation_episodes"],
            "seed": after["configuration"].get("seed"),
            "checkpoint_path": after.get("model_checkpoint"),
            "trained_success_rate": after_rate,
            "baseline_success_rate": after["baseline_policy"]["success_rate"],
        },
        "comparison": {
            "trained_success_rate_delta": after_rate - before_rate,
            "same_dataset": same_dataset,
            "same_evaluation_episodes_and_seed": same_evaluation,
        },
        "interpretation": (
            "The observed rates are not an isolated estimate of the safety interlock effect: "
            "the dataset identifiers differ, and the reports do not fingerprint checkpoint weights. "
            "Treat the difference as descriptive; a causal comparison requires the same dataset, "
            "checkpoint, environment settings, and evaluation seeds with only the interlock changed."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recorded safety-interlock evaluation trials")
    parser.add_argument("--before", type=Path, default=Path("reports/safety_trial/evaluation_summary.json"))
    parser.add_argument("--after", type=Path, default=Path("reports/safety_trial2/evaluation_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/safety_interlock_comparison.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compare_safety_trials(args.before, args.after, args.output)
    print(f"Generated comparison at {output}")


if __name__ == "__main__":
    main()
