from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_failure_analysis(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    trained_failures = evaluation_summary["trained_policy"]["failure_categories"]
    baseline_failures = evaluation_summary["baseline_policy"]["failure_categories"]
    merged = Counter(trained_failures)
    merged.update({f"baseline::{k}": v for k, v in baseline_failures.items()})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "most_common_failures": merged.most_common(5),
        "trained_failure_categories": trained_failures,
        "baseline_failure_categories": baseline_failures,
        "observed_limitations": [
            "Policy relies on imitation from scripted expert and may fail in unseen cube placements.",
            "Language conditioning uses learned token embeddings, not a pretrained language model.",
            "PyBullet fallback was used because ManiSkill dependency resolution failed in this environment.",
        ],
    }


def save_failure_analysis(reports_dir: Path, evaluation_summary: dict[str, Any]) -> Path:
    report = build_failure_analysis(evaluation_summary)
    out_path = reports_dir / "failure_analysis.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path

