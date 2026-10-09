#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m gripground.data.collect_demonstrations --output-dir artifacts/data --episodes 90 --seed 42 --max-steps 60
"$PYTHON" -m gripground.data.validate_dataset --dataset-dir artifacts/data
"$PYTHON" -m gripground.training.train_policy --dataset-dir artifacts/data --output-dir artifacts/checkpoints --epochs 50 --batch-size 64 --seed 42
"$PYTHON" -m gripground.training.evaluate_policy --dataset-dir artifacts/data --checkpoint artifacts/checkpoints/best.pt --reports-dir reports --episodes 30 --seed 123
"$PYTHON" -m gripground.evaluation.generate_report --reports-dir reports
