#!/usr/bin/env bash
set -euo pipefail

python -m gripground.data.collect_demonstrations --output-dir artifacts/data --episodes 30 --seed 42
python -m gripground.data.validate_dataset --dataset-dir artifacts/data
python -m gripground.training.train_policy --dataset-dir artifacts/data --output-dir artifacts/checkpoints --epochs 8 --seed 42
python -m gripground.training.evaluate_policy --dataset-dir artifacts/data --checkpoint artifacts/checkpoints/best.pt --reports-dir reports --episodes 20 --seed 123
python -m gripground.evaluation.generate_report --reports-dir reports
