from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gripground.data.dataset import read_manifest


def validate_episode(path: Path) -> list[str]:
    errors: list[str] = []
    arr = np.load(path, allow_pickle=True)
    for key in ("images", "states", "actions", "rewards", "dones"):
        if key not in arr:
            errors.append(f"{path.name}: missing key '{key}'")
    if errors:
        return errors
    images = arr["images"]
    states = arr["states"]
    actions = arr["actions"]
    if images.ndim != 4 or images.shape[-1] != 3:
        errors.append(f"{path.name}: images shape must be [T,H,W,3], got {images.shape}")
    if states.ndim != 2 or states.shape[1] != 12:
        errors.append(f"{path.name}: states shape must be [T,12], got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != 4:
        errors.append(f"{path.name}: action shape must be [T,4], got {actions.shape}")
    t = actions.shape[0] if actions.ndim > 0 else -1
    if states.ndim > 0 and states.shape[0] != t or images.ndim > 0 and images.shape[0] != t:
        errors.append(f"{path.name}: inconsistent sequence length across images/states/actions")
    for key in ("rewards", "dones"):
        if arr[key].ndim != 1 or arr[key].shape[0] != t:
            errors.append(f"{path.name}: {key} must have shape [T]")
    if not np.isfinite(images).all():
        errors.append(f"{path.name}: images contain non-finite values")
    if not np.isfinite(states).all():
        errors.append(f"{path.name}: states contain non-finite values")
    if not np.isfinite(actions).all():
        errors.append(f"{path.name}: actions contain non-finite values")
    if np.any(actions < -1.01) or np.any(actions > 1.01):
        errors.append(f"{path.name}: actions are outside expected clipped range [-1, 1]")
    return errors


def validate_dataset(dataset_dir: Path) -> dict[str, object]:
    manifest = read_manifest(dataset_dir)
    episode_dir = dataset_dir / "episodes"
    files = sorted(episode_dir.glob("episode_*.npz"))
    errors: list[str] = []
    ids = [f.stem for f in files]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate episode identifiers detected")
    split_union: list[str] = []
    for split in ("train", "val", "test"):
        split_union.extend(manifest["splits"].get(split, []))
    if len(split_union) != len(set(split_union)):
        errors.append("Train/validation/test leakage detected from overlapping episode IDs")
    if set(split_union) != set(ids):
        errors.append("Manifest splits do not cover exactly the episode files in the dataset")
    if manifest.get("episode_count") != len(files):
        errors.append("Manifest episode_count does not match the number of episode files")
    for path in files:
        errors.extend(validate_episode(path))
    report = {
        "dataset_dir": str(dataset_dir),
        "episode_count": len(files),
        "valid": len(errors) == 0,
        "errors": errors,
    }
    out_path = Path("reports")
    out_path.mkdir(exist_ok=True)
    with (out_path / "dataset_validation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate collected demonstration dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset_dir)
    if not report["valid"]:
        raise SystemExit(f"Dataset validation failed:\n" + "\n".join(report["errors"]))  # type: ignore[index]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
