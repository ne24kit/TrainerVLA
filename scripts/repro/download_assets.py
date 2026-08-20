#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import subprocess

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]

ASSETS = [
    ("lerobot/smolvla_base", "model", ROOT / "models" / "smolvla_base"),
    ("aliangdw/Robometer-4B-LIBERO", "model", ROOT / "models" / "Robometer-4B-LIBERO"),
]

ROBOMETER_REPO = os.environ.get("ROBOMETER_REPO", "https://github.com/aliang8/robometer.git")


def main() -> None:
    robometer_dir = ROOT / "robometer"
    if not robometer_dir.exists():
        print(f"Cloning Robometer code -> {robometer_dir}")
        subprocess.run(["git", "clone", "--depth", "1", ROBOMETER_REPO, str(robometer_dir)], check=True)
    else:
        print(f"Robometer code already exists: {robometer_dir}")

    for repo_id, repo_type, local_dir in ASSETS:
        print(f"Downloading {repo_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
        )
        print(f"Done: {path}")


if __name__ == "__main__":
    main()
