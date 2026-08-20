#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

for task_id in 0 1 2; do
  for demos in 5 10 25; do
    dataset_root="./data/libero_goal_replay_t${task_id}_d${demos}"
    if [[ -d "$dataset_root" && "${FORCE:-0}" != "1" ]]; then
      echo "Skip existing dataset: $dataset_root"
      continue
    fi

    FORCE="${FORCE:-0}" ./scripts/task_2/prepare_goal_replay_dataset.sh "$task_id" "$demos"
  done
done
