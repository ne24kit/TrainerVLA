#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ./.venv/bin/activate

for task_id in 0 1 2; do
  for demos in 5 10 25; do
    dataset_root="./data/libero_goal_replay_t${task_id}_d${demos}"
    if [[ ! -d "$dataset_root" ]]; then
      ./scripts/task_2/prepare_goal_replay_dataset.sh "$task_id" "$demos"
    fi

    for seed in 1000 2000; do
      ./scripts/task_2/train_goal_replay.sh "$task_id" "$demos" "$seed"
      ./scripts/task_2/eval_goal_replay.sh "$task_id" "$demos" 005000 "$seed"
    done
  done
done
