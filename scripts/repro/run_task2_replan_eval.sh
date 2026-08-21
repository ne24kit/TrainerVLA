#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ./.venv/bin/activate

for task_id in 0 1 2; do
  for demos in 5 10 25; do
    for action_steps in 10 5; do
      for seed in 1000 2000; do
        ./scripts/task_2/eval_goal_finetune_chunk.sh "$task_id" "$demos" "$action_steps" "$seed"
      done
    done
  done
done
