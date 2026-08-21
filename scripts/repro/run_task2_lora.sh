#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ./.venv/bin/activate

R="${R:-16}"
ALPHA="${ALPHA:-32}"

for task_id in 0 1 2; do
  for demos in 5 10 25; do
    for seed in 1000 2000; do
      ./scripts/task_2/train_goal_finetune_lora.sh "$task_id" "$demos" "$R" "$ALPHA" "$seed"
      ./scripts/task_2/eval_goal_finetune_lora.sh "$task_id" "$demos" "$R" "$ALPHA" 005000 "$seed"
    done
  done
done
