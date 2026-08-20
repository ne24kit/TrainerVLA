#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ./.venv/bin/activate

if [[ "${RUN_SEEN_PRETRAIN:-0}" == "1" ]]; then
  ./scripts/task_1/train.sh
fi

./scripts/task_1/eval.sh

for task_id in 0 1 2; do
  for demos in 5 10 25; do
    ./scripts/task_1/train_goal_finetune.sh "$task_id" "$demos"
    ./scripts/task_1/eval_goal_finetune.sh "$task_id" "$demos" 005000
  done
done
