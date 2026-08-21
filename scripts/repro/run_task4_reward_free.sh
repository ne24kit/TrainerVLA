#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source ./.venv/bin/activate

if [[ "${RUN_SARM_TRAIN:-1}" == "1" ]]; then
  ./scripts/task_4/train.sh
fi

for task_id in 0 1 2; do
  for demos in 0 5 10 25; do
    for seed in 1001 1002; do
      ./scripts/task_4/eval_with_record.sh "$task_id" "$demos" "$seed"
      ./scripts/task_4/eval_sarm.sh "$task_id" "$demos" "$seed"
      ./scripts/task_4/eval_robometer.sh "$task_id" "$demos" "$seed"
    done
  done
done
