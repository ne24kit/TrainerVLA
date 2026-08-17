#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS [CKPT] [TRAIN_SEED]"
  echo "  TASK_ID: 0, 1, or 2"
  echo "  DEMOS:   5, 10, or 25"
  echo "  CKPT:    checkpoint step, default 005000"
  echo "  TRAIN_SEED: training seed/run suffix, default 1000"
  echo "  Example: $0 0 5"
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TASK_ID="$1"
DEMOS="$2"
CKPT="${3:-005000}"
TRAIN_SEED="${4:-1000}"
EVAL_SEED=1000
N_EPISODES=50

case "$TASK_ID" in
  0|1|2)
    ;;
  *)
    echo "Unknown TASK_ID: $TASK_ID. Expected 0, 1, or 2." >&2
    exit 1
    ;;
esac

case "$DEMOS" in
  5|10|25)
    ;;
  *)
    echo "Unknown DEMOS: $DEMOS. Expected 5, 10, or 25." >&2
    exit 1
    ;;
esac

if [[ ! "$TRAIN_SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid TRAIN_SEED: $TRAIN_SEED. Expected integer." >&2
  exit 1
fi

EVAL_CMD="${LEROBOT_EVAL:-./.venv/bin/lerobot-eval}"
if [[ ! -x "$EVAL_CMD" ]]; then
  echo "lerobot-eval not found: $EVAL_CMD" >&2
  exit 1
fi

RUN="smolvla_libero90_replay_t${TASK_ID}_d${DEMOS}"
if [[ "$TRAIN_SEED" != "1000" ]]; then
  RUN="${RUN}_s${TRAIN_SEED}"
fi
POLICY_PATH="./outputs/$RUN/checkpoints/$CKPT/pretrained_model"

LOG_DIR="./logs/$RUN/${CKPT}_libero_goal_${TASK_ID}"
OUT_DIR="$LOG_DIR"

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "Checkpoint not found: $POLICY_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "RUN=$RUN"
echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "CKPT=$CKPT"
echo "TRAIN_SEED=$TRAIN_SEED"
echo "EVAL_SEED=$EVAL_SEED"
echo "N_EPISODES=$N_EPISODES"
echo "OUTPUT=$OUT_DIR"

"$EVAL_CMD" \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2","observation.images.image2":"observation.images.camera2"}' \
  \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[$TASK_ID]" \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=1 \
  \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN" \
  --seed="$EVAL_SEED" \
  2>&1 | tee "$LOG_DIR/eval.log"
