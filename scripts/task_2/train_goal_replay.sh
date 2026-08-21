#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS [SEED]"
  echo "  TASK_ID: 0, 1, or 2"
  echo "  DEMOS:   5, 10, or 25"
  echo "  SEED:    training seed, default 1000"
  echo "  Example: $0 0 5"
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TASK_ID="$1"
DEMOS="$2"
SEED="${3:-1000}"

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

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid SEED: $SEED. Expected integer." >&2
  exit 1
fi

TRAIN_CMD="${LEROBOT_TRAIN:-./.venv/bin/lerobot-train}"
if [[ ! -x "$TRAIN_CMD" ]]; then
  echo "lerobot-train not found: $TRAIN_CMD" >&2
  exit 1
fi

NAME="smolvla_libero90_replay_t${TASK_ID}_d${DEMOS}"
if [[ "$SEED" != "1000" ]]; then
  NAME="${NAME}_s${SEED}"
fi
RUN="$NAME"

DATASET_ROOT="./data/libero_goal_replay_t${TASK_ID}_d${DEMOS}"
LOG_DIR="./logs/$RUN"
OUT_DIR="./outputs/$RUN"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset not found: $DATASET_ROOT" >&2
  echo "Run: ./scripts/task_2/prepare_goal_replay_dataset.sh $TASK_ID $DEMOS" >&2
  exit 1
fi

if [[ -e "$LOG_DIR/train.log" || -d "$OUT_DIR/checkpoints" ]]; then
  echo "Run already exists: $RUN" >&2
  echo "Refusing to overwrite logs/checkpoints. Move the old run first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "RUN=$RUN"
echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "SEED=$SEED"
echo "DATASET=$DATASET_ROOT"
echo "LOG=$LOG_DIR/train.log"
echo "OUTPUT=$OUT_DIR"

"$TRAIN_CMD" \
  --policy.path=./outputs/smolvla_libero90_pretrain/checkpoints/070000/pretrained_model \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  \
  --dataset.repo_id="local/libero_goal_replay_t${TASK_ID}_d${DEMOS}" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.eval_split=0.0 \
  \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2","observation.images.image2":"observation.images.camera2"}' \
  \
  --batch_size=16 \
  --num_workers=4 \
  --steps=5000 \
  \
  --log_freq=50 \
  --save_freq=1000 \
  --eval_steps=0 \
  --eval.batch_size=1 \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids="[$TASK_ID]" \
  --eval.n_episodes=5 \
  --env_eval_freq=1000 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  --wandb.enable=false \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN" \
  --seed="$SEED" \
  2>&1 | tee "$LOG_DIR/train.log"
