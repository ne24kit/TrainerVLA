#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS R ALPHA CKPT TRAIN_SEED"
  
  echo "  TASK_ID:    0, 1, or 2"
  echo "  DEMOS:      5, 10, or 25"
  echo "  R:          LoRA rank"
  echo "  ALPHA:      LoRA alpha"
  echo "  CKPT:       checkpoint step, default 005000"
  echo "  TRAIN_SEED: training seed, default 1000"
  echo "  Example: $0 1 10 16 32 005000 1000"

}

if [[ $# -ne 6 ]]; then
  usage >&2
  exit 1
fi


TASK_ID="$1"
DEMOS="$2"
R="$3"
ALPHA="$4"
CKPT="$5"
SEED="$6"
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

RUN="smolvla_libero90_pretrained_finetune_t${TASK_ID}_d${DEMOS}_lora_r${R}_alpha${ALPHA}_s_${SEED}"

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
echo "N_EPISODES=$N_EPISODES"
echo "OUTPUT=$OUT_DIR"

lerobot-eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  --policy.use_peft=true \
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
  --seed=1000 \
  2>&1 | tee "$LOG_DIR/eval.log"
