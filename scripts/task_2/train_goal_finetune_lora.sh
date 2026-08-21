#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 TASK_ID DEMOS R ALPHA SEED"
  echo "  TASK_ID: 0, 1, or 2"
  echo "  DEMOS:   5, 10, or 25"
  echo "  R:       16"
  echo "  ALPHA:   32"
  echo "  SEED: 1000 or 2000"

  echo "  Example: $0 1 10 16 32 1000"
}

if [[ $# -ne 5 ]]; then
  usage >&2
  exit 1
fi

TASK_ID="$1"  
DEMOS="$2"
R="$3"
ALPHA="$4"
SEED="$5"

case "$TASK_ID" in
  0)
    episodes=(20 26 31 42 58 64 84 90 94 111 117 118 137 140 146 162 168 173 182 187 198 206 220 232 252)
    ;;
  1)
    episodes=(13 15 16 22 36 45 66 76 116 121 145 151 165 166 171 178 179 186 201 219 225 233 237 239 250)
    ;;
  2)
    episodes=(6 10 17 18 25 38 40 44 48 51 53 57 75 89 91 96 97 100 101 103 133 136 149 154 164)
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

selected=("${episodes[@]:0:$DEMOS}")
EPISODES="["
for i in "${!selected[@]}"; do
  if [[ "$i" -gt 0 ]]; then
    EPISODES+=", "
  fi
  EPISODES+="${selected[$i]}"
done
EPISODES+="]"

NAME="smolvla_libero90_pretrained_finetune_t${TASK_ID}_d${DEMOS}_lora_r${R}_alpha${ALPHA}_s_${SEED}"
RUN="${NAME}" #_$(date +%Y%m%d_%H%M%S)

LOG_DIR="./logs/$RUN"
OUT_DIR="./outputs/$RUN"

if [[ -e "$LOG_DIR/train.log" || -d "$OUT_DIR/checkpoints" ]]; then
  echo "Run already exists: $RUN" >&2
  echo "Refusing to overwrite logs/checkpoints. Move the old run first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "RUN=$RUN"
echo "TASK_ID=$TASK_ID"
echo "DEMOS=$DEMOS"
echo "EPISODES=$EPISODES"
echo "LOG=$LOG_DIR/train.log"
echo "OUTPUT=$OUT_DIR"

lerobot-train \
  --policy.path=./outputs/smolvla_libero90_pretrain/checkpoints/070000/pretrained_model \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --peft.method_type=LORA \
  --peft.r=$R \
  --peft.lora_alpha=$ALPHA \
  \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=./data/libero_goal \
  --dataset.episodes="$EPISODES" \
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
