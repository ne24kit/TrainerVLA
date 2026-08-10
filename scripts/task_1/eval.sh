#!/bin/bash

RUN=smolvla_libero90_pretrain
CKPT=100000

POLICY_PATH="./outputs/$RUN/checkpoints/$CKPT/pretrained_model"

LOG_DIR=./logs/$RUN/$CKPT
mkdir -p "$LOG_DIR"

OUT_DIR=$LOG_DIR

echo "RUN=$RUN"
echo "CKPT=$CKPT"
echo "OUTPUT=$OUT_DIR"

lerobot-eval \
  --policy.path="$POLICY_PATH" \
  --policy.device=cuda \
  \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2","observation.images.image2":"observation.images.camera2"}' \
  \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[0,1,2]' \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN" \
  --seed=1000 \
  2>&1 | tee "$LOG_DIR/eval.log"
