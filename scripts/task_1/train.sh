#!/bin/bash

NAME=smolvla_libero90_b64_10k
RUN=${NAME}_$(date +%Y%m%d_%H%M%S)

LOG_DIR=./logs/$RUN
OUT_DIR=./outputs/$RUN

mkdir -p "$LOG_DIR"

echo "RUN=$RUN"
echo "LOG=$LOG_DIR/train.log"
echo "OUTPUT=$OUT_DIR"

lerobot-train \
  --policy.path=./models/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=./data/libero_90 \
  --dataset.eval_split=0.05 \
  \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2","observation.images.image2":"observation.images.camera2"}' \
  \
  --batch_size=64 \
  --num_workers=16 \
  --steps=10000 \
  \
  --log_freq=50 \
  --save_freq=2000 \
  --eval_steps=1000 \
  --eval.batch_size=1 \
  --max_eval_samples=1800 \
  --env.type=libero \
  --env.task=libero_90 \
  --env.task_ids='[0,15,20,40,57,60,67,70]' \
  --eval.n_episodes=3 \
  --env_eval_freq=2000 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  --wandb.enable=false \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN" \
  --seed=1000 \
  2>&1 | tee "$LOG_DIR/train.log"
