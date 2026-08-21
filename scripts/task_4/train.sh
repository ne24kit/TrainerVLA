#!/bin/bash

NAME=sarm_libero90_single_stage_zero_state
RUN=${NAME}


LOG_DIR=./logs/$RUN
OUT_DIR=./outputs/$RUN

mkdir -p "$LOG_DIR"

echo "RUN=$RUN"
echo "LOG=$LOG_DIR/train.log"
echo "OUTPUT=$OUT_DIR"

lerobot-train \
  --reward_model.type=sarm \
  --reward_model.annotation_mode=single_stage \
  --reward_model.device=cuda \
  \
  --reward_model.image_key=observation.images.image \
  --reward_model.state_key=observation.state \
  \
  --reward_model.n_obs_steps=8 \
  --reward_model.frame_gap=10 \
  \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=./data/libero_90 \
  --dataset.eval_split=0.05 \
  \
  --batch_size=64 \
  --num_workers=16 \
  --steps=15000 \
  --save_freq=2500 \
  --log_freq=50 \
  \
  --eval_steps=500 \
  --max_eval_samples=1800 \
  \
  --env_eval_freq=0 \
  \
  --wandb.enable=false \
  --output_dir="$OUT_DIR" \
  --job_name="$RUN" \
  --seed=1000 \
  2>&1 | tee "$LOG_DIR/train.log"