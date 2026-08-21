#!/bin/bash
RUN="smolvla_libero90_pretrain"
CKPT="last"
OUT_DIR="./outputs/$RUN"
LOG_DIR="./logs/$RUN"
CKPT_DIR="$OUT_DIR/checkpoints/$CKPT"
CONFIG_PATH="$CKPT_DIR/pretrained_model/train_config.json"
TRAINING_STEP_PATH="$CKPT_DIR/training_state/training_step.json"
LOG_FILE="$LOG_DIR/train_resume_$(date +%Y%m%d_%H%M%S).log"


mkdir -p "$LOG_DIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RUN=$RUN"
echo "CKPT=$CKPT ($(basename "$(readlink -f "$CKPT_DIR")"))"
echo "CONFIG=$CONFIG_PATH"
echo "TRAINING_STEP=$TRAINING_STEP_PATH"
echo "OUTPUT=$OUT_DIR"
echo "LOG=$LOG_FILE"

lerobot-train \
  --config_path="$CONFIG_PATH" \
  --resume=true \
  2>&1 | tee "$LOG_FILE"
