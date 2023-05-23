#!/bin/bash
export RESUME_CHECKPOINT="latest"
export CHECKPOINTING_STEPS=1000
export NUM_INSTANCE_IMAGES=12695 #@param {type:"integer"}
export LEARNING_RATE=4e-8 #@param {type:"number"}

# Configure these values.
export MODEL_NAME="ptx0/pseudo-journey"
export BASE_DIR="/notebooks/images/datasets"
export INSTANCE_DIR="${BASE_DIR}/images"
export OUTPUT_DIR="${BASE_DIR}/models"

# Regularization data config.
# This helps retain previous knowledge from the model.
export CLASS_PROMPT="a person"
export CLASS_DIR="${BASE_DIR}/regularization"
export NUM_CLASS_IMAGES=$((NUM_INSTANCE_IMAGES * 12))

export MAX_NUM_STEPS=$((NUM_INSTANCE_IMAGES * 80))
export LR_SCHEDULE="polynomial"
export LR_WARMUP_STEPS=$((MAX_NUM_STEPS / 10))

export TRAIN_BATCH_SIZE=3
export RESOLUTION=768
export MIXED_PRECISION="bf16"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="${MODEL_NAME}"  \
  --instance_data_dir="${INSTANCE_DIR}" \
  --class_data_dir="${CLASS_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --class_prompt="${CLASS_PROMPT}" \
  --resolution=${RESOLUTION} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --use_8bit_adam \
  --learning_rate=${LEARNING_RATE} \
  --lr_scheduler=${LR_SCHEDULE} \
  --lr_warmup_steps=${LR_WARMUP_STEPS} \
  --num_class_images=${NUM_CLASS_IMAGES} \
  --max_train_steps=${MAX_NUM_STEPS} \
  --mixed_precision=${MIXED_PRECISION} \
  --checkpointing_steps=${CHECKPOINTING_STEPS} \
  --train_text_encoder \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --allow_tf32 \
  --resume_from_checkpoint=${RESUME_CHECKPOINT} \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a woman with ${2}"