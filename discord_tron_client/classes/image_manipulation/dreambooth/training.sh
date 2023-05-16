#!/bin/bash
export MODEL_NAME="$3"
export INSTANCE_DIR="${1}/datasets/images"
export CLASS_DIR="${1}/datasets/regularization"
export OUTPUT_DIR="${1}/datasets/models"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a woman with " \
  --class_prompt="digital art portrait of a person" \
  --resolution=768 \
  --train_batch_size=2 \
  --use_8bit_adam \
  --learning_rate=7e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=1500 \
  --max_train_steps=300 \
  --mixed_precision=bf16 \
  --allow_tf32 \
  --checkpointing_steps=10 \
  --resume_from_checkpoint="checkpoint-500"
#--train_text_encoder \