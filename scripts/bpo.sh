#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=1,3,4,5
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1  # Enable CUDA synchronous debug mode

# Set the log file path
DIRNAME="your/directory"
LOGFILE="./results/$DIRNAME/training.log"

# Ensure the log file directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Run command and log output to the log file
{
    echo "Starting training at $(date)"
    accelerate launch --config_file ./scripts/config_dpo.yaml \
        ./train_bpo.py \
        --do_train \
        --seed 42 \
        --config_train_path ./data/config_train.json \
        --keywords train \
        --dpo_beta 0.1 \
        --lora_enable False \
        --finetune_mm_projector True \
        --batch_size 1 \
        --policy_model_name_or_path liuhaotian/llava-v1.5-7b \
        --learning_rate 1e-6 \
        --warmup_steps 10 \
        --output_dir ./results/$DIRNAME \
        --total_epochs 1 \
        --evaluation_strategy "no" \
        --weight_decay 0.0 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --ddp_backend "nccl" \
        --bf16 True \
        --ddp_find_unused_parameters False \
        --max_grad_norm 1.0 \
        --clean_tokens_after_eos True \
        --temperature 1.0 \
        --model_max_length 768 \
        --image_folder directory/to/figures \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio 'pad'
    echo "Training completed at $(date)"
} | tee "$LOGFILE"