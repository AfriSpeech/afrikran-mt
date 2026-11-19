#!/usr/bin/env bash
set -euo pipefail

DIR="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/aligner"
VENV="/mnt/volume_d2wey28/projects/nkrane-africa-gemma/.venv"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=== Resuming Pan-African awesome-align to Full Convergence ==="
echo "Target: 4 Total Epochs (~20,500 Steps) with Evaluation"
echo "Model: $DIR/model"
echo "Date: $(date)"

nohup "$VENV/bin/awesome-train" \
  --output_dir "$DIR/model" \
  --model_name_or_path "$DIR/model" \
  --should_continue \
  --overwrite_output_dir \
  --extraction softmax \
  --do_train \
  --do_eval \
  --train_tlm \
  --train_so \
  --train_psi \
  --train_data_file "$DIR/train_align.txt" \
  --eval_data_file "$DIR/eval_align.txt" \
  --per_gpu_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1.5e-5 \
  --num_train_epochs 4 \
  --logging_steps 50 \
  --save_steps 2500 \
  --cache_data \
  >> "$DIR/train_aligner.log" 2>&1 &

PID=$!
echo "awesome-align resumed successfully with PID $PID"
echo "Log: $DIR/train_aligner.log"
