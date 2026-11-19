#!/usr/bin/env bash
set -euo pipefail

DIR="/mnt/volume_d2wey28/projects/nkrane-africa-gemma"
cd "$DIR"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=== Launching 5-Epoch Nkrane Pan-African Gemma-3-1B SFT Training ==="
echo "Target: 5 Full Epochs with Early Stopping (0.25 Epoch Patience)"
echo "Hugging Face Sync: AfriSpeech/nkrane-gemma-3-1b-all-africa"
echo "Host: $(hostname)"
echo "Time: $(date)"

nohup "$DIR/.venv/bin/python" train.py \
  --base-model "AfriSpeech/gemma-3-1b-translate-sft-best" \
  --tokenizer "$DIR/tokenizer" \
  --resume "$DIR/checkpoints/best" \
  --train-dir "$DIR/data/train" \
  --val-dir "$DIR/data/val" \
  --output-dir "$DIR/checkpoints" \
  --hf-repo "AfriSpeech/nkrane-gemma-3-1b-all-africa" \
  --batch-size 32 \
  --grad-accum 2 \
  --max-seq-len 256 \
  --lr 3e-5 \
  --warmup-steps 500 \
  --epochs 5.0 \
  --epoch-examples 32000000 \
  --patience-ratio 0.25 \
  --eval-every 2500 \
  --save-every 10000 \
  > "$DIR/train.log" 2>&1 &

PID=$!
echo "Training launched successfully in background with PID $PID"
echo "Log file: $DIR/train.log"
