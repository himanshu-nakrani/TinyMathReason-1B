#!/bin/bash
set -euo pipefail

cd ~/TinyMathReason-1B
source venv/bin/activate

# Load environment variables from .env if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
fi

# Ensure HF_TOKEN and WANDB_API_KEY are set
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set. Please set it in your environment or in a .env file."
  exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Error: WANDB_API_KEY is not set. Please set it in your environment or in a .env file."
  exit 1
fi

echo "Logging into Hugging Face..."
hf auth login --token "$HF_TOKEN"

echo "Downloading SFT Stage-2 model..."
hf download himanshunakrani9/TinyMathReason-1B-sft --local-dir src/sft/sft_output/stage2/final

echo "Downloading GRPO checkpoint-19000..."
hf download himanshunakrani9/TinyMathReason-1B-grpo-checkpoints --local-dir ./outputs/grpo-full

echo "Starting training in the background..."
export WANDB_API_KEY
WANDB_MODE=offline \
nohup python -u src/dpo/train_grpo.py \
  --model_path src/sft/sft_output/stage2/final \
  --output_dir ./outputs/grpo-full \
  --num_train_epochs 3 \
  --resume_from_checkpoint > grpo_training.log 2>&1 &

echo "Training started! Check grpo_training.log for progress."
