#!/bin/bash
set -euo pipefail
echo "Installing python3-venv..."
sudo apt-get update
sudo apt-get install -y python3-venv
cd ~/TinyMathReason-1B
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch
pip install transformers trl peft accelerate deepspeed wandb pyyaml math-verify huggingface_hub
pip install --upgrade --force-reinstall huggingface_hub
pip install git+https://github.com/huggingface/datasets.git
echo "Setup complete!"
