#!/bin/bash
# Setup script for TinyMathReason-1B on GCP TPU VM

set -e

echo "Starting TPU VM Setup..."

# 1. Update and install basic tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-pip git htop tmux zstd

# 2. Clone MaxText
if [ ! -d "maxtext" ]; then
    echo "Cloning MaxText..."
    git clone https://github.com/google/maxtext.git
fi

# 3. Setup MaxText dependencies
cd maxtext
bash setup.sh

# 4. Return to home and clone our repo if not present
cd ~
if [ ! -d "TinyMathReason-1B" ]; then
    echo "Cloning TinyMathReason-1B..."
    # Replace with the actual repo URL if needed
    git clone https://github.com/your-username/TinyMathReason-1B.git
fi

# 5. Configure MaxText
echo "Configuring MaxText..."
mkdir -p ~/maxtext/MaxText/configs/
cp ~/TinyMathReason-1B/src/train/maxtext_config.yml ~/maxtext/MaxText/configs/tinymath_1b.yml

echo "Setup Complete!"
echo "To start a smoke test:"
echo "cd ~/maxtext && python3 MaxText/train.py MaxText/configs/tinymath_1b.yml run_name=tinymath_smoke_test steps=1000"
