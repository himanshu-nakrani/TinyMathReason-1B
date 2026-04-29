# Supervised Fine-Tuning (SFT) Setup

This phase transitions from TPU (MaxText) to GPU (PyTorch/TRL). We use the **AMD MI300X (192GB VRAM)** via AMD Cloud.

## 1. Environment Setup

Once connected to your AMD MI300X instance:

```bash
# 1. Update and install dependencies
sudo apt update
sudo apt install -y git git-lfs

# 2. Setup Python environment
python3 -m venv ~/sft_env
source ~/sft_env/bin/activate

# 3. Install PyTorch with ROCm support for AMD GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers datasets trl peft accelerate deepspeed wandb pyyaml
```

## 2. Checkpoint Conversion

Before training, we must convert the MaxText checkpoint to HuggingFace format:

```bash
# In the project root
python src/train/convert_checkpoint.py \
    --orbax_dir /path/to/orbax/checkpoint \
    --hf_out_dir ./hf_checkpoints/tinymath-1b-base \
    --tokenizer_path ./tokenizer
```

## 3. Data Preparation

Format the datasets into ChatML:

```bash
cd src/sft
python prepare_sft_data.py --output_dir ./sft_data
```

## 4. Run SFT Training

With the massive 192GB memory of the MI300X, you can likely run SFT without deepspeed, or with a very aggressive DeepSpeed config to maximize throughput.

Run training with Accelerate:

```bash
# Initialize wandb
wandb login

# Launch
accelerate launch \
    train_sft.py --config sft_config.yaml
```

## Expected SFT Metrics
- **Dataset**: ~600k examples
- **Hardware**: 1x AMD MI300X 192GB
- **Time**: Due to the massive batch size capability, expect this to finish extremely fast.
