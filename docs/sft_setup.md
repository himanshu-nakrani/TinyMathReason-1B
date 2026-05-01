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

## 3. Two-Stage Curriculum Preparation

Based on the research "Can Tiny Language Models Reason?", we use a two-stage Supervised Fine-Tuning approach to prevent the model from forgetting how to talk.

**Stage 1: Conversational Prior**
First, we train on general instruction-following and chat formats *without* any chain-of-thought traces.
```bash
cd src/sft
python prepare_sft_data.py --stage 1 --output_dir ./sft_data/stage1_chat
```

**Stage 2: Reasoning Traces**
Next, we train on complex math datasets (MathInstruct, OpenThoughts) that utilize the `<think>` and `</think>` tags.
```bash
python prepare_sft_data.py --stage 2 --output_dir ./sft_data/stage2_reasoning
```

## 4. Run SFT Training (Two Stages)

**CRITICAL:** Before starting SFT, ensure your HuggingFace tokenizer config has been updated to include `<think>` and `</think>` as special tokens, and pass `--resize_token_embeddings` to your training script to add +2 to the vocabulary size.

```bash
# Initialize wandb
wandb login

# Stage 1: Teach it to talk
accelerate launch train_sft.py \
    --config sft_config_stage1.yaml \
    --dataset_path ./sft_data/stage1_chat \
    --output_dir ./hf_checkpoints/tinymath-1b-stage1

# Stage 2: Teach it to reason
accelerate launch train_sft.py \
    --config sft_config_stage2.yaml \
    --dataset_path ./sft_data/stage2_reasoning \
    --model_path ./hf_checkpoints/tinymath-1b-stage1 \
    --resize_token_embeddings \
    --output_dir ./hf_checkpoints/tinymath-1b-stage2
```

## Expected SFT Metrics
- **Hardware**: 1x AMD MI300X 192GB
- **Time**: Due to the massive batch size capability, expect this to finish extremely fast.
- **Stage 1 Target**: Model follows instructions and responds correctly without thinking.
- **Stage 2 Target**: Model naturally opens `<think>` tags and writes long chains of thought before answering.
