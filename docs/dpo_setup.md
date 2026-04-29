# Preference Optimization (DPO / GRPO) Setup

For preference optimization, we split the workload between **Modal** (for scalable generation) and the **AMD MI300X** (for the memory-heavy DPO/GRPO training).

## 1. Candidate Generation on Modal

Modal is perfect for serverless scale-out inference.

1. **Install Modal locally:**
```bash
pip install modal
modal setup
```

2. **Upload SFT Model to Modal Volume:**
```bash
modal volume create tinymath-models
modal volume put tinymath-models ./sft_output/final /sft-model
```

3. **Run Candidate Generation:**
Use the `src/dpo/generate_preferences.py` script packaged in a Modal app to rapidly generate candidate solutions for your math problems. Then download the resulting `dpo_dataset` back to your AMD instance.

## 2. Training on AMD MI300X

DPO requires running two models simultaneously (policy and reference). The MI300X's 192GB VRAM is perfect for this, allowing you to avoid slow CPU-offloading.

```bash
# SSH into your AMD MI300X instance
source ~/sft_env/bin/activate

# For DPO:
python src/dpo/train_dpo.py --model_path ./sft_output/final --dataset_path ./dpo_dataset

# For GRPO (DeepSeek-R1 style):
python src/dpo/train_grpo.py --model_path ./sft_output/final
```
