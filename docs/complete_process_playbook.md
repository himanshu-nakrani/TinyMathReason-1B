# TinyMathReason-1B: Complete End-to-End Process Documentation & Engineering Playbook

**Author:** Himanshu Nakrani  
**Project:** TinyMathReason-1B — 1.12B-Parameter Decoder-Only Mathematical Reasoning Model Trained from Scratch  
**Status:** Completed (Base Pretraining ➔ SFT ➔ GRPO ➔ Full Evaluation)  
**Total Compute Cost:** $0 out-of-pocket (Google TPU Research Cloud, AMD Developer Cloud, Vultr trial, Modal, Lightning AI)

---

## Table of Contents

1. [Executive Overview & Guiding Philosophy](#1-executive-overview--guiding-philosophy)
2. [Model Architecture & Specifications](#2-model-architecture--specifications)
3. [Phase 1: Tokenizer Design & Training](#3-phase-1-tokenizer-design--training)
4. [Phase 2: Pretraining Data Curation Pipeline (~57B Tokens)](#4-phase-2-pretraining-data-curation-pipeline-57b-tokens)
5. [Phase 3: TPU Pretraining on Google Cloud TPU v4-64 (MaxText/JAX)](#5-phase-3-tpu-pretraining-on-google-cloud-tpu-v4-64-maxtextjax)
6. [Phase 4: Checkpoint Conversion (Orbax/JAX ➔ PyTorch Safetensors)](#6-phase-4-checkpoint-conversion-orbaxjax--pytorch-safetensors)
7. [Phase 5: Supervised Fine-Tuning (SFT) on AMD MI300X](#7-phase-5-supervised-fine-tuning-sft-on-amd-mi300x)
8. [Phase 6: Reinforcement Learning with GRPO on NVIDIA GPUs](#8-phase-6-reinforcement-learning-with-grpo-on-nvidia-gpus)
9. [Phase 7: Full Evaluation Suite & Benchmark Analysis](#9-phase-7-full-evaluation-suite--benchmark-analysis)
10. [Critical Engineering Hurdles & Bug Resolution (The "War Stories")](#10-critical-engineering-hurdles--bug-resolution-the-war-stories)
11. [Scientific Root-Cause Analysis (Why the Scores Are What They Are)](#11-scientific-root-cause-analysis-why-the-scores-are-what-they-are)
12. [Infrastructure & Financial Accounting](#12-infrastructure--financial-accounting)
13. [Future Scaling & Optimization Roadmap (Tiers 1 to 5)](#13-future-scaling--optimization-roadmap-tiers-1-to-5)
14. [Repository File Map & Key Artifacts](#14-repository-file-map--key-artifacts)

---

## 1. Executive Overview & Guiding Philosophy

The objective of **TinyMathReason-1B** was not to chase a state-of-the-art leaderboard spot, but to design, execute, and rigorously document the **full end-to-end LLM lifecycle from zero**. 

Most published open-source models only release final weights or a single training script. This project was constructed as a fully reproducible engineering artifact demonstrating:
- How to assemble and clean a math-heavy web corpus.
- How to train a custom BPE tokenizer.
- How to orchestrate a distributed TPU v4-64 cluster using Google's MaxText (JAX/NNX) framework.
- How to resolve memory, compilation, and tensor alignment hurdles converting distributed Orbax/Zarr checkpoints to Hugging Face safetensors.
- How to apply two-stage post-training SFT using ChatML formatting and explicit reasoning boundaries (`<think>...</think>`).
- How to implement Group Relative Policy Optimization (GRPO) with automated AST-based correctness verification and anti-loop mode collapse penalties.
- How to measure performance objectively across standardized benchmarks (GSM8K, Minerva Math, MMLU, ARC, HellaSwag).

```text
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ 1. Tokenizer    │ ──▶ │ 2. Data Pipeline │ ──▶ │ 3. Pretraining   │ ──▶ │ 4. Checkpoint Conv │
│ 32k BPE         │     │ ~57B tokens      │     │ TPU v4-64        │     │ Orbax ➔ HF safeten │
│ tiktoken        │     │ 2x Vultr CPUs    │     │ 54k steps (JAX)  │     │ TensorStore / CPU  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────────┘
                                                                                     │
         ┌───────────────────────────────────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 5. SFT Stage    │ ──▶ │ 6. GRPO Stage    │ ──▶ │ 7. Evaluation    │
│ AMD MI300X      │     │ 22.4k steps      │     │ lm-eval-harness  │
│ 662k traces     │     │ NVIDIA L4        │     │ + custom math    │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 2. Model Architecture & Specifications

The architecture mirrors the proven structural proportions of **TinyLlama-1.1B** with key modern adjustments inspired by **LLaMA-3** (specifically, adopting a larger head dimension of 128 to capture richer variable bindings in multi-step mathematical derivations).

```text
                   TinyMathReason-1B
                   ═════════════════
                   1.12B parameters

       Input tokens → [Embedding 32,768 × 2048]
                              │
                              ▼
            ┌─────────────────────────────────┐
            │  Transformer Layer × 22         │
            │  ┌───────────────────────────┐  │
            │  │ RMSNorm (eps=1e-5)        │  │
            │  │ ↓                         │  │
            │  │ GQA (16 Q heads, 4 KV)    │  │ ◄── 4:1 ratio (4x KV cache compression)
            │  │ ↓ + residual              │  │
            │  │ RMSNorm (eps=1e-5)        │  │
            │  │ ↓                         │  │
            │  │ SwiGLU MLP (5632 dim)     │  │ ◄── 8/3 ratio, 128-byte TPU aligned
            │  │ ↓ + residual              │  │
            │  └───────────────────────────┘  │
            └─────────────────────────────────┘
                              │
                              ▼
                       [Final RMSNorm]
                              │
                              ▼
                    [LM Head 2048 × 32,768]
                              │
                              ▼
                        Output logits
```

### Architectural Details

| Parameter / Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Total Parameter Count** | **1,123,117,056** (~1.12B) | Optimal small-model tier balancing reasoning capacity with edge deployability |
| **Layers** | 22 | Established depth for 1B-class transformers |
| **Hidden Dimension ($d_{\text{model}}$)** | 2048 | Standard representation dimension |
| **MLP Intermediate Dimension** | 5632 (SwiGLU) | $\approx \frac{8}{3} \times d_{\text{model}}$, rounded to multiples of 128 for TPU XLA alignment |
| **Attention Architecture** | Grouped Query Attention (**GQA 4:1**) | 16 Query Heads, 4 Key-Value Heads; reduces KV cache by 75% at inference |
| **Head Dimension ($d_{\text{head}}$)** | **128** | LLaMA-3 standard (wider than TinyLlama's 64); richer multi-token math representations |
| **Context Length** | 4096 tokens | Sufficient context to support deep Chain-of-Thought (CoT) reasoning |
| **Vocabulary Size** | 32,000 active (padded to **32,768**) | Padded to exact power of 2 for FSDP / TPU matrix multiplication tiling |
| **Rotary Position Embedding (RoPE)** | $\theta = 10000.0$ | Standard frequency base |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Pre-normalization on attention and MLP blocks |
| **Weight Tying** | False | Untied token embeddings and LM head |
| **Native Precision** | `bfloat16` | Dynamic range stability on TPUs and modern GPUs |

### Parameter Allocation Math
```text
Token Embeddings (untied):      32,768 × 2048                =    67,108,864
Attention Blocks (22 layers):
  - Q Projection:              2048 × (16 × 128) = 4,194,304
  - K Projection:              2048 × (4 × 128)  = 1,048,576
  - V Projection:              2048 × (4 × 128)  = 1,048,576
  - Output Projection:         (16 × 128) × 2048 = 4,194,304
  - RMSNorm (Attn):            2048
  Subtotal per layer:          10,487,808 × 22 layers        =   230,731,776
MLP Blocks (22 layers):
  - Gate (W1) Projection:      2048 × 5632       = 11,534,336
  - Down (W2) Projection:      5632 × 2048       = 11,534,336
  - Up (W3) Projection:        2048 × 5632       = 11,534,336
  - RMSNorm (FFN):             2048
  Subtotal per layer:          34,605,056 × 22 layers        =   761,311,232
Final RMSNorm:                  2048                         =         2,048
LM Output Head (untied):        2048 × 32,768                =    67,108,864
─────────────────────────────────────────────────────────────────────────────
Total Trainable Parameters:                                    1,126,262,784
(Exact unpadded parameter count:                               1,123,117,056)
```

---

## 3. Phase 1: Tokenizer Design & Training

- **Implementation:** [`src/data/train_tokenizer.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/data/train_tokenizer.py)
- **Output Artifacts:** `tokenizer.tiktoken`, `tokenizer/tokenizer_config.json`

### Design Decisions
1. **BPE Format:** Custom BPE trained using the `tiktoken` engine for speed and vocabulary efficiency.
2. **Vocabulary Size:** 32,000 active tokens, padded to 32,768 in the model configuration.
3. **Special Control Tokens:**
   - `<|im_start|>` (ID: 32000): ChatML message start.
   - `<|im_end|>` (ID: 32001): ChatML message termination.
   - `<think>` (ID: 32002): Beginning of reasoning trace.
   - `</think>` (ID: 32003): Conclusion of reasoning trace.

### The Multi-Digit Tokenization Defect (Critical Post-Mortem)
An audit conducted after pretraining revealed that the custom BPE chunks multi-digit numbers inconsistently:
```text
"100"     ──▶ ["100"]                     (1 token)
"2024"    ──▶ ["20", "24"]                (2 tokens)
"1234567" ──▶ ["12", "345", "67"]         (3 arbitrary tokens)
"382 + 491 = 873" ──▶ ["38", "2", " +", " 4", "91", " =", " 8", "73"]
```
*Impact:* Unlike modern math models (LLaMA-3, DeepSeek-Math) which enforce **per-digit splitting** (every digit `0-9` is an isolated token), arbitrary BPE chunking forces the model to memorize addition tables across multi-digit combinations rather than learning composable column-wise arithmetic. This established a hard intrinsic ceiling on pure calculation accuracy.

---

## 4. Phase 2: Pretraining Data Curation Pipeline (~57B Tokens)

- **Directory:** [`src/data/pipeline/`](file:///Users/himanshu/Git/TinyMathReason-1B/src/data/pipeline/)
- **Compute:** 2× Vultr `c2-standard-30` Bare-Metal Compute Nodes ($150 trial credit).
- **Storage Target:** `gs://tinymath-reason-data-himanshu/pretraining-data/`

### Pipeline Flowchart
```text
┌─────────────────────────┐
│ a_download_datasets.py  │  Stream HuggingFace datasets to local scratch NVMe
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  b_clean_and_filter.py  │  word_count > 20, alpha_ratio > 0.3, MD5 hash deduplication
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    c_mix_datasets.py    │  Interleave according to defined sampling proportions
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ d_tokenize_and_pack.py  │  Tokenize with custom BPE, append EOS, pack into 4096-token rows
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   e_create_shards.py    │  Compress into ~50MB .jsonl.zst shards (~1000 shards total)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   f_upload_to_gcs.py    │  Parallel rsync / gsutil upload to Google Cloud Storage
└─────────────────────────┘
```

### Dataset Composition
| Dataset | Target Share | Actual Tokens | Role |
| :--- | :---: | :---: | :--- |
| **FineWeb-Edu** | 40% | ~10B | High-grade general educational web text (synthetic scoring $\ge 3$) |
| **OpenWebMath** | 35% | ~30B | LaTeX-extracted mathematical web pages, forums, and derivations |
| **MathPile** | 15% | ~9.5B | High-density textbook chapters, arXiv preprints, and Olympiad proofs |
| **Stack-Edu** | 10% | ~7.7B | Educational programming code and algorithmic logic |
| **Proof-Pile-2** | *15% (Planned)* | *0B* | *Omitted due to a download-script oversight; mixer automatically re-normalized ratios, concentrating OpenWebMath to 35%* |

**Total Processed Corpus:** **~57 Billion Tokens** partitioned across 1,629 compressed `.jsonl.zst` shards.

---

## 5. Phase 3: TPU Pretraining on Google Cloud TPU v4-64 (MaxText/JAX)

- **Configuration:** [`src/train/maxtext_config.yml`](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/maxtext_config.yml)
- **Launch Scripts:** [`scripts/setup_tpu.sh`](file:///Users/himanshu/Git/TinyMathReason-1B/scripts/setup_tpu.sh), [`scripts/provision_tpu.sh`](file:///Users/himanshu/Git/TinyMathReason-1B/scripts/provision_tpu.sh)
- **Hardware:** GCP TPU `v4-64` (64 TPU chips across 8 Host VMs in a 2D/3D toroidal mesh) via TPU Research Cloud (TRC).

### Pretraining Hyperparameters
```yaml
base_learning_rate: 3.0e-4
learning_rate_schedule_steps: 54363
cosine_decay: true
decay_to: 0.1
warmup_steps_fraction: 0.0066   # ~360 steps
opt_type: adamw
adam_b1: 0.9
adam_b2: 0.95
adam_eps: 1.0e-8
weight_decay: 0.1
global_batch_size_tokens: 262144 # 64 sequences × 4096 context length
per_device_batch_size: 2
max_target_length: 4096
pure_nnx_decoder: true
scan_layers: false
```

### Operational Metrics & Trajectory
- **Run Identifier:** `tinymath-1b-prod-run11`
- **Total Steps:** **54,362 steps**
- **Tokens Consumed:** **~57 Billion tokens** (~50 tokens per parameter)
- **Sustained Hardware Speed:** **~8,900 tokens/sec/chip** (~66 TFLOP/s per device)
- **Loss Convergence:** Began at $\approx 10.4$, achieved stable log-linear decay, finishing at $\approx 2.60$.
- **Final Checkpoint (Orbax):** `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run11/checkpoints/54362/`

---

## 6. Phase 4: Checkpoint Conversion (Orbax/JAX ➔ PyTorch Safetensors)

- **Converter:** [`src/train/convert_checkpoint.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/convert_checkpoint.py)
- **Inspector:** [`src/train/inspect_checkpoint.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py)
- **Output:** [`hf_1b_model/`](file:///Users/himanshu/Git/TinyMathReason-1B/hf_1b_model/) (2.1GB bfloat16 safetensors)

Converting from MaxText/Orbax to Hugging Face format was one of the most demanding engineering tasks of the project. Budgeted for one day, it required five days to resolve five mathematical and structural discrepancies:

```text
    MaxText / Orbax (JAX)                      HuggingFace Llama (PyTorch)
    ═════════════════════                      ═══════════════════════════
 1. Sharded Zarr on GCS         ──TensorStore──▶ Flat NumPy arrays on local CPU
 2. Padded Vocab (32,000)       ───────────────▶ Explicit vocab_size = 32,768
 3. Stacked Layer Dim [22, ...] ──Slice & Loop─▶ Individual model.layers.{i}
 4. Baked 1/√128 Q-scaling      ──× √128───────▶ Raw unscaled Q projection
 5. RoPE Half-Split Order       ──Interleave───▶ RoPE Complex-Interleaved Order
```

### Mathematical Fixes during Conversion
1. **Topology Bypass via TensorStore:** Standard Orbax loaders crashed on local CPUs because they attempted to initialize the TPU v4-64 device mesh. The converter used `tensorstore` to read raw Zarr/OCDBT parameter chunks directly from GCS into host memory.
2. **Reversing Query Scaling:** MaxText bakes $1/\sqrt{d_{\text{head}}} = 1/\sqrt{128} \approx 0.088388$ directly into the query weights at save time to eliminate runtime division during TPU attention kernels. PyTorch Hugging Face computes this division on-the-fly. The converter multiplied query weights by $\sqrt{128}$ to prevent double-scaling.
3. **RoPE Permutation Inversion:** MaxText groups rotary frequencies by real and imaginary halves:
   $$\mathbf{W}_{\text{MaxText}} = [re_0, re_1, \dots, re_{d/2-1}, im_0, im_1, \dots, im_{d/2-1}]$$
   PyTorch expects interleaved frequency pairs:
   $$\mathbf{W}_{\text{HF}} = [re_0, im_0, re_1, im_1, \dots, re_{d/2-1}, im_{d/2-1}]$$
   The converter reshaped $[d_{\text{model}}, \text{heads}, 2, d/2]$ and transposed to produce standard Hugging Face RoPE ordering.
4. **Tokenizer Config Standardization:** Fixed `tokenizer_config.json` by mapping invalid `"tokenizer_class": "TokenizersBackend"` to `"PreTrainedTokenizerFast"`.

---

## 7. Phase 5: Supervised Fine-Tuning (SFT) on AMD MI300X

- **Data Prep:** [`src/sft/prepare_sft_data.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/sft/prepare_sft_data.py)
- **Trainer:** [`src/sft/train_sft.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/sft/train_sft.py)
- **Hardware:** AMD Instinct MI300X (192GB VRAM) via AMD Developer Cloud.

### SFT Curriculum
- **Stage 1 (Conversational Prior):**
  - Dataset: 52,000 Alpaca samples.
  - Purpose: Teach the base model turn-taking and conversational ChatML syntax.
  - Hyperparameters: 2 epochs, learning rate $2 \times 10^{-5}$, effective batch size 32.
- **Stage 2 (Reasoning Traces):**
  - Dataset: ~662,000 mathematical problem-solution pairs (MetaMathQA ~395k, MathInstruct ~260k, GSM8K train ~7.5k).
  - Format: ChatML wrapping step-by-step thinking in explicit `<think>...</think>` tags.
  - Vocabulary Expansion: Token embeddings resized to accommodate `<think>` and `</think>`.

```text
<|im_start|>system
You are a mathematical reasoning assistant. Solve problems step by step inside <think> tags.
<|im_end|>
<|im_start|>user
What is 12 × 13?
<|im_end|>
<|im_start|>assistant
<think>
12 × 13 = 12 × (10 + 3) = 120 + 36 = 156.
</think>
156<|im_end|>
```

### SFT Post-Mortem & Learned Insights
1. **Prompt Loss Masking Absence:** The run used standard `SFTTrainer` on full text without `DataCollatorForCompletionOnlyLM`. Consequently, 50–70% of backpropagation gradients were expended fitting the static system prompt and user query tokens rather than optimizing reasoning steps.
2. **Embedding Initialization Shock:** When adding `<think>` and `</think>`, embeddings were initialized with small random normal noise. This transiently disrupted output head calibration, causing general reasoning benchmarks like **ARC-Easy** to regress from 29.9% to 25.5%. (Hewitt mean initialization should be used in future iterations).

---

## 8. Phase 6: Reinforcement Learning with GRPO on NVIDIA GPUs

- **Implementation:** [`src/dpo/train_grpo.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/dpo/train_grpo.py)
- **Report:** [`docs/grpo_report.md`](file:///Users/himanshu/Git/TinyMathReason-1B/docs/grpo_report.md)
- **Hardware:** GCP NVIDIA L4 (24GB VRAM).
- **Run Stats:** 22,419 total steps across 3 epochs of GSM8K train (7,473 prompts).

```text
Prompt: "What is 12 × 13?"
           │
           ▼  Generate G=8 completions concurrently
    ┌───────────────────────────────────────────────┐
    │ Rollout 1: ... <think>...</think> 156   (r=1.0)│
    │ Rollout 2: ... <think>...</think> 144   (r=0.0)│
    │ Rollout 3: ... <think>...</think> 156   (r=1.0)│
    │ Rollout 4: ... <think>...</think> 130   (r=0.0)│
    │ Rollout 5: ... <think>...</think> 156   (r=1.0)│
    │ Rollout 6: ... [Loop Collapse]          (r=-1.5)│
    │ Rollout 7: ... <think>...</think> 156   (r=1.0)│
    │ Rollout 8: ... <think>...</think> 156   (r=1.0)│
    └───────────────────────┬───────────────────────┘
                            │
                            ▼
          Normalize rewards across the 8 samples:
          A_i = (Reward_i - Mean_G) / Std_G
                            │
                            ▼
        Reinforce Rollouts 1, 3, 5, 7, 8 (Positive A)
        Penalize Rollouts 2, 4, 6 (Negative A)
```

### Hardened Multi-Objective Reward Stack
1. **AST Correctness Verification (`math_verify`):** Evaluates mathematical equivalence using abstract syntax tree parsing (handling LaTeX, fractions, and algebra), falling back to numeric regex extraction after `</think>`.
2. **Strict Regex Format Enforcement:** Returns 1.0 if the completion matches `^\s*<think>\s*\S.*?</think>\s*\S.*` (strict reasoning + answer layout), 0.5 for partial `<think>` containment (gradient stepping stone), and 0.0 otherwise.
3. **3-Gram Uniqueness Repetition Penalty:** Penalizes mode collapse loops. If the unique 3-gram ratio drops below 20%, a linear penalty up to -1.5 is applied.
4. **Explicit Stop Token Synchronization:** Appends `<|im_end|>` to generation stop tokens, preventing the model from simulating continuous multi-turn conversations in a single rollout.
5. **Tokenizer Decode Monkey-Patch:** Solved a critical TRL flaw where `skip_special_tokens=True` stripped `<think>` tags before passing text to reward functions. Monkey-patched `tokenizer.decode` to preserve `<think>` tags while stripping ChatML control tokens.

### GRPO Training Dynamics
- **Total Mean Reward:** 0.5139
- **Format Reward:** 0.50 (highly stable; format layout was thoroughly acquired)
- **Correctness Reward:** 0.01389 (~1.4% correctness rate in rollouts)
- **KL Divergence:** 2.975 (controlled drift from SFT policy)
- **Entropy:** 4.679 (stable token diversity maintained)

---

## 9. Phase 7: Full Evaluation Suite & Benchmark Analysis

Evaluations were performed using `lm-evaluation-harness` ([`src/eval/run_benchmarks.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/eval/run_benchmarks.py)) and standalone math scripts ([`src/eval/run_custom_eval.py`](file:///Users/himanshu/Git/TinyMathReason-1B/src/eval/run_custom_eval.py)).

### Complete Benchmark Matrix Across Stages

| Benchmark | Setting | Metric | Base Pretraining | Post-SFT | **Post-GRPO (Final)** | Stage Delta |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **GSM8K** | 8-shot (Template-aligned) | Flexible Match | 1.00% | 1.00% | **2.20%** 🚀 | **+120% relative gain** |
| **Minerva Math** | 4-shot | Math Verify | 0.00% | 0.00% | **2.02%** 🚀 | **+2.02% absolute gain** |
| **ARC-Challenge** | 25-shot | Accuracy Norm | 21.70% | **24.66%** | 22.78% | +1.08% vs Base |
| **ARC-Easy** | 0-shot | Accuracy Norm | **29.90%** | 25.51% | 28.79% | -1.11% vs Base |
| **HellaSwag** | 10-shot | Accuracy Norm | 25.80% | **26.70%** | 26.30% | +0.50% vs Base |
| **MMLU** | 5-shot | Accuracy | 23.50% | **24.60%** | 23.62% | +0.12% vs Base |

### Industry Comparison (1B-Parameter Weight Class)

| Model | Total Pretrain Tokens | Tokens / Parameter | GSM8K Base | GSM8K Post-Train | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TinyMathReason-1B** | **57 Billion** | **~50** | **1.0%** | **2.2%** | Trained from scratch, zero out-of-pocket cost |
| **TinyLlama-1.1B** | 3.0 Trillion | ~2,700 | 2.6% | ~5.8% | 54× more pretraining tokens |
| **Pythia-1.4B** | 300 Billion | ~214 | ~1.8% | ~4.5% | 5× more pretraining tokens |
| **Llama-3.2-1B** | ~9.0 Trillion | ~7,500 | 8.1% | ~44.0% | 150× more pretraining tokens |
| **SmolLM2-1.7B** | 11.0 Trillion | ~6,500 | ~15.0% | ~51.6% | Uses 28B synthetic Cosmopedia textbook tokens |

---

## 10. Critical Engineering Hurdles & Bug Resolution (The "War Stories")

### 1. The "Zero Layer" 0.134B Parameter Silent Failure
- **Symptom:** Runs 1–9 initialized normally and converged loss, but saved checkpoints were only ~270MB and contained only 13 keys (embeddings + LM head).
- **Cause:** MaxText's Linen-to-NNX wrapper had an issue where `scan_layers: True` caused `nn.scan` to silently fail to register the 22 transformer layers in the JAX PyTree during compiled execution.
- **Solution:** Configured `pure_nnx_decoder: True` and explicitly forced `scan_layers: False` in `src/train/maxtext_config.yml`.

### 2. Host RAM Out-Of-Memory during XLA Graph Compilation
- **Symptom:** TPU VMs crashed with SIGKILL (Exit code 137) during startup before training began.
- **Cause:** Compiling the XLA computation graph for 1.1B parameters with `per_device_batch_size: 8` required over 300GB of host system RAM (separate from TPU HBM).
- **Solution:** Reduced `per_device_batch_size: 2`, bringing host compilation memory requirements within VM hardware limits.

### 3. Google-Internal Dependency Mocking on TPU VMs
- **Symptom:** TPU training script threw `ModuleNotFoundError` importing `jax.experimental.pallas.ops.tpu.splash_attention` (an internal Google module not available in public JAX).
- **Solution:** Built a dynamic `MetaPathFinder` in `scripts/setup_tpu.sh` using a custom `types.ModuleType` subclass that served a dummy module with appropriate `__path__` and `__spec__` properties, satisfying Python 3.12's import machinery without brittle source edits.

### 4. Distributed TPU Worker Uptime Desynchronization
- **Symptom:** Spot TPU instances failed with SSH 255 or broken network connections during job launch.
- **Cause:** If one worker in a TPU v4-64 slice rebooted due to a transient failure, its uptime diverged from the other 7 nodes, preventing the JAX distributed mesh from forming.
- **Solution:** Built an orchestration loop in `src/train/preemption_handler.py` that executed a synchronized sequential reboot across all 8 host VMs to guarantee identical uptimes before initiating JAX mesh compilation.

### 5. TRL GRPO Special Token Stripping & Format Invalidation
- **Symptom:** The GRPO training loop reported format rewards of 0.0 despite rollouts visibly generating reasoning traces.
- **Cause:** TRL's `GRPOTrainer` invoked `tokenizer.decode(..., skip_special_tokens=True)`, stripping `<think>` and `</think>` before passing text to the reward functions.
- **Solution:** Monkey-patched `tokenizer.decode` inside `src/dpo/train_grpo.py` to preserve `<think>` tags while stripping ChatML boundary tokens (`<|im_start|>`, `<|im_end|>`).

---

## 11. Scientific Root-Cause Analysis (Why the Scores Are What They Are)

An honest analysis of why GSM8K scored 2.2%:

```text
                   PRETRAIN CEILING (57B Tokens, ~50 tok/param)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   Tokenizer Digit Merging                             Data Hygiene & Packing
("1234567" -> 3 arbitrary chunks;                  (Cross-document attention leakage;
 arithmetic unlearnable)                            no document boundary masking)
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        SFT DISTORTION & GRADIENT DILUTION
            (No completion-only loss masking: ~60% gradient wasted on prompt;
             random embedding init regressed ARC-Easy by 4.4%)
                                      │
                                      ▼
                        GRPO REWARD VARIANCE COLLAPSE
            (~1.4% correctness rate -> ~89% of groups have zero reward variance;
             format reward saturated at 0.50, dominating gradient by 35x)
```

1. **The Pretraining Budget Constraint:** Pretraining on 57B tokens equates to ~50 tokens/param. Empirical scaling literature shows small models require **1,000 to 10,000 tokens/param** to develop robust reasoning circuits.
2. **Tokenizer Digit Splitting Absence:** Numbers were merged into arbitrary chunks, preventing linear composition of arithmetic operations.
3. **GRPO Reward Variance Collapse:** GRPO computes advantage by normalizing rewards across $G=8$ rollouts. With a ~1.4% base correctness rate:
   $$P(\text{at least 1 correct rollout in } G=8) = 1 - (1 - 0.014)^8 \approx 10.6\%$$
   Thus, **~89.4% of training steps had zero reward variance for correctness** (all 8 rollouts failed). The format reward (which saturated at 0.50) dominated the gradient by roughly 35×, reinforcing the structural `<think>` wrapper rather than deep calculation logic.

---

## 12. Infrastructure & Financial Accounting

Every stage of TinyMathReason-1B was completed using research grant allocations and trial credits, resulting in **$0 out-of-pocket expenses**:

| Pipeline Stage | Infrastructure Platform | Hardware Allocated | Funding / Grant Mechanism | Commercial Value Equivalent |
| :--- | :--- | :--- | :--- | :--- |
| **Local Dev & Tokenizer** | Local Host | Apple M-Series Silicon | Self-hosted | — |
| **Data Download & Prep** | Vultr Bare Metal | 2× `c2-standard-30` (60 vCPUs, NVMe) | Vultr $150 Trial Credit | ~$150 |
| **Pretraining (Run 11)** | Google Cloud Platform | TPU `v4-64` Pod Slice (64 chips) | TPU Research Cloud (TRC) Grant | ~$15,000 (Spot) / ~$50,000 (On-Demand) |
| **Checkpoint Conversion**| GCP VM / Local CPU | High-Memory Compute Instance | Google Cloud Free Credits | ~$30 |
| **Supervised Fine-Tuning**| AMD Developer Cloud | 1× AMD Instinct MI300X (192GB VRAM) | AMD Developer Cloud Grant | ~$1,200 |
| **GRPO Reinforcement** | GCP Compute Engine | 1× NVIDIA L4 GPU (24GB VRAM) | Google Cloud Research Credits | ~$180 |
| **Benchmark Evaluations** | Lightning AI / Thunder | NVIDIA A100 / L4 Instances | Trial Credits | ~$75 |
| **Total Out-of-Pocket** | — | — | **All Grants & Credits** | **$0.00** |

---

## 13. Future Scaling & Optimization Roadmap (Tiers 1 to 5)

Based on [`CRITIQUE.md`](file:///Users/himanshu/Git/TinyMathReason-1B/CRITIQUE.md) and [`opt_olan.md`](file:///Users/himanshu/Git/TinyMathReason-1B/opt_olan.md), here is the prioritized roadmap to scale this model's capabilities:

```text
Tier 1: Protocol & Inference Fixes (Hours, $0)
├── Apply native ChatML chat template to lm-eval-harness
└── Test-time self-consistency (cons@8 to cons@32 majority voting)
    Expected GSM8K lift: 2.2% ➔ 5.0%–8.0%

Tier 2: SFT Overhaul (1–3 days, 1x MI300X)
├── Enable DataCollatorForCompletionOnlyLM (loss on assistant tokens only)
├── Mean-initialize token embeddings for new tokens (Hewitt init)
├── Train on 30k distilled long-CoT solutions from Qwen2.5-Math-7B-Instruct
└── Train for 5–7 epochs with lr=5e-6 and cosine decay
    Expected GSM8K lift: 8.0% ➔ 15.0%–20.0%

Tier 3: RL Redesign (2–4 days, 1x A100/MI300X)
├── Gated reward function: reward = format_pass * (1.0 if correct else -0.1)
├── Increase completion limit to max_new_tokens=2048 to prevent reasoning truncation
└── Rejection Sampling Fine-Tuning (RFT) prior to GRPO
    Expected GSM8K lift: 20.0% ➔ 25.0%–30.0%

Tier 4: Continual Pretraining (1–3 weeks, TPU/H100)
├── Re-tokenize with digit-split tokenizer (LLaMA-3 style)
├── Continual pretrain on +20B tokens (FineMath, OpenWebMath, Proof-Pile-2)
└── Apply Selective Language Modeling (SLM / Rho-1 style token filtering)
    Expected GSM8K lift: 35.0% ➔ 50.0%+

Tier 5: Test-Time Tool Augmentation (Days, $0)
└── Add a regex-based <python> calculator execution sandbox
    Expected GSM8K lift: +15.0%–25.0% absolute boost on arithmetic
```

---

## 14. Repository File Map & Key Artifacts

```text
TinyMathReason-1B/
├── README.md                         # Project overview, quickstart, and Hub links
├── STATUS.md                         # Milestone tracking and verified scores across stages
├── CRITIQUE.md                       # Comprehensive post-mortem and 5-tier technical critique
├── opt_olan.md                       # Optimization roadmap anchored to published 1B baselines
├── handoff.md                        # Technical state summary across phases
├── interview_prep_guide.md           # STAR-method technical interview narrative
├── Makefile                          # Unified build commands (data, pretrain, sft, dpo, eval)
├── docs/                             # Technical deep-dives
│   ├── architecture.md               # Model parameter math and design choices
│   ├── pretraining_setup.md          # TPU v4-64 setup and execution guide
│   ├── sft_setup.md                  # SFT data formatting and TRL configuration
│   ├── dpo_setup.md                  # Preference generation and DPO specifications
│   └── grpo_report.md                # Phase 4 GRPO training metrics and dynamic analysis
├── src/
│   ├── data/                         # Tokenizer and data processing pipeline
│   │   ├── train_tokenizer.py        # 32k BPE tokenizer trainer
│   │   └── pipeline/                 # 6-stage distributed data pipeline
│   │       ├── a_download_datasets.py
│   │       ├── b_clean_and_filter.py
│   │       ├── c_mix_datasets.py
│   │       ├── d_tokenize_and_pack.py
│   │       ├── e_create_shards.py
│   │       └── f_upload_to_gcs.py
│   ├── model/
│   │   └── modeling_tinymath.py      # Standalone PyTorch reference model
│   ├── train/                        # TPU pretraining and checkpoint conversion
│   │   ├── maxtext_config.yml        # MaxText JAX training config
│   │   ├── tinymath-1b.yml           # Architecture specification
│   │   ├── inspect_checkpoint.py     # Checkpoint PyTree inspector
│   │   ├── convert_checkpoint.py     # TensorStore JAX-to-PyTorch converter
│   │   └── preemption_handler.py     # Distributed worker reboot synchronization
│   ├── sft/                          # Supervised Fine-Tuning
│   │   ├── prepare_sft_data.py       # ChatML and <think> dataset formatting
│   │   └── train_sft.py              # TRL SFTTrainer script
│   ├── dpo/                          # Reinforcement Learning
│   │   ├── train_grpo.py             # Multi-objective GRPO training loop
│   │   └── train_dpo.py              # Direct Preference Optimization script
│   └── eval/                         # Benchmarking and Visualization
│       ├── run_benchmarks.py         # lm-evaluation-harness runner
│       ├── run_custom_eval.py        # ChatML-aligned mathematical evaluation
│       ├── generate_comparison.py    # Cross-model benchmark comparison
│       └── plot_training_curves.py   # Loss and reward visualization generator
├── scripts/                          # Cloud orchestration and bootstrap scripts
│   ├── setup_tpu.sh                  # Python 3.12 setup, JAX patches, MetaPathFinder
│   ├── provision_tpu.sh              # GCloud TPU VM creation script
│   ├── run_sft.sh                    # MI300X SFT launcher
│   └── upload_to_hub.py              # Automated Hugging Face Hub exporter
└── writeups/                         # Publication writeups
    ├── technical_report.md           # In-depth technical report
    ├── medium_part1_pretraining.md   # Article: Pretraining journey
    ├── medium_part2_posttraining.md  # Article: SFT and GRPO alignment
    └── twitter_thread.md             # Summary launch thread
```

---
*TinyMathReason-1B demonstrates that with rigorous systems engineering, distributed framework mastery, and creative resource orchestration, a complete modern LLM training stack can be built from scratch.*
