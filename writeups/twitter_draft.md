# I built TinyMathReason-1B — a 1.12B-parameter math reasoning model entirely from scratch

I completed the entire run in 40 days with zero out-of-pocket cost by stacking free developer and research cloud credits.

I wrote the custom tokenizer, built the data pipeline, pretrained on a TPU v4-64 mesh, SFT-aligned it, and finished with GRPO reinforcement learning.

It got 2.2% on GSM8K.

For context, GPT-4 gets ~95%. Peer 1B base models (like TinyLlama) get similar low single digits on a comparable compute budget. This is a technical post-mortem on why it scored so low: the engineering issues I ran into, the math behind why my RL gradients collapsed, and the roadmap to fix it.

---

### The Pipeline and Architecture

I built a standard Llama-2 decoder-only lookalike to minimize architectural variables.

* Parameters: 1,123,117,056 parameters exactly (22 layers, hidden 2048)
* Intermediate SwiGLU MLP: 5632 dimension (computed as $8/3 \times 2048$, rounded to a 128-multiple for optimal compiler alignment on the TPU)
* Attention: GQA 4:1 (16 query heads, 4 KV heads) — shrinks the KV cache memory footprint 4×, allowing for larger batch sizes and higher generation throughput during reinforcement learning
* Head dim: 128 (Llama-3 style, capturing richer feature representations than TinyLlama's 64-dim heads)
* Context: 4096 tokens (RoPE θ=10000, RMSNorm ε=1e-5, untied embedding and output matrices in bfloat16)

The end-to-end pipeline rode across a hybrid stack of 5 different hardware platforms and 5 cloud providers:

1. **Tokenizer & Data:** Trained a custom 32k BPE tokenizer `(tokenizer.tiktoken)` from scratch. Rented two Vultr c2-standard-30 CPU instances in parallel to process a 57B-token corpus: Node A processed FineWeb-Edu and MathPile, while Node B streamed and cleaned OpenWebMath and Stack-Edu. Documents were concatenated with `EOS` tokens into packed 4096-token sequence shards and rsync-uploaded to Google Cloud Storage.
2. **Pretraining:** Trained on a GCP TPU v4-64 cluster (64 chips across 8 VM hosts) using the MaxText (JAX/Flax) distributed mesh framework, funded by the Google TPU Research Cloud (TRC) program. I ran a custom preemption handler in the background to poll the GCP metadata server for preemption alerts, gracefully triggering MaxText’s auto-checkpointing before spot node reboots.
3. **Checkpoint Conversion:** Built a custom conversion pipeline (`convert_checkpoint.py`) to map JAX Orbax checkpoints to HuggingFace safetensors. Because the JAX run had layers un-scanned, the attention kernels were stored as a single stacked array. I had to slice this layer-axis, undo MaxText's pre-baked query weight scaling ($1/\sqrt{d_{\text{head}}}$), and apply a permutation to the head dimension to bridge the differences in MaxText and HF RoPE interleaving.
4. **SFT:** Fine-tuned on an AMD MI300X (192GB VRAM) via TRL's `SFTTrainer` with a two-stage curriculum: Stage 1 trained on 52k Alpaca instructions for conversational priors, and Stage 2 resized the embeddings to add `<think>` and `</think>` tags, training on a 662k math reasoning mixture (MetaMathQA + MathInstruct + GSM8K) at $lr=2\text{e-}5$ for 2 epochs.
5. **Preference Optimization:** Ran Group Relative Policy Optimization (GRPO) using vLLM in colocate mode. To prevent VRAM collisions on a single GPU, I restricted the vLLM rollout engine to 30% memory utilization (`vllm_gpu_memory_utilization=0.3`) and enabled sleep mode (`vllm_enable_sleep_mode=True`) to swap vLLM parameters to host CPU RAM during PyTorch's backward passes. Training survived a 3-GPU cross-cloud relay (AMD MI300X #1 → MI300X #2 → 2× NVIDIA L4) by dynamically pushing checkpoints to the HuggingFace Hub.

---

### The Scores

General knowledge was preserved within the KL-divergence budget, while math scores successfully doubled (albeit from a very low base):

| Benchmark               | Base   | SFT    | GRPO   | Delta (Base → RL) |
|-------------------------|--------|--------|--------|-------------------|
| GSM8K (8-shot)          | 1.00%  | 1.00%  | 2.20%  | +1.20%            |
| Minerva Math (4-shot)   | 0.00%  | 0.00%  | 2.02%  | +2.02%            |
| ARC-Easy (0-shot)       | 29.90% | 25.51% | 28.79% | -1.11%            |
| ARC-Challenge (25-shot) | 21.70% | 24.66% | 22.78% | +1.08%            |
| HellaSwag (10-shot)     | 25.80% | 26.70% | 26.30% | +0.50%            |
| MMLU (5-shot)           | 23.50% | 24.60% | 23.62% | +0.12%            |

---

### Why the Scores Are Low

#### 1. The Pretraining Token Deficit

My model saw ~57B pretraining tokens. At 1.12B parameters, that equals 50 tokens per parameter. TinyLlama saw 2,700 tokens/param (54× more). Llama-3.2-1B saw 7,500 tokens/param (150× more). At 50 tokens/param, a 1B model simply hasn't seen enough tokens to memorize basic arithmetic identities, let alone compose reasoning chains.

#### 2. The Hostile Tokenizer

Standard BPE shatters less common numbers into unpredictable 2- and 3-digit chunks. For example, "382 + 491 = 873" tokenizes as ["38", "2", "Ġ+", "Ġ4", "91", "Ġ=", "Ġ8", "73"]. The number "382" tokenizes differently depending on context. A 1B model cannot learn consistent arithmetic rules when the inputs are constantly moving.

#### 3. SFT Formatting and Masking Pitfalls

SFTTrainer computed loss over the entire sequence. Because my prompts were structured, 50% to 70% of my gradients went toward memorizing the static system prompt rather than the reasoning steps. Additionally, resizing the vocabulary to add `<think>` and `</think>` tags used random initialization, perturbing the output-head distribution and causing the ARC-Easy regression.

#### 4. The Mathematics of GRPO Signal Collapse

With empirical base-model correctness at $p \approx 1.4\%$ and group size $G=8$:

$$P(\text{group has} \ge 1 \text{ correct rollout}) = 1 - (1 - 0.014)^8 \approx 10.6\%$$

For 89% of training steps, there was zero variance in correctness rewards across the group (all rollouts got 0.0). This meant zero correctness gradient.

The only signal that consistently varied was the formatting reward. Because my reward was additive (correctness + format), formatting dominated the gradient by ~35×. The model optimized for the easiest path: it became a "format hacker," producing beautifully formatted `<think>` blocks filled with empty space or math-sounding gibberish to collect format points without doing the math.

---

### The 7-Stage Optimization Roadmap

Unlocking real math capabilities on top of this pretraining base requires a systematic post-training overhaul. I designed this plan by running a rigorous post-mortem audit on my 40-day run. Specifically, I audited my tokenizer's representation of digits, calculated the statistical advantage decay of the GRPO group rollouts, and mapped the binding constraints at each stage of training. I then cross-referenced these constraints against published reference models in the 1B parameter class (like Microsoft's Rho-1, InfiR-1B, and SmolTulu) to compile exact, ROI-driven score projections.

Here is the stage-by-stage engineering roadmap:

* **Stage 0: Audit & Aligned Rebaseline** — Audit vocabulary splits to check number merging, align evaluation templates with ChatML 0-shot `gsm8k_cot`, and measure `pass@k` to identify baseline test-time scaling capability.
* **Stage 1: Math-Dense Continual Pretraining (CPT)** — Inject 30B tokens of math-dense web and proof data using Selective Language Modeling (SLM), adding document-boundary masks and QK-norm for training stability.
* **Stage 2: Clean SFT Redo** — Retrain on a high-quality SFT math blend with completion-only loss masking (`assistant_only_loss=True`), averaged embedding initialization for `<think>` and `</think>` tags, and a low learning rate ($lr=5\text{e-}5$) across 5 to 8 epochs.
* **Stage 3: Rejection-Sampling Fine-Tuning (RFT)** — Distill and filter 16 candidate reasoning paths per prompt from the Stage 2 model, fine-tuning for one additional epoch strictly on the verified-correct chains.
* **Stage 4: Redesigned GRPO RL** — Train on multiple math datasets with a gated reward formula ($\text{Format\_Pass} \times \text{Correctness}$) to prevent format hacking, using a larger group size ($G=16$) and longer context lengths to provide stable gradients.
* **Stage 5: Test-Time Scaling** — Scale inference capacity dynamically using self-consistency majority voting (`cons@32`) and a sibling 1B outcome verifier model trained on Math-Shepherd to rerank paths.
* **Stage 6: Tool Augmentation** — Train the model to write and execute sandbox-run Python blocks for intermediate arithmetic steps, bypassing calculation bottlenecks.

---

### Projected Scores After the Improvement Plan

Every projection is anchored directly to published reference models in the same parameter class (Rho-Math-1B, InfiR-1B, SmolTulu).

Single point estimates are avoided — each stage is bracketed by [Pessimistic / Central / Optimistic] bounds to reflect representation-capacity limits and the digit-merging tokenizer audit results.

| Stage                          | GSM8K (pess/cent/opt) | MATH (pess/cent/opt)  | MMLU (pess/cent/opt)  | Anchored Reference         |
|--------------------------------|-----------------------|-----------------------|-----------------------|----------------------------|
| Current state                  | 1.0% / 1.0% / 2.2%    | 0.0% / 0.0% / 2.0%    | 23.5% / 23.5% / 24.6% | Verified baselines         |
| 0. Audit & Aligned Rebaseline  | 1.0% / 3.0% / 5.0%    | 0.0% / 1.0% / 3.0%    | 23.5% / 24.0% / 25.0% | Protocol correction only   |
| 1. Continual Pretraining (30B) | 8.0% / 18.0% / 30.0%  | 4.0% / 10.0% / 16.0%  | 26.0% / 32.0% / 38.0% | Rho-Math-1B (15B SLM)      |
| 2. Correct SFT Redo            | 20.0% / 38.0% / 55.0% | 6.0% / 14.0% / 22.0%  | 28.0% / 35.0% / 42.0% | MetaMath SFT, SmolTulu     |
| 3. Rejection-Sampling (RFT)    | 28.0% / 47.0% / 62.0% | 8.0% / 17.0% / 26.0%  | 28.0% / 36.0% / 43.0% | Yuan et al. SFT multiplier |
| 4. Redesigned GRPO RL          | 32.0% / 53.0% / 68.0% | 9.0% / 19.0% / 28.0%  | 28.0% / 36.0% / 44.0% | SimpleRL math gains        |
| 5. Test-Time Scaling (cons@32) | 38.0% / 62.0% / 78.0% | 12.0% / 24.0% / 35.0% | 30.0% / 38.0% / 46.0% | Math-Shepherd ORM          |
| 6. Tool-Augmentation (Python)  | 40.0% / 64.0% / 80.0% | 18.0% / 35.0% / 50.0% | 30.0% / 38.0% / 46.0% | Rho-Math-Interpreter       |

At the final Stage 6 central estimate (GSM8K 64%, MATH 35%), TinyMathReason-1B enters the same league as the strongest open 1.5B math models in the world (such as Qwen2.5-Math-1.5B). The pessimistic case assumes the tokenizer digits are unaligned, while the optimistic case assumes digit splitting and Selective Language Modeling work at maximum efficiency.

---

### The $0 Cloud Credit Stack

I ran this entire pipeline using free developer and research allocations:

* Google TPU Research Cloud (@googlecloud): Free TPU v4-64 time (pretraining), and $30 monthly credits via Google One AI Premium (used for running GRPO experiments).
* AMD Developer Cloud (@AMD): $100 credit, used for MI300X (SFT & GRPO).
* Vultr trial (@Vultr): $300 credit, used for CPU data processing.
* Thunder Compute (@ThunderCompute): $25 trial credit, used for lightweight evaluations.
* Modal: $30 credit mostly, used for small experiments.

If you are an independent researcher, stack these.
