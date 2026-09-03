# TinyMathReason-1B — X/Twitter thread

---

**1/**
I spent 25 days building a 1.12B math reasoning model totally from scratch.

Tokenizer. 57B tokens of pretraining. TPU. Checkpoint conversion. SFT. GRPO. The whole pipeline.

Final GSM8K: 2.2%. GPT-4 gets ~95%.

Here's everything I learned, including why it scored so low 👇

---

**2/**
The architecture is just Llama-2 in a small coat:

- 22 layers, hidden 2048
- GQA 4:1 (16 Q heads, 4 KV)
- SwiGLU, RoPE, 4096 ctx
- 128-dim heads (Llama-3 style for richer features)
- Custom 32k BPE
- 1,123,117,056 params exactly

I wanted architecture to NOT be the variable I had to debug.

---

**3/**
Phase 1: data.

Two Vultr CPU boxes ran the pipeline in parallel:
- FineWeb-Edu (~50%)
- OpenWebMath (~15%)
- MathPile (~15%)
- Stack-Edu (~10%)

Total: ~57B tokens uploaded to GCS.

Going back through the code six weeks later, I realized I'd left a lot of stuff broken.

---

**4/**
Stuff I claimed in the README but the code didn't actually do:

- "MinHash dedup" was just hash(text) exact dedup
- Proof-Pile-2 was in the mix config at 15% but I never added it to the download script
- "Quality filtering" was len(words) > 20

Audit your own pipeline. Six weeks later.

---

**5/**
Phase 2: pretraining.

Hardware: Google Cloud TPU v4-64 (64 chips!)
Framework: MaxText (JAX/Flax)
Steps: 54,362
Optimizer: AdamW, lr 3e-4 cosine, bf16
Throughput: ~8,900 tok/sec/chip, ~66 TFLOP/s/device

Loss converged cleanly. No NaNs. No gradient explosions. The pretraining itself was the calm part.

---

**6/**
Things in pretraining that ate days I'll never get back:

🔥 "Zero-layers" bug: scan_layers=True silently dropped all 22 transformer layers from the checkpoint. Lost a week.

🔥 XLA host RAM OOM: 1.1B model wants >300GB host RAM to compile HLO graph with batch=8

🔥 jax.Ref removed in 0.6.2

🔥 SSH 255 across 8 TPU host VMs

---

**7/**
Phase 3: checkpoint conversion.

I budgeted a day. It took five. MaxText → HF safetensors required fixing five separate bugs:

1. Query scaling baked INTO the weights
2. GQA tensor shapes differ
3. RoPE interleaving order differs
4. Vocab padding mismatch
5. Invalid tokenizer class name (silent fail)

Write the inspector before the converter. Trust me.

---

**8/**
Phase 4: SFT on AMD MI300X (192GB).

Two-stage:
- Alpaca 52k (instruction following)
- GSM8K + MathInstruct + MetaMathQA = 662k (math + reasoning with <think> tags)

Result: model learned to PRODUCE <think> blocks ✅
But couldn't fill them with correct math ❌

The format got learned. The reasoning didn't.

---

**9/**
SFT bugs I shipped:

❌ No completion-only loss masking → 50-70% of gradients went to memorizing the system prompt
❌ Random init for new tokens → ARC-Easy crashed 29.9% → 25.5%
❌ Heuristic CoT extractor put answers INSIDE <think> blocks half the time on MathInstruct
❌ 2 epochs (research says small models need 5-10)
❌ Curriculum was backwards (chat first, math second)
❌ lr too high (2e-5 on a base with near-zero math prior)

---

**10/**
Phase 5: GRPO.

Why not DPO? With 1% correctness you can't generate preference pairs. Everything is "rejected."

GRPO config:
- GSM8K train only (7,473 problems)
- 22,419 steps, 3 epochs
- G=8 group size
- β=0.01 KL
- math_verify (SymPy) for AST-based correctness
- 3-gram repetition penalty

Result: GSM8K 1.0% → 2.2% 🚀

---

**11/**
The math of why GRPO didn't move the needle further:

With 1.4% empirical correctness and G=8:
P(group has ≥1 correct rollout) = 1 - (0.986)^8 ≈ 10.6%

→ 89% of training steps had ZERO correctness gradient
→ Format reward dominated by 35×
→ Model learned to produce valid <think> shells with garbage inside

DeepSeek-R1-Zero observation: RL needs a non-zero base.

---

**12/**
GRPO infrastructure adventure:

GSM8K, 3 GPUs:
- MI300X #1: steps 0→12,500, quota exhausted, clean SIGINT, pushed ckpt to HF Hub
- MI300X #2: steps 12,500→19,000, vLLM/ROCm version conflicts
- GCP 2× NVIDIA L4: steps 19,000→22,419, ROCm→CUDA hop

Checkpoint survived a 3-machine relay. 12,500 steps of optimizer state preserved.

---

**13/**
📈 Final benchmarks:

| Bench | Base | SFT | GRPO |
|-------|------|-----|------|
| GSM8K | 1.0% | 1.0% | 2.2% |
| Minerva | 0.0% | 0.0% | 2.0% |
| ARC-E | 29.9% | 25.5% | 28.8% |
| MMLU | 23.5% | 24.6% | 23.6% |

Math doubled (from 1%). General knowledge preserved within KL budget. About what the math predicted.

---

**14/**
The ONE number that explains why scores are low:

Tokens per parameter:
- TinyMathReason: 50
- TinyLlama: 2,700
- Llama-3.2-1B: 7,500
- Qwen2.5-1.5B: 12,000

My model saw 54× FEWER tokens than TinyLlama. It literally has not seen enough math to memorize basic arithmetic, let alone compose reasoning.

This explains ~80% of the gap.

---

**15/**
After everything was done, I audited my tokenizer:

"100"     → ["100"]
"50"      → ["50"]
"2024"    → ["20", "24"]
"1234567" → ["12", "345", "67"]
"382391"  → ["38", "23", "91"]

Frequent numbers = 1 token. Less common = unpredictable 2/3-digit chunks. Llama-3 splits every digit. I should have.

Hard arithmetic ceiling.

---

**16/**
What actually went RIGHT:

✅ Full E2E pipeline runs (tokenizer → pretrain → SFT → RL → eval)
✅ Zero mode collapse in 22,419 GRPO steps
✅ Stable training across THREE hardware platforms
✅ Checkpoint conversion infra is reusable
✅ Every bug is documented

If you're learning how LLMs are built, the failures are the value.

---

**17/**
What went WRONG (cliff notes):

- Token budget 50× too low
- Tokenizer inconsistently merges multi-digit numbers
- No completion-only loss masking (50-70% wasted gradients!)
- Additive rewards let format hack correctness by 35×
- Proof-Pile-2 was in the mix config but never downloaded 🤦
- Only 2 SFT epochs

Each one is a known thing. I just learned them the hard way.

---

**18/**
Lessons I'd tattoo on my hand:

1. Token budget beats every clever trick. Get to 1k+ tok/param before anything else.
2. Audit your tokenizer BEFORE training. Multi-digit merges = arithmetic ceiling.
3. Loss masking isn't optional. One flag = 30-50% gradient efficiency.
4. RL needs a non-zero base. 1% correctness = 89% noise.
5. Gate rewards, don't sum them.

---

**19/**
The 5-day plan to actually fix this (1 MI300X, ~$560 spot):

Day 1: tokenizer audit + distill 30k GSM8K solutions with Qwen2.5-Math-7B
Day 2-3: redo SFT (NuminaMath, loss masking, mean-init, 5 epochs)
Day 4: rejection-sampling fine-tuning
Day 5: GRPO with gated reward + multi-dataset

Target: GSM8K 12-20%. TinyLlama-Math territory.

---

**20/**
Full optimization roadmap (anchored to published reference models):

Current:                  GSM8K 2.2%
+ Eval fixes:           → 3-8%
+ Better SFT:           → 8-15%
+ Continual pretrain:   → 18-30%
+ Better GRPO:          → 32-53%
+ Test-time scaling:    → 38-62%

Central case is Rho-Math-1B / SmolTulu territory. Real 1B math model.

---

**21/**
💰 Cost breakdown:

- Data processing (Vultr CPUs): ~$15
- TPU v4-64 pretraining: ~$2-4k spot
- SFT (MI300X): ~$30
- GRPO (MI300X + L4s): ~$15
- Evaluation: ~$5

Pretraining was ~99% of total spend. Post-training is cheap if you have a base worth post-training.

---

**22/**
Bottom line for researchers:

At the 1B scale, the single biggest lever for math is continual pretraining on math-dense data.

Rho-Math-1B: +15B tokens → MATH 0% → 15.6%
InfiR-1B: +940B tokens → GSM8K 8% → 63%

SFT/RL on top of an undertrained base is polishing a rough stone. The lever is upstream.

---

**23/**
📦 Everything is open:

- Repo: github.com/himanshu-nakrani/TinyMathReason-1B
- SFT: huggingface.co/himanshunakrani9/TinyMathReason-1B-sft
- GRPO: huggingface.co/himanshunakrani9/TinyMathReason-1B-grpo

Every script. Every config. Every bug. Every training log. Every "this didn't work" note.

---

**24/**
The math scores are low. The project still taught me more about how LLMs actually work than two years of paper-reading.

Most repos ship weights. This one ships the full journey, failures included.

If you're trying to actually understand LLMs (not just use them), this is the one.

🧮 fin

---

*Posting notes: ~24 tweets total. Optimal spacing: 30-90 seconds between tweets. Pin tweet 1 with the repo link. Best window: weekday 9-11 AM PST. Tweet 11 (the GRPO math) and tweet 15 (the tokenizer audit) are the most quotable; consider screenshots there.*
