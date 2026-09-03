# TinyMathReason-1B: End-to-End Technical Report

Building a 1.12B math reasoning model from scratch.

Author: Himanshu Nakrani. Last updated May 2026.

---

## 1. TL;DR

TinyMathReason-1B is a 1.12B-parameter Llama-style decoder model that I built solo, from tokenizer all the way to GRPO RL. The whole thing took about 25 days and rode across 5 different machines (my MacBook, two Vultr CPU boxes, a TPU v4-64 cluster, an AMD MI300X, and a pair of NVIDIA L4s).

The final scores aren't impressive. GSM8K sits at 2.2%. Minerva Math at 2%. The general-knowledge benchmarks (MMLU, ARC, HellaSwag) hover around the base-model levels, which for a model trained on 57B tokens is honestly fine.

But the goal was never to beat anyone's leaderboard. It was to actually do every stage of the pipeline, hit every bug, write every conversion script, and understand exactly why a 1B model with a 50-tokens-per-param budget does what it does.

This report covers everything: what I built, what broke, what I'd do differently, and where the real performance ceiling actually sits.

### Final benchmark numbers

| Benchmark | Base | SFT | GRPO |
|---|:---:|:---:|:---:|
| GSM8K (8-shot) | 1.00% | 1.00% | **2.20%** |
| Minerva Math (4-shot) | 0.00% | 0.00% | **2.02%** |
| ARC-Easy (0-shot) | 29.90% | 25.51% | 28.79% |
| ARC-Challenge (25-shot) | 21.70% | 24.66% | 22.78% |
| HellaSwag (10-shot) | 25.80% | 26.70% | 26.30% |
| MMLU (5-shot) | 23.50% | 24.60% | 23.62% |

GRPO did double the math score. Doubling 1% to get 2% is still 2%, but the trajectory was real.

---

## 2. Architecture

I went with a Llama-style decoder, basically because that's the architecture everyone has settled on and I didn't want my architecture choice to be the variable I had to debug.

Exact config:

| Component | Value | Why |
|---|---|---|
| Parameters | 1,123,117,056 | Matches TinyLlama's proven 1.1B shape |
| Layers | 22 | Same as TinyLlama; battle-tested |
| Hidden dim | 2048 | Standard for the param budget |
| Attention | GQA 4:1 (16 Q heads, 4 KV heads) | 4× smaller KV cache at inference |
| Head dim | 128 | Llama-3's pick; bigger heads, richer features |
| MLP | SwiGLU, intermediate 5632 | 8/3 × 2048, rounded to a 128-multiple for the TPU |
| Vocab | 32,000 (padded to 32,768) | Compact, leaves more capacity for the layers |
| Context | 4096 | Enough room for multi-step CoT |
| Precision | bfloat16 | TPU standard |
| Position encoding | RoPE (θ=10000) | Standard |
| Norm | RMSNorm (ε=1e-5) | Standard |

### How TinyMathReason stacks up

| Feature | TinyMathReason | TinyLlama-1.1B | Llama-3.2-1B | SmolLM2-1.7B |
|---|:---:|:---:|:---:|:---:|
| Params | 1.12B | 1.1B | 1.2B | 1.7B |
| Layers | 22 | 22 | 16 | 24 |
| Head dim | 128 | 64 | 64 | 64 |
| GQA | 4:1 | 8:1 | 4:1 | MHA |
| Pretrain tokens | **57B** | 3T | ~9T | 11T |
| Tokens/param | **~50** | ~2,700 | ~7,500 | ~6,500 |

That last row is the whole story. 50 tokens per parameter. TinyLlama got 54× more. Llama-3.2-1B got 150× more. Everything downstream is gated by this number.

### Parameter math

```
Embeddings (untied)           65,536,000   (32k × 2048)
Per-layer attention           10,485,760   (Q 4.19M + K 1.05M + V 1.05M + O 4.19M)
Per-layer MLP                 34,603,008   (gate + up + down, 11.53M each)
Per-layer norms                    4,096
× 22 layers                  992,043,008
Final norm                         2,048
LM head (untied)              65,536,000
                            ─────────────
Total                       1,123,117,056
```

The MLP eats roughly twice what attention does per layer. Standard Llama-shape.

---

## 3. Phase 1: data and tokenizer

### Tokenizer

Trained a 32k BPE on a math-heavy sample. Reserved `<think>` and `</think>` as special tokens for later SFT.

This was a mistake, and I have the receipts. After everything was done I ran a direct audit on the tokenizer that's baked into the HF model:

```
"100"     -> ["100"]
"50"      -> ["50"]
"2024"    -> ["20", "24"]
"1234567" -> ["12", "345", "67"]
"382391"  -> ["38", "23", "91"]
"382 + 491 = 873" -> ["38", "2", "Ġ+", "Ġ4", "91", "Ġ=", "Ġ8", "73"]
```

So common round numbers are single tokens, less common numbers shatter into 2- and 3-digit chunks at unpredictable boundaries. The boundary for "382" inside "382 + 491" is different from "382" inside "382391". That kind of inconsistency is the worst case for a small model trying to learn addition. Llama-3 and DeepSeek-Math both split every digit; I should have done the same.

### Data corpus

Two Vultr c2-standard-30 instances ran the data pipeline in parallel. Datasets:

| Dataset | Role | Mix |
|---|---|---|
| FineWeb-Edu | General educational text | ~50% |
| OpenWebMath | Math web content | ~15% |
| MathPile | Math textbooks/papers | ~15% |
| Stack-Edu | Educational code | ~10% |
| Proof-Pile-2 | Proofs/theorems | "15%" in config but **never downloaded** |

Yes, I wrote Proof-Pile-2 into the mix ratio in `c_mix_datasets.py` and somehow never added it to `a_download_datasets.py`. So the model never saw a single token of formal proofs. I didn't notice until I read the code six weeks later and traced through the pipeline.

### Pipeline shape

```
a_download → b_clean_filter → c_mix → d_tokenize_pack → e_shard → f_upload_to_gcs
```

All 57B tokens ended up in `gs://tinymath-reason-data-himanshu/pretraining-data/`.

### Things the docs claim that the code doesn't actually do

Going back through the pipeline code with fresh eyes, I found a bunch of stuff that I'd written README copy about but never actually implemented:

- "MinHash deduplication" — nope, only `hash(text)` exact-string dedup. Near-duplicates passed through.
- "Quality filtering" — `len(words) > 20` and `alpha_ratio > 0.3`. That's it. No perplexity filter, no learned classifier, no PII scrub.
- Sequence packing — straight concat with EOS into 4096-token chunks. No document-attention mask. Cross-document attention contamination is real, and it's worst for math because each problem needs to be self-contained.
- Curriculum / annealing — none. Single uniform mix. SmolLM2 and MiniCPM both anneal math-heavy at the end; I didn't.
- Synthetic data — none. Cosmopedia is what gave SmolLM2 its edge over TinyLlama. I had nothing.

In hindsight a chunk of the SFT and GRPO struggles trace back to these data-side compromises, not to anything that happened later.

---

## 4. Phase 2: TPU pretraining

### Setup

- Hardware: Google Cloud TPU v4-64 (64 chips, mostly spot)
- Framework: MaxText (JAX/Flax, Google's training stack)
- Storage: GCS, streamed at training time

### Training config

| Param | Value |
|---|---|
| Steps | 54,362 |
| Batch | 64 sequences × 4096 tokens = 262k tokens/step |
| Total tokens | ~57B |
| Optimizer | AdamW (lr=3e-4, cosine decay, β₁=0.9, β₂=0.95) |
| Warmup | 0.66% (~360 steps; should have been 1–2%) |
| Important flags | `pure_nnx_decoder: True`, `scan_layers: False` |
| Throughput | ~8,900 tokens/sec/chip, ~66 TFLOP/s/device |

### Bugs I'd rather not remember

**The zero-layers bug.** I lost a full week to this. Early runs were producing checkpoints with 0.134B parameters. Just the embeddings and the LM head. All 22 transformer layers were silently missing. MaxText's `scan_layers: True` path was failing to register the NNX blocks. Switching to `scan_layers: False` fixed it but slowed training enough that I had to stop the run at 57B tokens instead of pushing it longer.

**XLA host-RAM OOM.** Once the layers were back, the TPU VM started getting OOM-killed during XLA compilation. Not HBM, host RAM. Compiling a 1.1B model with per-device batch size 8 wanted >300GB of host RAM to build the HLO graph. Dropping to per-device batch size 2 fixed it.

**JAX 0.6.2 breaking changes.** JAX removed `jax.Ref` and MaxText was still using it. I added a shim in `setup_tpu.sh` that aliases `jax.Ref` to `typing.Any`. Also had to `sed`-patch MaxText's ragged attention kernels directly. Not my finest hour.

**Multi-worker SSH choreography.** v4-64 is 8 host VMs, each driving 8 chips, all of which need to be in sync. If one host reboots and the others don't, the JAX distributed mesh collapses with "SSH 255" or "Connection refused" and you have to do a rolling reboot of all 8 and wait 5 minutes for them to sync. I have a Makefile target for it now.

**Pallas mock injector.** MaxText imports `jax.experimental.pallas.ops.tpu.splash_attention` which doesn't exist in public JAX. Python 3.12's strict import system wouldn't let me just stub it out. I ended up writing a real `MetaPathFinder` subclass that spoofs `__spec__` and `__path__` so the imports succeed but the actual functions never get called.

### What came out

After all that, the final run (`tinymath-1b-prod-run11`) finished cleanly. Loss converged. Final checkpoint at `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run11/checkpoints/54362/`.

---

## 5. Checkpoint conversion

Converting an Orbax/JAX checkpoint to HuggingFace safetensors is one of those things that sounds like a 10-line script and is actually 5 days of work. I had to rewrite `convert_checkpoint.py` from scratch and fix 5 separate bugs:

1. **Vocab size.** MaxText pads vocab to 32,768 for FSDP alignment. HF needs that mapping spelled out. Off-by-one here gives you garbage outputs.
2. **Stacked layers.** MaxText stores all 22 layers' attention weights as one tensor with shape `[num_heads, layer_idx, head_dim, hidden_dim]`. My first script assumed per-layer keys (`layers_0`, `layers_1`, ...) which is the structure when `scan_layers: True` produces actual checkpoints — but I'd disabled scan, so the storage was different. Had to slice along the layer axis.
3. **Query scaling.** MaxText bakes `1/√d_head` into the Q projection at save time. HF expects to apply it at runtime. So you have to multiply the Q weights by `√d_head` during conversion to undo the bake-in.
4. **RoPE interleaving.** MaxText uses one interleaving order for Q/K rotary embeddings, HF uses another. You need a permutation on the head-dim axis. Get this wrong and the model generates fluent-looking text that means nothing.
5. **Tokenizer config.** `tokenizer_config.json` had `"tokenizer_class": "TokenizersBackend"` which isn't a real class. AutoTokenizer fails silently. Fix is to set it to `PreTrainedTokenizerFast`.

I also wrote `inspect_checkpoint.py` which just dumps the entire Orbax PyTree with tensor shapes. That tool saved me from a lot of blind debugging. If you're ever doing this conversion, write the inspector first.

Verification was a forward pass on my MacBook in bfloat16, just to confirm shapes and outputs were sane. Took some memory tuning to not OOM during the verification — ended up reorganizing the script to save the safetensors first, then do the verification pass second.

---

## 6. Phase 3: SFT

Once the HF checkpoint was good, I moved to SFT on an AMD MI300X (192GB VRAM, plenty of headroom). Used HuggingFace Transformers + TRL's SFTTrainer.

### Two-stage curriculum (in retrospect, the wrong order)

**Stage 1 — conversational prior.** Trained on `tatsu-lab/alpaca` (~52k examples). Goal was to give the base model some sense of "respond when prompted." 2 epochs, lr=2e-5, no CoT, no special tokens.

**Stage 2 — reasoning traces.** Added `<think>` and `</think>` to the tokenizer, resized embeddings, and trained on GSM8K + MathInstruct + MetaMathQA (~662k examples). 2 epochs, lr=2e-5, ChatML template. I wrote a heuristic CoT extractor that wrapped the reasoning step in `<think>` tags.

### Results

| Benchmark | Base → SFT | Delta |
|---|---|---|
| GSM8K (8-shot) | 1.00% → 1.00% | 0 (eval mismatch though; 1% with aligned ChatML template) |
| MATH (4-shot) | 0.00% → 0.00% | 0 |
| ARC-Easy | 29.90% → 25.51% | **−4.39%** (regression) |
| ARC-Challenge (25-shot) | 21.70% → 24.66% | +2.96% |
| HellaSwag | 25.80% → 26.70% | +0.90% |
| MMLU (5-shot) | 23.50% → 24.60% | +1.10% |

The ARC-Easy regression is the interesting one. It's a well-known symptom of resizing token embeddings without doing a proper init for the new rows.

### Where the SFT went wrong

Walking through what I actually shipped:

**No completion-only loss masking.** SFTTrainer with `dataset_text_field="text"` computes loss over the whole sequence by default. System prompt, user question, assistant answer — all of it. For my data, that's something like 50–70% of the gradient being wasted on memorizing the system prompt template. There's a one-line fix (`DataCollatorForCompletionOnlyLM` or the newer `assistant_only_loss=True`) and I didn't use it. Across 660k examples × 2 epochs, that's a lot of compute thrown away.

**Noisy CoT extraction.** My extractor split responses on "The answer is" or "####" or fell back to splitting on the last sentence. For GSM8K that mostly works. For MathInstruct's 260k examples — which include program-aided traces, GPT-4 chain-of-thought, and a bunch of weird half-formatted outputs — it routinely put the final answer *inside* `<think>` and left the after-`</think>` answer empty or partial. So the model learned the syntax of opening a `<think>` block without learning what's supposed to go in it.

**Random init for new tokens.** `model.resize_token_embeddings(len(tokenizer))` initializes the new rows from a tiny random normal. For 2 new tokens (`<think>`, `</think>`) that's enough to perturb the LM head distribution. Hewitt 2021's mean-init fix would have prevented the ARC-Easy regression. Three lines of code.

**Only 2 epochs.** Recent work (arxiv 2507.08267) finds that small models need 5–10 SFT epochs for the breakthrough to actually land. I did 2 and stopped.

**Stage order is backwards.** DeepSeek-R1 and Tulu-3 both do rich math/CoT first, then broad chat as a polish. I did chat first, math second, which means the math stage is fighting against newly-fresh "be a chatbot" gradients.

**LR is too high.** 2e-5 on a 1B base with essentially zero math prior is enough to memorize SFT formatting noise instead of amplifying any latent math signal. Should have been 5e-6.

---

## 7. Phase 4: GRPO

### Why not DPO

DPO needs preference pairs. To get those, I'd have to sample completions from the SFT model and label them chosen/rejected. With ~1% correctness, almost every pair would be "rejected vs. rejected" — there's no preference signal. So GRPO it was. GRPO just needs a scalar reward and a group of rollouts to compute relative advantage.

### What I built into `train_grpo.py`

A bunch of stuff that the basic TRL examples don't do, because I learned the hard way that the basics aren't enough:

- **AST-based correctness check** using `math_verify` (SymPy under the hood) so `1/2`, `0.5`, and `2/4` all count as equal. With GSM8K's `####` numeric fallback for when SymPy can't parse.
- **Binary format reward.** Strict regex: must contain both `<think>` and `</think>` and a non-empty answer after. Either 1.0 or 0.0. (I also kept a 0.5 partial credit for "valid `<think>` block but messy answer" because pure binary made early training unstable.)
- **3-gram repetition penalty.** When unique 3-gram ratio drops below 20%, scale a penalty up to -1.5. This was the mode-collapse killer.
- **Tokenizer decode monkey-patch.** This one took me hours to figure out. TRL calls `tokenizer.decode(..., skip_special_tokens=True)` internally before passing the text to the reward functions. That strips `<think>`/`</think>`. So my format reward was always seeing 0 because the tokens were gone by the time it got the text. The fix is to monkey-patch `decode` to only strip ChatML control tokens, not all special tokens.
- **Explicit `<|im_end|>` stop.** Without it, the model would just keep going and simulate the user's next turn.
- **Left-padding enforcement** because decoder-only batched generation needs left-padding and TRL doesn't always set it.
- **PyTorch 2.6+ `weights_only` shim** for checkpoint loading.

### Config

> Note: an earlier internal `grpo_report.md` said the GRPO dataset was "MathInstruct + MetaMathQA". That's wrong — I cross-checked the code. `train_grpo.py` loads `openai/gsm8k` train split only. All GRPO numbers below are from single-dataset training on GSM8K's 7,473 problems.

| Param | Value |
|---|---|
| Dataset | GSM8K train, 7,473 problems |
| Epochs | 3 (22,419 steps total) |
| Group size G | 8 |
| KL coefficient β | 0.01 |
| Learning rate | 5e-6, cosine, 5% warmup |
| Max completion length | 512 |
| Temperature | 0.9 |
| Effective batch | 2 × grad accum 4 = 8 |

### The infrastructure mess

GRPO ran across three different GPUs because of quota issues:

1. **AMD MI300X (first run).** Steps 0 → 12,500, about 9 hours. AMD Cloud quota ran out. I sent a SIGINT, let the trainer save a clean checkpoint, then pushed it to HuggingFace Hub.
2. **AMD MI300X (second box).** Steps 12,500 → 19,000. New machine, fresh quota, but vLLM and PyTorch ROCm versions disagreed and I had to sort that out before the resume would work.
3. **GCP 2× NVIDIA L4.** Steps 19,000 → 22,419. Different hardware (CUDA instead of ROCm), required reinstalling drivers and reconfiguring distributed training across 2 cards. Finished here.

The resume-from-checkpoint flow worked perfectly through both hops. That was a small win.

### Training dynamics at the end

```
Total reward         0.5139
  Correctness        0.01389   (~1.4% of rollouts correct)
  Format             0.50      (~50% strict format compliance)
  Repetition penalty 0.0       (never fired across 22k steps)

KL divergence        2.975
Entropy              4.679
Train loss           0.004795
Tokens processed     ~81.4M
Wall time            13h 32m
```

The repetition penalty never firing was actually the thing I was most proud of. Across 22,419 steps the model never collapsed into a loop. The n-gram tripwire was set right.

### GRPO results

| Benchmark | SFT → GRPO | Delta |
|---|---|---|
| GSM8K | 1.00% → 2.20% | **+1.20%** (2.2×) |
| Minerva Math | 0.00% → 2.02% | **+2.02%** (from absolute zero) |
| ARC-Easy | 25.51% → 28.79% | +3.28% (recovered the SFT regression) |
| ARC-Challenge | 24.66% → 22.78% | -1.88% |
| HellaSwag | 26.70% → 26.30% | -0.40% |
| MMLU | 24.60% → 23.62% | -0.98% |

So: math doubled, ARC-Easy came back, the rest drifted slightly down within KL budget. About what you'd expect.

---

## 8. Why the scores are low

This is the part I want to be honest about. The numbers are bad and I want to spell out exactly why instead of hand-waving.

### Root cause 1: the base is way undertrained

50 tokens per parameter. That's the dominant factor. Every other problem in this report is a fraction of this one.

Modern overtraining wisdom for inference-deployable small models is 1,000–10,000 tokens/param (see Beyond Chinchilla-Optimal, SmolLM2 paper). I'm at 50. That means the model has not seen enough math to memorize `7+5=12`, let alone string together a multi-step word problem. No SFT, no RL, no decoding trick fixes this. The downstream stages have a ceiling baked in by this number.

Roughly: this single fact explains about 80% of the GSM8K gap to comparable 1B-class models.

### Root cause 2: the tokenizer is hostile to arithmetic

The audit (§3) showed inconsistent multi-digit chunking. "100" is one token, "2024" is two, "1234567" is three at arbitrary boundaries. A model trying to learn that "382 + 491 = 873" sees "38, 2" + "4, 91" = "8, 73" — and the chunking is different in every example.

Models that do math well either split every digit (Llama-3, DeepSeek-Math) or just use a big enough tokenizer that the merges are stable (Qwen2.5 at 151k vocab). I did neither.

### Root cause 3: the data pipeline cut corners I didn't realize

Recapping from §3: no real dedup, no curriculum, no synthetic data, Proof-Pile-2 missing, no document-boundary attention mask. Each one of those is a known knob that lifts small-model math by single-digit percentage points. Stacked together they're a real loss.

### Root cause 4: SFT mostly learned format

Without completion-only loss masking, half-plus of the gradient went toward fitting the system prompt. Combined with noisy CoT boundaries from the heuristic extractor, the model learned the *shape* of `<think>...</think>` without learning to put correct math inside it.

You can see this in the GRPO logs: format reward saturates near 0.5 (the partial-credit ceiling), correctness reward never moves. That's a direct fingerprint of "model knows the syntax, can't fill the content."

### Root cause 5: GRPO was mathematically doomed by the SFT base

With p ≈ 1.4% empirical correctness and group size G=8:

```
P(group has ≥1 correct rollout) = 1 - (1 - 0.014)^8 ≈ 10.6%
```

So about 89% of training steps had zero variance in the correctness reward across the group — meaning zero advantage signal, meaning the correctness gradient was just noise. The only signal that consistently varied across rollouts was the format reward, so that's what got reinforced.

Even worse, the additive reward shape (`correctness + format + repetition`) let format dominate by roughly 35× in expectation. So the model's rational optimum became: produce a syntactically valid `<think>` block with any content. Which is exactly what the qualitative outputs show — coherent format, garbage reasoning.

This is the DeepSeek-R1-Zero observation in miniature: pure RL only works when the base policy can occasionally succeed by chance. Tiny models on hard math don't qualify without an SFT cold-start of long-CoT distilled data.

### Root cause 6: eval protocol drift

A bunch of the numbers in this report aren't strictly comparable across stages:

- lm-eval-harness's default `gsm8k` task uses raw few-shot text, but my SFT/GRPO models were trained with ChatML. So evaluating them against the default task is OOD prompting. There's a difference of ~1pp between raw-extraction and aligned-template eval.
- Strict-match vs. flexible-extract — SFT used strict, GRPO used flexible. Apples and oranges.
- 8-shot GSM8K eats ~5KB of the 4096 context, which hurts small models more than large ones.
- My custom eval used temperature=0.7 with `do_sample=True` and a single sample. High variance per run.

If I had to redo the eval today I'd use 0-shot CoT with aligned templates, greedy primary, and cons@8 secondary across all three checkpoints.

---

## 9. What actually went right

The benchmarks are bad. The project isn't. Things I'm proud of:

1. **The full stack runs.** Every stage end-to-end, from tokenizer to RL, actually works. A lot of "from scratch" projects ship the pretraining and stop. I went all the way to GRPO and benchmarks.

2. **Cross-platform checkpoint plumbing.** Migrating training state across MaxText/JAX → HF/PyTorch → ROCm → CUDA actually worked. The conversion script is reusable.

3. **Zero mode collapse during GRPO.** 22,419 RL steps and the model never entered a loop. The n-gram tripwire was tuned right.

4. **Stable everywhere.** No gradient explosions in pretraining. No NaN loss in SFT. No catastrophic forgetting in GRPO. Three different hardware platforms.

5. **Documentation that doesn't lie.** Every problem in this report is recorded. Anyone using the repo can see exactly what worked, what didn't, and why.

---

## 10. What I'd do next

I sketched out a 5-tier optimization roadmap. Listed by ROI.

### Decision tree

| Path | Compute | Realistic GSM8K |
|---|---|---|
| A. Eval fixes only | 0 GPU-hours | 3–8% |
| B. Better SFT + RFT + GRPO | ~150 MI300X-h | 8–18% |
| C. Continual pretrain + post-train | 5–15k GPU-h | 35–60% |
| D. Teacher distillation | ~300 MI300X-h | 30–55% |
| E. Full pretrain redo | TPU v4-32, 6 weeks | 55–75% |

### Tier 1: free fixes

- Re-eval with aligned ChatML templates, 0-shot CoT, both strict and flex metrics, greedy primary
- Self-consistency cons@32 (sample 32, majority vote)
- Tokenizer audit (already done; result above)

Expected: +2–5pp on GSM8K from protocol alone.

### Tier 2: redo SFT properly

- Single-stage mix: 70% NuminaMath-CoT, 15% OpenMathInstruct-2 (verified-correct subset), 10% GSM8K with distilled long CoT, 5% Tulu-3 for chat preservation
- `assistant_only_loss=True` (or DataCollatorForCompletionOnlyLM)
- Deterministic CoT formatting per dataset (no heuristic splitting)
- Mean-init for new token embeddings
- lr=5e-6, cosine, 5% warmup, 5–8 epochs
- Validate epoch-by-epoch on a 500-problem GSM8K dev split

Expected: GSM8K → 8–15%, MATH → 3–8%.

### Tier 3: redesigned GRPO (only after Tier 2)

- Gated reward: `format_pass * (1.0 if correct else -0.1)`. No format = 0. Format alone doesn't pay.
- `max_completion_length = 2048` (up from 512)
- Multi-dataset prompt mix (GSM8K + MATH-Algebra + ASDiv + SVAMP)
- G=16, temperature=0.7
- If correctness is still <5% after Tier 2, skip GRPO and do RFT (rejection-sampling fine-tuning) first

Expected: +3–8pp over Tier 2.

### Tier 4: continual pretrain (this is where the real gains live)

30B tokens of FineMath-4+ + InfiWebMath + Proof-Pile-2 + MathPile + 10% FineWeb-Edu retention. Apply Rho-1 Selective Language Modeling (only backprop on the highest-utility 60–70% of tokens). Add document-boundary attention masks. Add z-loss and QK-norm. Anneal math-heavy in the last 10% of tokens.

Cost: ~70 MI300X-hours for 30B, ~350 for 150B.

Anchor: Rho-1B did +30pp absolute on math with 15B tokens of selective continual pretraining. InfiR-1B went from GSM8K 8% to 63% with 940B tokens.

Realistic landing for TinyMathReason after 30B continual: GSM8K ~18%, MATH ~10%, central case.

### Tier 5: test-time scaling

cons@N, best-of-N with a tiny PRM, step-level beam search. All free at training time. Easy +5–15pp.

### The 5-day plan I'd actually run

If you handed me one MI300X for 5 days right now:

1. Day 1: tokenizer audit (done), eval rebaseline, distill 30k GSM8K solutions with Qwen2.5-Math-7B as teacher, filter by ground truth.
2. Days 2–3: Tier-2 SFT redo on the distilled + curated mix.
3. Day 4: RFT — sample 16 CoTs per GSM8K problem with the new SFT, keep correct, SFT 1 more epoch.
4. Day 5: Tier-3 GRPO with gated reward, 2048 max length, multi-dataset prompts.

Target: GSM8K 12–20%, MATH 4–8%. Comparable to TinyLlama-1.1B-Math at the same weight class.

---

## 11. Hardware and cost

| Phase | Hardware | Time | Approx cost |
|---|---|---|---|
| Data processing | 2× Vultr c2-standard-30 | ~2 days | ~$15 |
| Pretraining | TPU v4-64 (spot) | ~7 days | ~$2,000–4,000 |
| SFT | AMD MI300X (192GB) | ~15h | ~$30 |
| GRPO run 1 | AMD MI300X | ~9h | in quota |
| GRPO run 2 (resume) | AMD MI300X | ~4h | in quota |
| GRPO finish | 2× NVIDIA L4 (GCP) | ~5h | ~$10 |
| Evaluation | GCP VMs, Modal | ~2h | ~$5 |

So the pretraining was ~99% of total spend. Everything else is rounding error. If you're starting a project like this, this is the breakdown to budget around.

---

## 12. Lessons that I'd tattoo on the back of my hand

1. **Token budget beats every clever trick.** At 50 tok/param, no SFT recipe can save you. Get to at least 1,000 tok/param before optimizing anything else.
2. **Audit your tokenizer before training.** If "100" is one token and "1234567" is three asymmetric chunks, you've capped your math performance before step one.
3. **Loss masking isn't a tuning detail.** It's a 30–50% gradient efficiency lever. One flag.
4. **RL needs a non-zero base.** If your model can't occasionally get a problem right, RL just amplifies whatever else varies — usually format. Cold-start with distilled long CoT first.
5. **Gate rewards, don't sum them.** Additive rewards always get exploited by whichever component is easier to satisfy. Make format a precondition, not a competing objective.
6. **Checkpoint conversion is a real project.** Budget days, not hours. Write the inspector before the converter.
7. **Document failure as carefully as success.** I trust this report more than I'd trust a clean "here's the recipe" writeup, and so should you.

---

## 13. Repo layout

```
TinyMathReason-1B/
├── src/
│   ├── data/             tokenizer + data pipeline (a-f)
│   ├── model/            PyTorch model definition
│   ├── train/            MaxText configs, conversion, monitoring
│   ├── sft/              SFT data prep + training
│   ├── dpo/              DPO + GRPO training
│   └── eval/             benchmarks + custom eval
├── docs/                 architecture, setup guides
├── scripts/              TPU/GPU provisioning
├── hf_1b_model/          converted HF checkpoint
├── STATUS.md             phase tracker
├── CRITIQUE.md           root-cause analysis
└── opt_olan.md           optimization roadmap with projections
```

---

## 14. Artifacts

| What | Where |
|---|---|
| Base model (HF) | `./hf_1b_model/` |
| SFT model | https://huggingface.co/himanshunakrani9/TinyMathReason-1B-sft |
| GRPO model | https://huggingface.co/himanshunakrani9/TinyMathReason-1B-grpo |
| Pretraining data | `gs://tinymath-reason-data-himanshu/pretraining-data/` |
| Final pretraining checkpoint | `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run11/checkpoints/54362/` |
| Repo | https://github.com/himanshu-nakrani/TinyMathReason-1B |

---

If the takeaway from this report had to fit in one sentence: a 1B model trained on 57B tokens with a multi-digit-merging tokenizer is going to get ~2% on GSM8K no matter what SFT or RL recipe you bolt on top. The interesting question is what the next 5 days of compute should look like, and that's all in §10.
