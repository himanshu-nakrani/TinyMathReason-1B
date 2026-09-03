# Building TinyMathReason-1B: 25 days, 5 machines, 2.2% on GSM8K

A long, honest writeup of what it actually takes to train a 1B math model from scratch.

Most open-source LLM projects hand you a set of weights and a model card. This article is the opposite. I'm going to walk through the whole journey of building TinyMathReason-1B — every architecture decision, every infrastructure migration, every bug I shipped, and every lesson I'm still mad about.

The project took 25 days of solo work across 5 different compute environments. The model is a 1.12-billion parameter Llama-style transformer trained from scratch. Its final GSM8K score is 2.2%. The number is small. The learning is enormous, and I think the honest version of the story is more useful than another polished "here's how I did it" post.

---

## Part 1: The architecture

The first decision I had to make was the architecture. I went with a Llama-style decoder-only transformer because that's what everyone has settled on for autoregressive language models, and I didn't want my architecture to be the experimental variable. The interesting variable here is the training pipeline, not the model shape.

The exact config is 1,123,117,056 parameters distributed as:

- 22 transformer layers (TinyLlama's proven depth)
- 2048 hidden dimension with SwiGLU MLP, intermediate 5632 (≈ 8/3 × 2048, rounded up to a 128-multiple so TPU loads stay aligned)
- Grouped Query Attention with 16 query heads and 4 KV heads — a 4:1 ratio
- 128-dimensional attention heads, following Llama-3's choice
- RoPE positional encoding, θ=10000
- 32k BPE tokenizer (padded to 32,768 to keep FSDP happy)

The GQA ratio was a deliberate pick. With 4 KV heads instead of 16, the KV cache at inference time is 4× smaller. At inference for a 1B model on commodity hardware, that's the difference between fitting a 4096-token context and running out of memory.

The 128-dim heads versus the more common 64 was the other deliberate call. Llama-3 went bigger on head dim and got better reasoning out of it. Each head can model more complex dependencies, which matters when you're trying to track multiple variables across a multi-step derivation. The trade-off is fewer heads, but for math reasoning that felt right.

---

## Part 2: The data pipeline

### The tokenizer (where I made my first big mistake)

I trained a 32,000-token BPE tokenizer on a math-heavy sample. Reserved `<think>` and `</think>` as special tokens for the SFT stage later. Took maybe a day.

In hindsight this was the single most consequential decision of the project, and I made it badly. I trained the tokenizer on "sample data" instead of a massive, representative math corpus. After everything else was done I ran a direct audit on the trained tokenizer and the results were the worst case:

```
"100"     → ["100"]
"50"      → ["50"]
"2024"    → ["20", "24"]
"1234567" → ["12", "345", "67"]
"382391"  → ["38", "23", "91"]
"382 + 491 = 873" → ["38", "2", "Ġ+", "Ġ4", "91", "Ġ=", "Ġ8", "73"]
```

So common round numbers like "100" and "50" became single opaque tokens. Less common numbers got shattered into 2- and 3-digit chunks at arbitrary boundaries. The boundary of "382" inside "382 + 491" was *different* from "382" inside "382391". The model would have to learn arithmetic across a representation that changes underneath it.

The models that actually do math well take one of two paths. Llama-3 and DeepSeek-Math split every digit into its own token, so addition and subtraction are linearly composable. Qwen2.5 just uses a 151k-vocab tokenizer where the merges are stable enough that arithmetic patterns can lock in. I did neither, and I was paying for it from step one of pretraining.

### Building the corpus

I spun up two Vultr c2-standard-30 CPU boxes to process the data pipeline in parallel. Four datasets:

- **FineWeb-Edu** for general educational web text, about 50% of the mix
- **OpenWebMath** for math-focused web content, about 15%
- **MathPile** for curated math textbooks and papers, about 15%
- **Stack-Edu** for educational code, about 10%

Pipeline ran in six stages: download → clean and filter → mix → tokenize and pack → shard → upload to GCS.

Six weeks later, going through the code as part of writing the postmortem, I found a bunch of stuff that the documentation claimed but the code didn't actually do:

**Proof-Pile-2 was in the mix config at 15%, but I never added it to the download script.** So the model never saw a single token of formal mathematical proofs during pretraining. I had it listed in `c_mix_datasets.py` but `a_download_datasets.py` only fetched four datasets. This kind of bug is exactly the reason I'm writing this article — it's invisible from the outside.

**The "MinHash deduplication" I claimed in the README was just `hash(text)` exact-string dedup.** No near-duplicate removal, no shingling, none of the actual MinHash machinery. So all the slightly-modified versions of the same Wikipedia paragraph passed through as if they were unique training signal.

**Quality filtering was minimal.** Just `len(words) > 20` and `alpha_ratio > 0.3`. No perplexity filter, no learned classifier, no PII scrubbing, no language ID.

**Sequence packing used straight concatenation with an EOS token between documents.** No document-boundary attention mask, which means the model could attend across documents inside a packed sequence. This is particularly bad for math where each problem needs to be self-contained.

**No curriculum, no annealing.** SmolLM2, MiniCPM, and Qwen2.5 all use staged pretraining where they ramp up the math fraction in the final 10-20% of training. I used a uniform mix from start to finish.

**No synthetic data.** Cosmopedia — the 28B-token synthetic textbook corpus — is the single thing that gave SmolLM2 its edge over TinyLlama. I had none.

Total corpus shipped: roughly 57 billion tokens.

---

## Part 3: TPU pretraining

### The setup

Pretraining ran on Google Cloud TPU v4-64 (64 chips) using Google's MaxText framework (JAX/Flax). Data streamed from GCS during training.

The basic shape: 54,362 steps × 64 sequences × 4096 tokens per sequence ≈ 57B tokens. AdamW with lr=3e-4 cosine-decayed to 10% of peak, β₁=0.9, β₂=0.95, bfloat16. Roughly 8,900 tokens/sec/chip, about 66 TFLOP/s/device.

### The bugs I'd rather forget

**The zero-layers bug.** I lost a week to this. Early runs were producing checkpoints with 0.134B parameters. Just the embedding matrix and the LM head. All 22 transformer layers were silently missing from the saved checkpoint. The cause turned out to be a compatibility issue between MaxText's `scan_layers: True` path (the performance-optimized one) and the NNX-style transformer blocks I was using. The scan op was failing to register the layers in the PyTree without raising any error. The fix was `scan_layers: False`, which works but runs slower. That cost is part of why my final run came in at 57B tokens instead of pushing further.

**XLA host RAM OOM.** Once the layers were actually being saved, the TPU VM started getting OOM-killed during XLA compilation. Not HBM, host RAM. Compiling a 1.1B-parameter model with per-device-batch-size 8 wanted more than 300GB of host memory to build the HLO graph. Dropping to per-device-batch-size 2 fixed it. So my effective batch became 64 sequences (32 chips × 2) instead of the original plan.

**JAX 0.6.2 breaking changes.** JAX removed `jax.Ref`. MaxText was still using it. I added a `setup_tpu.sh` patch that aliases `jax.Ref` to `typing.Any` and `sed`-patches the ragged attention kernels directly in MaxText's source.

**Pallas mock injector.** MaxText imports `jax.experimental.pallas.ops.tpu.splash_attention`, which doesn't exist in the public JAX release. Python 3.12's strict import system wouldn't let me stub it out trivially. I ended up writing a proper `MetaPathFinder` subclass that spoofs `__spec__` and `__path__` so the imports succeed but never get called. This kind of thing is what eats whole afternoons in TPU land.

**Multi-worker SSH choreography.** A v4-64 is 8 host VMs each driving 8 chips, and they all need to be in sync for the JAX distributed mesh to come up. If one host reboots while the others don't, you get "SSH 255" or "Connection refused" and the mesh collapses. The fix is a rolling reboot across all 8 hosts followed by a 5-minute sync wait. I have a Makefile target for it now. Spot TPUs make this a regular occurrence.

### The number that matters

After all that, the final run completed cleanly. Loss converged. But here's the harsh part:

| Model | Params | Pretrain tokens | Tokens / param |
|---|---|---|---|
| **TinyMathReason** | **1.12B** | **57B** | **~50** |
| TinyLlama-1.1B | 1.1B | 3T | ~2,700 |
| Llama-3.2-1B | 1.2B | ~9T | ~7,500 |
| Qwen2.5-1.5B | 1.5B | ~18T | ~12,000 |

50 tokens per parameter. The Chinchilla-optimal number for compute-optimal pretraining is around 20 tok/param, but for inference-deployable small models the modern wisdom is 1,000–10,000 tok/param (see Beyond Chinchilla-Optimal, SmolLM2 paper). At 50, the model doesn't have enough exposure to math to memorize basic identities like `7+5=12`, let alone compose multi-step reasoning. This one fact, by my back-of-envelope estimate, explains around 80% of all downstream performance gaps.

---

## Part 4: The checkpoint conversion saga

Converting from MaxText/JAX (Orbax format) to HuggingFace (PyTorch safetensors) was the single most technically demanding task in the project. I budgeted a day. It took five.

I had to completely rewrite `convert_checkpoint.py`. Five separate bugs, none of which were obvious from outside:

**1. Query scaling.** MaxText bakes the `1/√d_head` attention scaling *inside* the query projection weights at save time. HuggingFace applies it at runtime. So when you load MaxText weights into an HF model, the scaling gets applied twice. You have to multiply the query weights by `√d_head` during conversion to undo the bake-in.

**2. GQA tensor shapes.** MaxText stores Q, K, V as stacked tensors. For my 16-Q-head, 4-KV-head GQA, the shapes differ from HF's expectation. The slicing logic to split them correctly is non-obvious and I got it wrong the first three times.

**3. RoPE interleaving.** MaxText and HuggingFace use different interleaving orders for rotary position embeddings. Get this wrong and your model generates fluent-looking text that means nothing. The fix is a permutation on the head-dim axis. I only caught this when the converted model started producing word salad on the verification pass.

**4. Vocab size padding.** MaxText padded my vocab to 32,768 for TPU alignment. The HF config needed that mapping spelled out explicitly. Off-by-one errors here produce garbage logits.

**5. Tokenizer config.** The `tokenizer_config.json` file had `"tokenizer_class": "TokenizersBackend"` which isn't a real class. AutoTokenizer was failing silently. Switched it to `PreTrainedTokenizerFast`.

The thing I'd recommend to anyone doing this conversion: write the inspector first. I built `inspect_checkpoint.py` after the third failed conversion attempt. It just dumps the full Orbax PyTree with tensor shapes. After that, conversion went from blind to mechanical.

For verification, I ran a forward pass on my MacBook in bfloat16. Memory was tight, so I had to reorganize the script to save the safetensors first and only run verification afterward.

---

## Part 5: SFT, where format got learned and reasoning didn't

SFT ran on an AMD MI300X (192GB VRAM, plenty of room) using HuggingFace Transformers and TRL's SFTTrainer.

### Two stages, in retrospect the wrong order

**Stage 1: conversational prior.** Trained on `tatsu-lab/alpaca` (~52k examples) for 2 epochs at lr=2e-5. Goal was to give the base model a sense of "respond when asked." No CoT, no special tokens.

**Stage 2: reasoning traces.** Added `<think>` and `</think>` to the tokenizer, resized embeddings, and trained on GSM8K + MathInstruct + MetaMathQA (~662k examples) for 2 epochs at lr=2e-5 with a ChatML template. I wrote a heuristic CoT extractor that wrapped the reasoning step in `<think>` tags.

### What went wrong

**No completion-only loss masking.** SFTTrainer with `dataset_text_field="text"` computes loss over the *entire* sequence. System prompt, user question, assistant answer — all of it. For my templates, that means something like 50–70% of every gradient was the model learning to reproduce the system prompt boilerplate. Across 660k examples × 2 epochs, that's a lot of wasted compute. There's a one-line fix (`DataCollatorForCompletionOnlyLM` in older TRL, or `assistant_only_loss=True` in newer versions) and I didn't use it.

**Noisy CoT extraction.** My extractor split responses on "The answer is" or "####" or fell back to splitting on the last sentence. For GSM8K (which uses `####` reliably) it works. For MathInstruct's 260k examples, which contain GPT-4 chains, program-aided traces, and multi-format outputs, the heuristic routinely put the answer *inside* the `<think>` block and left the after-`</think>` part empty or partial. So the model learned the syntax of opening a `<think>` block without learning what was supposed to go in it. That's the fingerprint that showed up later in GRPO too.

**Random initialization for new tokens.** `model.resize_token_embeddings(len(tokenizer))` initializes new rows from a tiny random normal. For just 2 new tokens (`<think>`, `</think>`) that's enough to perturb the LM head distribution. The classic fix (Hewitt 2021) is to initialize new rows as the mean of existing embeddings. Without it, ARC-Easy regressed from 29.9% to 25.5% during SFT — a 4.4pp drop that took weeks of post-training to half-recover.

**Only 2 epochs.** Recent work (arxiv 2507.08267) finds that small models need 5-10 SFT epochs for capability breakthroughs to actually land. I trained for 2.

**Stage order is backwards for weak bases.** DeepSeek-R1 and Tulu-3 both do math/CoT first, broad chat second. The chat stage acts as a polish on top of strong reasoning. I did chat first, which means the math stage was fighting against fresh "be a chatbot" gradients.

**Learning rate too high.** 2e-5 on a 1B base with essentially zero math prior is enough to memorize SFT formatting noise instead of amplifying any latent math signal. Should have been 5e-6.

### The results

| Benchmark | Base → SFT | Delta |
|---|---|---|
| GSM8K (8-shot) | 1.00% → 1.00% | 0 (eval mismatch; 1% with ChatML template) |
| MATH (4-shot) | 0.00% → 0.00% | 0 |
| ARC-Easy | 29.90% → 25.51% | −4.4% (the embedding regression) |
| ARC-Challenge | 21.70% → 24.66% | +3.0% |
| HellaSwag | 25.80% → 26.70% | +0.9% |
| MMLU | 23.50% → 24.60% | +1.1% |

So general-knowledge benchmarks moved a little, math didn't move at all, and ARC-Easy regressed. About what you'd expect from a curriculum that's mostly teaching format rather than capability.

---

## Part 6: GRPO

### Why not DPO

DPO needs preference pairs — chosen vs. rejected completions. To get those I'd have to sample from the SFT model and label each completion. With ~1% correctness, almost every sample is "rejected", so there's no preference signal. GRPO sidesteps this: it only needs a scalar reward function and a group of G rollouts per prompt. You compute relative advantage across the group and reinforce the better-than-average ones.

### What I had to build into `train_grpo.py`

The basic TRL example would not have worked here. Things I added:

**AST-based correctness verification** using `math_verify` (SymPy under the hood) so `1/2`, `0.5`, and `2/4` all count as equal. With a GSM8K numeric fallback for when SymPy can't parse.

**Binary format reward.** Strict regex: must contain both `<think>` and `</think>` and a non-empty answer afterwards. 1.0 or 0.0. (I kept a 0.5 partial credit for "valid `<think>` block but malformed answer" because pure binary made early training unstable.)

**3-gram repetition penalty.** When the unique-3-gram ratio drops below 20%, scale a penalty up to -1.5. This was the mode-collapse killer.

**Tokenizer decode monkey-patch.** This one took me hours to figure out. TRL calls `tokenizer.decode(..., skip_special_tokens=True)` internally before passing text to the reward functions. That strips `<think>`/`</think>`. So my format reward was always seeing 0 because the tokens were already gone. The fix is to monkey-patch `decode` to only strip ChatML control tokens while preserving the reasoning tags.

**Explicit `<|im_end|>` stop token injection.** Without it, the model would just keep going and simulate the user's next turn.

**Left-padding enforcement** for decoder-only batched generation, because TRL doesn't always set it correctly.

**PyTorch 2.6+ `weights_only` shim** for checkpoint loading.

### The infrastructure odyssey

GRPO training migrated across **three GPU environments** because of quota issues:

1. **AMD MI300X #1.** Steps 0 → 12,500 over ~9 hours. AMD Cloud quota ran out. I sent a clean SIGINT, the trainer saved a final checkpoint, and I pushed it to HuggingFace Hub.
2. **AMD MI300X #2.** Steps 12,500 → 19,000. New machine, fresh quota, but the vLLM and PyTorch ROCm versions disagreed. Had to sort that out before the resume would work. Different ROCm version meant different optimizer state serialization too, which made the resume non-trivial.
3. **GCP 2× NVIDIA L4.** Steps 19,000 → 22,419 (completion). Migrated to CUDA, reinstalled NVIDIA drivers, reconfigured distributed training across two cards. Total wall time for the GRPO run came in at 13h 32m, finishing on the L4s.

The resume-from-checkpoint flow worked perfectly through both hops. Optimizer state, learning rate schedule, RNG state — all preserved. That was a small but real engineering win.

### The math of why the GRPO reward signal failed

Final GRPO metrics tell the story:

```
Correctness reward    0.014    (~1.4% of rollouts correct)
Format reward         0.500    (~50% strict format compliance)
Repetition penalty    0.000    (never fired in 22k steps)
KL divergence         2.975    (stayed anchored to SFT)
Entropy               4.679    (high, no collapse)
```

With p = 1.4% empirical correctness and G = 8:

```
P(group has ≥1 correct rollout) = 1 − (1 − 0.014)^8 ≈ 10.6%
```

So roughly 89% of training steps had zero variance in the correctness reward across the group, which means zero advantage signal, which means the correctness gradient was just noise. The only signal that consistently varied was format. Format reward dominated the additive total reward by roughly 35× in expectation.

The model's rational optimum became: produce a syntactically valid `<think>` shell with any content inside. Which is exactly the reward-hacking behavior I later confirmed by reading the actual completions. The traces are formally correct (open tag, content, close tag, answer) and semantically empty.

This is the DeepSeek-R1-Zero observation in miniature. Pure RL only works when the base policy can occasionally succeed by chance. Tiny models on hard math don't qualify without an SFT cold-start of strong long-CoT distilled data. I skipped that step.

### The actual GRPO results

| Benchmark | SFT → GRPO | Delta |
|---|---|---|
| GSM8K | 1.00% → **2.20%** | +1.20% (2.2× lift, but from a tiny base) |
| Minerva Math | 0.00% → **2.02%** | +2.02% (real signal from absolute zero) |
| ARC-Easy | 25.51% → 28.79% | +3.28% (recovered the SFT regression) |
| ARC-Challenge | 24.66% → 22.78% | −1.88% (within KL budget) |
| HellaSwag | 26.70% → 26.30% | −0.40% |
| MMLU | 24.60% → 23.62% | −0.98% |

GRPO doubled the math score and recovered the ARC-Easy regression. The other benchmarks drifted slightly within the KL constraint, as expected.

---

## Part 7: The honest postmortem

### What worked

**The full stack actually runs.** Every stage end-to-end — tokenizer training, data pipeline, TPU pretraining, checkpoint conversion, SFT, GRPO, evaluation — completed successfully. A lot of "from scratch" projects ship the pretraining and stop. I made it through reinforcement learning to benchmark-able outputs.

**Zero mode collapse during 22,419 GRPO steps.** The repetition penalty never fired. The model never entered an infinite loop. The n-gram uniqueness threshold was tuned correctly the first time. This is one of those wins that doesn't make for a great chart but mattered a lot.

**Cross-platform checkpoint migration works.** Moving training state from MaxText/JAX to HF/PyTorch to ROCm to CUDA actually preserved 12,500 steps perfectly. The conversion infrastructure is reusable.

**Stable training everywhere.** No gradient explosions, no NaN loss, no catastrophic forgetting visible across stages. Three hardware platforms. Two cloud providers.

### What broke (and stayed broken)

**Token budget.** 50 tok/param is roughly 1-2 orders of magnitude below where 1B-class math models start to work. Everything downstream inherits this.

**Tokenizer.** Trained on "sample data". Multi-digit numbers get inconsistently chunked. Hard ceiling on arithmetic.

**SFT gradient efficiency.** Without completion-only loss masking, half-plus of every gradient was wasted on prompt boilerplate.

**RL reward shape.** Additive `correctness + format + repetition` lets format dominate by 35× when correctness is rare. Format-hacking was the model's rational optimum.

---

## Part 8: What I'd actually do next

I sketched out a five-tier optimization roadmap, ordered by ROI per dollar.

**Tier 1 — Free fixes (hours).** Re-evaluate everything with aligned ChatML templates and 0-shot CoT instead of 8-shot raw text. Add self-consistency cons@32. Audit the tokenizer (already done; results above). Expected: +2-5pp on GSM8K from protocol alone.

**Tier 2 — Redo SFT properly (1-3 days).** Single-stage mix: 70% NuminaMath-CoT, 15% OpenMathInstruct-2 (verified-correct subset only), 10% GSM8K with distilled long CoT from a 7B teacher, 5% Tulu-3 for chat preservation. Use `assistant_only_loss=True`. Deterministic CoT formatting per dataset. Mean-init for new token embeddings. lr=5e-6, 5-10 epochs, validate epoch-by-epoch on a held-out dev split. Expected: GSM8K → 8-15%.

**Tier 3 — Redesigned GRPO (2-4 days, only after Tier 2).** Replace the additive reward with a gated one: `format_pass * (1.0 if correct else -0.1)`. Format becomes a precondition, not a competing objective. Bump `max_completion_length` from 512 to 2048. Multi-dataset prompt mix (GSM8K + MATH-Algebra + ASDiv + SVAMP) to prevent prompt-distribution memorization. G=16, temperature 0.7. If correctness is still below 5% after Tier 2, skip GRPO and do rejection-sampling fine-tuning first.

**Tier 4 — Continual pretraining (1-4 weeks, this is where the real gains live).** 30B tokens of FineMath-4+ + InfiWebMath + Proof-Pile-2 + MathPile, with Rho-1 Selective Language Modeling (only backprop on the top 60-70% highest-utility tokens). Add document-boundary attention masks. Add z-loss and QK-norm. Anneal math-heavy in the final 10%. Anchor: Rho-1B did +30pp on MATH with 15B selective continual tokens. InfiR-1B went from GSM8K 8% to 63% with 940B continual tokens on top of Llama-3.2-1B.

**Tier 5 — Test-time scaling (orthogonal).** cons@N, best-of-N with a tiny PRM, step-level beam search. Free at training time, easily +5-15pp.

### The five-day plan

If you handed me one MI300X for five days:

- Day 1: tokenizer audit (done), eval rebaseline, distill 30k GSM8K solutions with Qwen2.5-Math-7B-Instruct as teacher, filter by ground-truth correctness.
- Days 2-3: Tier-2 SFT redo on the distilled + curated mix.
- Day 4: rejection-sampling fine-tuning — sample 16 CoTs per GSM8K problem with the new SFT, keep correct, SFT one more epoch.
- Day 5: Tier-3 GRPO with gated reward, 2048 max length, multi-dataset prompts.

Target: GSM8K 12-20%, MATH 4-8%. Genuinely competitive with TinyLlama-1.1B-Math at the same weight class. That would be the right place to stop the post-training-only phase of the project.

---

## The takeaway

TinyMathReason-1B won't win any benchmarks today. But I think it's one of the most complete, honest, end-to-end LLM training artifacts in the open-source ecosystem.

Every failure is documented. Every workaround is explained. Every design decision has a rationale and, more importantly, a retrospective from after I knew how it played out.

If you're trying to learn how LLMs are actually built — not just how to call an API — this project shows you the full picture. The architecture trade-offs. The infrastructure chaos. The reward functions that get gamed by the very model you're trying to train. The humbling reality that a 1B model trained on 57B tokens just doesn't know enough math to reason about it, no matter what recipe you bolt on top.

The model and all training artifacts are open source: **[github.com/himanshu-nakrani/TinyMathReason-1B](https://github.com/himanshu-nakrani/TinyMathReason-1B)**. SFT and GRPO weights are on Hugging Face under **[himanshunakrani9](https://huggingface.co/himanshunakrani9)**.

Sometimes the most valuable thing a project can teach you is exactly where and why it falls short. That's the lesson I'm taking from this one.
