# I built a 1B math model from scratch. It scored 2.2%. Here's what I learned.

I spent about 25 days building TinyMathReason-1B. 1.12 billion parameters, Llama-style, trained from scratch on Google Cloud TPU v4-64, fine-tuned on an AMD MI300X, and pushed through GRPO reinforcement learning across two more GPU environments.

Final GSM8K score: **2.2%**. GPT-4 gets around 95%. So that's where we are.

I'm not writing this to brag. I'm writing it because the project taught me more about how LLMs actually work than two years of reading papers, and most of what I learned came from things going wrong.

## The architecture

TinyMathReason-1B is a Llama-2 lookalike: 22 layers, hidden dim 2048, GQA at a 4:1 ratio (16 query heads, 4 KV heads), SwiGLU MLP, RoPE, 4096-token context. I gave it 128-dimensional attention heads instead of the more common 64, following Llama-3's choice that larger heads capture more complex stuff. Custom 32k BPE tokenizer with `<think>` and `</think>` reserved for the SFT stage later.

Total: 1,123,117,056 parameters. Almost the same shape as TinyLlama-1.1B, which was on purpose. I wanted my architecture decisions to not be the experimental variable.

## The full pipeline

It's a five-phase thing. I'll spare you the deep dive, but here's the shape:

1. **Data.** Two Vultr CPU boxes ran the pipeline in parallel: FineWeb-Edu, OpenWebMath, MathPile, Stack-Edu. Six stages: download, clean, mix, tokenize, pack into 4096-token sequences, shard, upload to GCS. About 57 billion tokens out the other end.
2. **Pretraining.** TPU v4-64, Google's MaxText framework (JAX/Flax), 54,362 steps. AdamW with cosine decay. bfloat16. Took roughly a week.
3. **Checkpoint conversion.** MaxText/JAX Orbax → HuggingFace safetensors. This is the part nobody talks about, and it took me five days.
4. **SFT.** Two stages on the MI300X: instruction-following on Alpaca first, then math reasoning with `<think>` tags on GSM8K + MathInstruct + MetaMathQA (~660k examples total).
5. **GRPO.** Group Relative Policy Optimization on GSM8K's 7,473 training problems, 3 epochs, 22,419 steps. Migrated across three GPUs because of quota issues.

## The hardest part wasn't training

It was the checkpoint conversion. I had to rewrite the conversion script and fix five separate bugs, none of which were obvious:

1. MaxText bakes the `1/√d` attention scaling *into* the query weights. HuggingFace applies it at runtime. So you have to un-scale during conversion.
2. GQA tensors are stored in different shapes between MaxText and HF, you need to slice along the right axis.
3. RoPE uses different interleaving orders in the two frameworks. Get it wrong and your model generates fluent gibberish.
4. MaxText padded the vocab to 32,768 for TPU alignment, HF needs that mapping spelled out.
5. The tokenizer config file had an invalid class name, so AutoTokenizer was failing silently.

I ended up writing a separate inspection tool that just dumps the full Orbax checkpoint tree with tensor shapes. Without it I'd still be debugging.

## The results, honestly

| Benchmark | Base | After SFT | After GRPO |
|---|:---:|:---:|:---:|
| GSM8K | 1.0% | 1.0% | **2.2%** |
| ARC-Challenge | 21.7% | 24.7% | 22.8% |
| HellaSwag | 25.8% | 26.7% | 26.3% |
| MMLU | 23.5% | 24.6% | 23.6% |

GRPO doubled the math score. Doubling 1% to get 2% is still 2%, but the curve was real and not just noise.

## Why is it 2% and not 50%

This is the part I want to be honest about. There's one number that explains almost everything:

**My model saw 50 tokens per parameter. TinyLlama saw 2,700. Llama-3.2-1B saw 7,500. Qwen2.5-1.5B saw 12,000.**

That's 54 times fewer tokens than TinyLlama, 150 times fewer than Llama-3.2-1B. At that budget the model literally hasn't seen enough math to memorize arithmetic facts, let alone compose them into multi-step reasoning. No amount of SFT or RL on top can extract knowledge that isn't latent in the base.

Then there's the tokenizer. I audited it after the fact:

```
"100"     → ["100"]
"50"      → ["50"]
"2024"    → ["20", "24"]
"1234567" → ["12", "345", "67"]
```

Frequent round numbers are one token. Less common numbers shatter into 2- or 3-digit chunks at unpredictable boundaries. Llama-3 and DeepSeek-Math both split every digit into its own token, so addition and subtraction are linearly composable. Mine isn't. So even if the model had seen enough math, the tokenizer would have capped how well it could learn arithmetic.

The SFT stage learned the *format* of reasoning without learning the content. Heuristic CoT extraction put the answers inside `<think>` blocks half the time, so the model learned to open the tag without learning what goes inside.

And GRPO was mathematically doomed. With only 1.4% base correctness and group size 8, about 89% of training steps had zero variance in the correctness reward, which means zero gradient signal. The only signal that consistently varied was the format reward — which dominated the additive reward shape by 35x. So the model optimized for "produce a valid `<think>` shell with any content," exactly the reward-hacking behavior I later saw in the outputs.

## What I'd actually change

1. **More pretraining tokens.** This is the single biggest lever. 30B additional tokens of FineMath + Proof-Pile-2 with Selective Language Modeling could plausibly take GSM8K from 2% to 18%.
2. **Replace the tokenizer.** Either rebuild with digit-splitting, or just adopt Llama-3's tokenizer wholesale.
3. **Use completion-only loss masking.** TRL has a flag for this (`assistant_only_loss=True`). One line. 30-50% better gradient efficiency.
4. **Gate the GRPO reward, don't sum it.** Use `format_pass × (correct ? 1 : -0.1)`. Format becomes a precondition, not a competing objective.
5. **More SFT epochs.** Recent work says small models need 5-10, not 2.

## The infrastructure reality

This project touched five different compute environments. Local Mac for development. Two Vultr CPU boxes for data. TPU v4-64 for pretraining. AMD MI300X for SFT and the first GRPO runs. And finally two NVIDIA L4s on GCP when AMD quota ran out.

Migrating GRPO across three GPUs was a small adventure. I had to send a clean SIGINT during training to get a final checkpoint, push that to HuggingFace Hub, then resume on a different cloud. The resume worked perfectly, which I think is genuinely cool — 12,500 steps of optimizer state survived a transit that included a ROCm-to-CUDA hop.

## What this is actually for

Most open-source LLM projects ship the weights and stop. This repo ships the tokenizer training script, the data pipeline (including the bugs I found in my own code six weeks later), the TPU SSH debugging, the conversion script with all five bug fixes, and the honest analysis of why the benchmarks are bad.

If you're trying to actually understand how LLMs get built — not just use them — this is a reference implementation of the entire stack, warts and all. The optimization roadmap projects that with ~$560 of MI300X time I could get to ~50% GSM8K just by fixing the things I now know are wrong. I might run that next.

For now, the model, code, and every training log are open: [github.com/himanshu-nakrani/TinyMathReason-1B](https://github.com/himanshu-nakrani/TinyMathReason-1B). The SFT and GRPO weights are at [huggingface.co/himanshunakrani9](https://huggingface.co/himanshunakrani9).

Sometimes the best a project can do is show you exactly how and why it falls short. That's what this one delivered.
