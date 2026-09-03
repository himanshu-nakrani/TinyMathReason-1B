# How I post-trained a 1B model with SFT + GRPO for $0 (Part 2 of 2)

*Supervised fine-tuning, GRPO reinforcement learning across three different GPUs, and the full $0-cash credit-stacking story.*

---

In [Part 1](#) I walked through pretraining: tokenizer, data, TPU run, and the painful Orbax → HuggingFace conversion. At the end of that I had a 1.12B base model that could predict tokens but didn't know how to be helpful, didn't know to put answers after `<think>` tags, and would happily fall into infinite LaTeX loops on hard prompts.

This is **Part 2**: SFT, GRPO, evaluation, and how the whole 40-day project cost $0 out of pocket.

```
       ▼
┌──────────────┐   ┌────────────┐   ┌──────────────┐
│     SFT      │──▶│   GRPO     │──▶│  EVALUATE    │
│ AMD MI300X   │   │ 3 GPUs (!) │   │  lm-eval     │
│  (15h)       │   │  (13.5h)   │   │  + custom    │
└──────────────┘   └────────────┘   └──────────────┘
```

---

## Step 5: SFT — teaching it to follow instructions

The base model knows how to predict the next token. It doesn't know how to be helpful. SFT (Supervised Fine-Tuning) is where you show it labeled examples of "user asks → model answers correctly" and let it imitate.

I ran SFT on an **AMD MI300X** (192GB VRAM, ~$2/hour list). I got the time through the [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) program — they hand out $100 in free MI300X credits if you apply. HuggingFace Transformers + TRL's `SFTTrainer`.

The data flow:

```
  Raw dataset (GSM8K, MathInstruct, MetaMathQA)
              │
              ▼
  ┌────────────────────────┐
  │ prepare_sft_data.py    │
  │  - extract Q & A       │
  │  - wrap reasoning in   │
  │    <think>...</think>  │
  │  - apply ChatML        │
  └────────────┬───────────┘
               │
               ▼
  ChatML-formatted text:
  ┌──────────────────────────────────────────────────┐
  │ <|im_start|>system                               │
  │ You are a math assistant...<|im_end|>            │
  │ <|im_start|>user                                 │
  │ What's 12 × 13?<|im_end|>                        │
  │ <|im_start|>assistant                            │
  │ <think>12 × 13 = 12 × 10 + 12 × 3 = 120 + 36 ... │
  │ </think>156<|im_end|>                            │
  └──────────────────────────────────────────────────┘
               │
               ▼
        SFTTrainer (TRL)
```

The actual mix was two stages: Stage 1 = 52k Alpaca examples for a conversational prior (the curriculum mistake — see point 3 below), then Stage 2 = ~662k math examples = MetaMathQA (~395k) + MathInstruct (~260k) + GSM8K train (~7.5k). ChatML-wrapped, two epochs on stage 2.

### The config that worked (mostly)

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir="./sft-out",
    num_train_epochs=2,              # I'd use 5-10 next time
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  # effective batch 64
    learning_rate=2e-5,              # I'd use 5e-6 next time
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    save_strategy="epoch",
    logging_steps=10,
    # The flag I should have set:
    # assistant_only_loss=True,
)
```

### The three things I'd change

**1. Use `assistant_only_loss=True`.** This is the single most important SFT flag. Without it, `SFTTrainer` computes loss over the *entire* sequence — system prompt, user question, assistant answer, all of it.

```
What gets loss without the flag:
   "<|im_start|>system You are a..."  ◄── 30% of loss (boilerplate!)
   "<|im_start|>user What's 12×13?"   ◄── 20% of loss (the question, not the answer)
   "<|im_start|>assistant <think>..." ◄── 50% of loss (the only useful part)

What gets loss with assistant_only_loss=True:
   "<|im_start|>assistant <think>..." ◄── 100% of loss
```

More than half of every gradient was wasted on memorizing my own system prompt. One flag. Three to five percentage points of efficiency.

**2. Initialize new token embeddings properly.** When you `model.resize_token_embeddings()` to add `<think>` and `</think>`, HF initializes the new rows from a small random normal. That's enough to perturb the LM head and regress your general benchmarks. ARC-Easy crashed 29.9% → 25.5% in my SFT. Classic symptom.

The [Hewitt 2021 fix](https://nlp.stanford.edu/~johnhew/vocab-expansion.html) — initialize new rows to the mean of the existing embedding/output matrices:

```python
import torch
orig_vocab_size = model.config.vocab_size
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    emb  = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    mean_in  = emb[:orig_vocab_size].mean(dim=0)
    mean_out = head[:orig_vocab_size].mean(dim=0)
    for tid in tokenizer.convert_tokens_to_ids(["<think>", "</think>"]):
        emb[tid]  = mean_in
        head[tid] = mean_out
```

Three lines. Saves you a 4-percentage-point regression on general benchmarks.

**3. Math first, chat last.** I did Alpaca first, then math, on the intuition that "easy chat warms up the model, hard reasoning specializes it." The DeepSeek-R1 and Tulu-3 reports convinced me afterwards that this is backwards: when reasoning is the goal, train math hard *first* so the chain-of-thought circuits get the bulk of the gradient, then polish with broad chat to keep helpfulness intact. Doing chat first dilutes the reasoning signal and you spend the math epochs partially undoing it. Or just skip the curriculum entirely with a single-stage mix like 70% NuminaMath-CoT + 15% OpenMathInstruct-2 + 10% GSM8K + 5% Tulu-3.

SFT took about 15 hours on the MI300X across 660k examples × 2 epochs. Bill: $0 out of pocket — covered by the AMD Developer Cloud credits.

---

## Step 6: GRPO — when RL gets weird

GRPO (Group Relative Policy Optimization) is the simplest "real" RL algorithm for LLMs. The idea is dead simple:

```
            GRPO in one diagram
            ═══════════════════

Prompt: "What's 12 × 13?"
                │
                ▼  generate G=8 different completions
         ┌──────────────────────────────────────────┐
         │ Completion 1: ... = 156  ✓  reward = 1.0 │
         │ Completion 2: ... = 144  ✗  reward = 0.0 │
         │ Completion 3: ... = 156  ✓  reward = 1.0 │
         │ Completion 4: ... = 169  ✗  reward = 0.0 │
         │ Completion 5: ... = 156  ✓  reward = 1.0 │
         │ ...                                       │
         └─────────────────┬────────────────────────┘
                           ▼
         Mean reward in group: 0.625
         Advantage per sample: reward - 0.625
                           ▼
         Reinforce above-average completions
         Suppress below-average completions
```

That's it. No value network. No critic. Just rollouts and group-relative advantage.

**Why GRPO, not DPO?** DPO needs preference *pairs* — "completion A is better than completion B." With ~1% correctness in my SFT model, almost every pair would be "rejected vs. rejected." No signal. GRPO sidesteps this by working with absolute rewards.

### My GRPO config

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo-out",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    beta=0.01,                    # KL coefficient (anchors to SFT policy)
    num_generations=8,            # group size G
    max_completion_length=512,    # I'd raise to 2048
    temperature=0.9,
    bf16=True,
)

trainer = GRPOTrainer(
    model="./hf_sft_model",
    args=config,
    train_dataset=dataset,
    reward_funcs=[correctness_reward_func, format_reward_func,
                  repetition_penalty_func],
    tokenizer=tokenizer,
)
trainer.train()
```

### Three things you have to add that the tutorial won't tell you

**1. AST-based correctness checking.** Don't string-match math answers. Use SymPy via `math_verify` so `1/2`, `0.5`, and `2/4` all count as equal:

```python
from math_verify import parse, verify

def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        if "</think>" in completion:
            pred = completion.split("</think>", 1)[1].strip()
        else:
            pred = completion.strip()
        try:
            rewards.append(1.0 if verify(parse(gt), parse(pred)) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards
```

**2. Tokenizer decode monkey-patch.** This one took me hours to figure out. TRL internally calls `tokenizer.decode(..., skip_special_tokens=True)` before passing text to your reward functions. That strips `<think>` and `</think>` along with everything else, so your format-reward function never sees them and always returns 0.

```python
_orig_decode = tokenizer.decode

def patched_decode(self, token_ids, **kwargs):
    kwargs["skip_special_tokens"] = False
    text = _orig_decode(token_ids, **kwargs)
    # Strip only ChatML control tokens, preserve reasoning tags
    for t in ["<|im_start|>", "<|im_end|>", "<|bos|>", "<|eos|>", "<|pad|>"]:
        text = text.replace(t, "")
    return text

tokenizer.decode = patched_decode.__get__(tokenizer)
```

**3. N-gram repetition penalty.** When the model collapses into a loop, the unique-3-gram ratio drops below 0.2. I scale a penalty up to -1.5 in those cases. Across 22,419 training steps it never fired — but I'd added it after watching the post-SFT model fall into infinite LaTeX loops on hard prompts (think `e^x + e^x + e^x ...` until it hits max tokens — a classic absorbing-Markov mode collapse on a low-entropy pretraining budget). Cheap insurance against the same mode reappearing under RL pressure. Probably overkill in retrospect, but I'd add it again.

### The reward-shape mistake I'd undo

My total reward was additive: `correctness + format + repetition`. The problem is that with correctness rare (~1.4%) and format common (~50%), format dominated the gradient by **~35×**. The model's rational optimum was: "produce a valid `<think>` shell with any content inside." Exactly the reward-hacking behavior I saw in the outputs.

**The fix is to gate, not sum:**

```python
# What I did (wrong):
total_reward = correctness + format + repetition

# What I should have done:
total_reward = format_pass * (1.0 if correct else -0.1) + repetition_penalty
```

Format becomes a *precondition* for any positive reward, not a competing objective. No format → 0. Bad answer → small negative. Correct → +1.

### The infrastructure odyssey

GRPO ran across **three different GPUs** because of quota issues. The resume-from-checkpoint flow worked perfectly through all of it.

```
  Steps 0 ────── 12,500 ────── 19,000 ────── 22,419
        │           │              │              │
        ▼           ▼              ▼              ▼
  ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
  │ AMD      │ │   HF     │  │ AMD      │ │  GCP     │
  │ MI300X#1 │▶│   Hub    │ ▶│ MI300X#2 │▶│ 2× L4    │
  └──────────┘ │ (resume) │  └──────────┘ │ (CUDA)   │
   quota out   └──────────┘   vLLM/ROCm   └──────────┘
   → SIGINT                   conflicts    ROCm→CUDA hop
```

Total wall time: 13h 32m across the three machines. The two MI300X stints came out of the AMD Developer Cloud credit, and the L4 finish came out of free Google Cloud developer credits. Cash bill: $0.

---

## Step 7: evaluate honestly

For evaluation I used a small A100 instance on [Thunder Compute](https://www.thundercompute.com/), paid out of $25 of free trial credit they hand to new accounts. lm-evaluation-harness is light enough that this is plenty.

```bash
lm_eval --model hf \
    --model_args pretrained=./hf_grpo_model,dtype=bfloat16 \
    --tasks gsm8k,minerva_math,arc_easy,arc_challenge,hellaswag,mmlu \
    --batch_size 8 \
    --apply_chat_template \
    --output_path ./eval_results
```

**The flag everyone misses:** `--apply_chat_template`. If your model was trained with ChatML, the default lm-eval-harness raw few-shot text is out-of-distribution for it. I lost ~1pp on the SFT GSM8K score until I noticed.

For more interesting eval, run **self-consistency** at inference time — sample N=32 completions at temperature 0.7, take the majority-vote final answer. For small models on math this is typically +5-15pp essentially free.

```python
from collections import Counter

def cons_at_n(model, tokenizer, prompt, n=32, temperature=0.7):
    answers = []
    for _ in range(n):
        out = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device),
            do_sample=True, temperature=temperature, max_new_tokens=512,
        )
        answers.append(extract_answer(tokenizer.decode(out[0])))
    return Counter(answers).most_common(1)[0][0]
```

---

## What it actually cost

The honest answer: **$0 out of pocket.** The honest-with-an-asterisk answer: at preemptible/spot pricing this would have run roughly **~$15k**, almost all of it pretraining (on-demand list price would be 3-4× that).

| Stage | Hardware | Wall time | List price | What I paid |
|---|---|---|---|---|
| Data processing | 2× Vultr c2 CPU | ~2 days | ~$15 | $0 (Vultr $150 trial) |
| Pretraining | TPU v4-64 | ~10 days | **~$15k spot / ~$50k on-demand** | **$0** (Google TPU Research Cloud) |
| SFT | AMD MI300X | ~15h | ~$30 | $0 (AMD Developer Cloud, $100 grant) |
| GRPO | MI300X + L4 | ~13.5h | ~$20 | $0 (AMD + Google dev credits) |
| Evaluation | A100 (Thunder Compute) | a few hours | ~$5 | $0 ($25 trial credit) |
| **Total** | | **~40 days** | **~$15k spot / ~$50k on-demand** | **$0** |

Pretraining is ~99% of what the bill *would* have been. Everything else is rounding.

None of these programs are secret. They're public, the applications are short, and they're explicitly meant for this kind of work:

- **[Google TPU Research Cloud](https://sites.research.google/trc/about/)** — free TPU time for open research. Apply with a one-page proposal.
- **[AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)** — $100 of MI300X credits to anyone with a developer account.
- **[Google Cloud free tier + new-user credits](https://cloud.google.com/free)** — what funded the L4 hop.
- **[Vultr free trial](https://www.vultr.com/promo/)** — $150 credit for new accounts.
- **[Thunder Compute](https://www.thundercompute.com/)** — $25 trial credit, enough for an evaluation pass.

If you're an independent researcher or student, stack these. The "I can't afford to train a model" story is mostly out of date.

---

## What I'd tell someone starting this from scratch

The numbered list version, in priority order — the first five are from [Part 1](#); the post-training ones are below:

6. **Use `assistant_only_loss=True` in SFT.** Just do it. It's one flag.
7. **Mean-init new token embeddings**, not random. Three lines of code. Saves you a 4pp regression.
8. **Gate your RL rewards, don't sum them.** Format as precondition, not competing objective.
9. **Have a clean checkpoint resume path** between cloud providers. You'll need it.

The model I shipped isn't a strong math model — 57B tokens just isn't enough at this parameter count, and the tokenizer caps arithmetic. But the *pipeline* is solid, and most of the lessons apply at any scale.

Everything is open source: [github.com/himanshu-nakrani/TinyMathReason-1B](https://github.com/himanshu-nakrani/TinyMathReason-1B). The model weights are on Hugging Face under [himanshunakrani9](https://huggingface.co/himanshunakrani9).

The two files that hold most of the real decisions in this half of the pipeline:

- `src/sft/train_sft.py` — SFT pipeline
- `src/dpo/train_grpo.py` — the hardened GRPO setup, monkey-patches and all

That's the playbook. Now go build something with it.
