# How I pretrained a 1B language model for $0 (Part 1 of 2)

*Tokenizer, data, TPU pretraining, and the five-day fight to get the checkpoint into HuggingFace.*

---

I built a 1B language model — TinyMathReason-1B — from scratch. Tokenizer up through GRPO reinforcement learning. The final model gets 2.2% on GSM8K, which is roughly in the band peer 1B-class models land in (TinyLlama-1.1B-Chat, Pythia-1B), and TinyLlama needed ~50× more pretraining tokens to get there. The point of this post isn't the score; it's the pipeline, and the half-dozen places along it where a wrong default or a silent bug can quietly cost you a week.

This is **Part 1 of 2**. It covers the pretraining half — tokenizer, data, TPU run, and the painful Orbax → HuggingFace conversion. [Part 2](#) covers SFT, GRPO, and evaluation.

Here's the half of the pipeline this post covers:

```
┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌─────────────┐
│  TOKENIZER   │──▶│    DATA    │──▶│  PRETRAIN    │──▶│   CONVERT   │
│  32k BPE     │   │  ~57B tok  │   │  TPU v4-64   │   │ Orbax → HF  │
│  (1 day)     │   │  (3 days)  │   │  (~10 days)  │   │  (5 days!)  │
└──────────────┘   └────────────┘   └──────────────┘   └─────────────┘
```

About three weeks of wall time, and — this is the punchline — **$0 out of pocket**. TPU time came from Google's TPU Research Cloud; CPU data work came from a Vultr trial credit. At preemptible/spot pricing this half of the pipeline would have cost roughly **~$15k** (and ~$50k at on-demand list). Full cost breakdown is in Part 2.

---

## Step 0: pick the architecture

Before I touched a keyboard I spent two days writing down four things on one page: target architecture, data sources, infra budget, and how I'd decide the run had succeeded. That's it — no Gantt chart, no design doc. The point isn't ceremony; it's that you can write a tokenizer in an afternoon but the consequences of that tokenizer follow you for weeks. Cheap to plan, expensive to redo.

I went with a Llama-2 lookalike. Here's the shape:

```
                   TinyMathReason-1B
                   ════════════════
                       1.12B parameters

       Input tokens → [Embedding 32k × 2048]
                              │
                              ▼
            ┌─────────────────────────────────┐
            │  Transformer Layer × 22         │
            │  ┌───────────────────────────┐  │
            │  │ RMSNorm                   │  │
            │  │ ↓                         │  │
            │  │ GQA (16 Q heads, 4 KV)    │  │ ◄── 4:1 ratio
            │  │ ↓ + residual              │  │     shrinks KV cache 4×
            │  │ RMSNorm                   │  │
            │  │ ↓                         │  │
            │  │ SwiGLU MLP (5632 dim)     │  │
            │  │ ↓ + residual              │  │
            │  └───────────────────────────┘  │
            └─────────────────────────────────┘
                              │
                              ▼
                       [Final RMSNorm]
                              │
                              ▼
                       [LM Head 2048 × 32k]
```

The diagram covers the standard choices. The three I want to call out, because they're the ones I'd actually defend in a code review:

| Decision | Value | Why it's not the default |
|---|---|---|
| Attention | GQA 4:1 | 4 KV heads vs. 16 query heads — KV cache shrinks 4× at inference |
| Head dim | 128 (not 64) | Llama-3 showed bigger heads capture richer relationships; matters for tracking math variables |
| Vocab | 32k padded to 32,768 | The padding to a power of 2 is for FSDP alignment, not vocab size |

**Why so similar to TinyLlama?** I wanted my *pipeline* to be the variable I was testing, not my architecture. If something went wrong I needed to be able to say "it's not the architecture" with confidence.

The exact final parameter count: **1,123,117,056**. I computed this by hand before writing any code. If you don't, you'll be staring at a checkpoint six weeks from now wondering why the embedding shape doesn't match.

---

## Step 1: train the tokenizer

The tokenizer is what turns text into integers the model can consume. Sounds simple, hides traps.

A BPE tokenizer learns to merge frequently-co-occurring character sequences into single tokens. "the" becomes one token. "ing" becomes one token. For a math model, the question is: what happens to numbers?

Here's the code I ran:

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[...],
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
                    "<|im_start|>", "<|im_end|>"]
)
```

Reserved `<|im_start|>` and `<|im_end|>` for ChatML templating, and saved `<think>` / `</think>` for adding later during SFT.

**Here's the test I should have run before training:**

```python
for s in ["1234567", "382 + 491 = 873", "1/2 + 3/4"]:
    print(s, "->", tokenizer.encode(s).tokens)
```

If multi-digit numbers stay as single tokens or split at unpredictable boundaries, fix it now. The fix is either:

- **Pre-split every digit** (Llama-3, DeepSeek-Math do this). Now "42" becomes `["4", "2"]` always.
- **Use a much larger vocab** (Qwen2.5 uses 151k). With more vocabulary slots the merges become stable.

I did neither, and the consequences cascaded through every later stage. My audit later showed "100" → 1 token, "2024" → 2 tokens, "1234567" → 3 chunks at arbitrary boundaries. Inconsistent chunking is the *worst* case for arithmetic. Don't ship this.

---

## Step 2: build the data pipeline

The boring stage nobody writes about. Roughly a third of the total work.

I rented two Vultr c2-standard-30 CPU instances ($1.50/hour each, paid out of $150 in trial credits Vultr gives new accounts) and ran the pipeline in parallel. Six stages, six Python scripts: download → clean/filter → mix → tokenize-and-pack → shard → upload to GCS.

### The corpus mix (and a happy accident)

The mix I actually ended up training on:

- **FineWeb-Edu: 40%** — general educational web text
- **OpenWebMath: 35%** — the high-quality math web corpus
- **MathPile: 15%** — curated math textbooks and papers
- **Stack-Edu: 10%** — educational code
- ~~Proof-Pile-2: 15%~~ — *planned, but never downloaded*

I originally planned a 5-dataset mix including Proof-Pile-2. Due to an oversight in the download script, it was never fetched. My mixing script auto-normalized probabilities based on folders present on disk, which concentrated my high-quality math web data to **35% OpenWebMath** (up from a planned 30%) — a happy accident, not a happy plan.

A Makefile drives the whole pipeline. `make all` end-to-end. The output was ~1000 shards averaging 50MB compressed each, totaling **~57 billion tokens**.

### Two things I'd change next time

**Document-boundary attention masks.** I packed sequences by concatenating documents with an EOS token between them. The problem is that vanilla attention can still flow *across* documents inside a single packed sequence. For math, that's especially bad — each problem should be self-contained.

```
What I did:
   [doc A tokens][EOS][doc B tokens][EOS][doc C tokens]
   ◄────────── attention can see everything ──────────►

What I should have done:
   [doc A tokens][EOS][doc B tokens][EOS][doc C tokens]
   ◄── doc A ──►     ◄── doc B ──►     ◄── doc C ──►
```

Fix: use FlashAttention's varlen mode, or emit segment IDs and add a block-diagonal attention mask.

**Real MinHash, not exact-string dedup.** I claimed MinHash in my docs. The code shipped `hash(text)` exact-string dedup. Result: lots of near-duplicate web pages slipped through. The `datasketch` library makes real MinHash trivial; I just didn't bother.

The expensive part of data work isn't compute. It's your future self debugging your past self's shortcuts.

---

## Step 3: TPU pretraining

> **If you read nothing else in this section, read this.** Two flags in `maxtext_config.yml` are the difference between a working 1B run and a week of wasted compute:
>
> - `scan_layers: False` — `True` is faster but in the MaxText/NNX combo I was on it silently produced checkpoints with **zero transformer layers**. I lost a week to this.
> - `per_device_batch_size: 2` — `8` triggers XLA host-RAM allocations large enough to OOM the host (not the HBM) during compilation.

This is the part that *would* cost real money. List price for a TPU v4-64 is in the thousands per week. I got access through Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program, which grants free TPU time for open research. Apply with a one-page proposal — they really do approve individuals.

I trained on a TPU v4-64 — 64 chips spread across 8 host VMs — using the preemptible queue that TRC allocations run on. About a week of compute, ~10 days of wall time after preemption restarts.

### Setup: the parts they don't put in the tutorial

I wrote `scripts/setup_tpu.sh` to do everything on every worker:

1. Install Python 3.12 (MaxText needs it; the default 3.10 segfaults instantly)
2. Create a `venv312` virtualenv per worker
3. Clone MaxText, run their `setup.sh`
4. Apply two **critical** compatibility shims:
   - Alias `jax.Ref → typing.Any` (JAX 0.6.2 removed `jax.Ref`, MaxText still imports it)
   - Register a `MetaPathFinder` that spoofs Google-internal Pallas modules, so `from jax.experimental.pallas.ops.tpu.splash_attention import ...` succeeds without crashing on the public JAX

The Pallas thing in particular ate me alive. Python 3.12's strict import system wouldn't let me stub these modules with simple `MagicMock`. The fix is a proper `MetaPathFinder` subclass that returns a real-looking module spec.

### The actual config

`src/train/maxtext_config.yml`:

```yaml
model_name: tinymath-1b

# Architecture
base_emb_dim: 2048
base_num_query_heads: 16
base_num_kv_heads: 4
base_mlp_dim: 5632
base_num_decoder_layers: 22
head_dim: 128
vocab_size: 32768          # padded for FSDP alignment
max_target_length: 4096

# The flags that mattered most
per_device_batch_size: 2   # 8 OOMs host RAM during compile
pure_nnx_decoder: True
scan_layers: False         # True silently drops all layers!

# Optimizer
learning_rate: 3.0e-4
adam_b1: 0.9
adam_b2: 0.95
adam_weight_decay: 0.1
warmup_steps_fraction: 0.0066
cosine_learning_rate_final_fraction: 0.1
steps: 54363
```

Expanding on the two flags from the callout above:

- `scan_layers: False` — `True` is faster, but in my setup (MaxText `main` ~May 2026, JAX/jaxlib 0.6.2, `pure_nnx_decoder: True`) it silently produced checkpoints with zero transformer layers. Worth re-checking on newer MaxText, but until you've verified your layers actually got serialized, leave it `False`.
- `per_device_batch_size: 2` — On the same stack, XLA compilation with batch=8 wanted *>300GB of host RAM* (not HBM, host RAM). Batch=2 fits comfortably and gives an effective batch of 64 sequences × 4096 tokens = **~262k tokens/step**.

### Launching the run

```bash
gcloud compute tpus tpu-vm ssh tinymath-tpu \
  --worker=all --zone=us-central2-b \
  --command="cd ~/maxtext && source ~/venv312/bin/activate && \
             python MaxText/train.py MaxText/configs/tinymath_1b.yml \
                    run_name=tinymath-1b-prod-run11"
```

`--worker=all` is the magic word. It runs the command on every host in parallel, which is what you need to bring up the JAX distributed mesh.

### Handling preemption

Spot TPUs get preempted. A lot. I ran `src/train/preemption_handler.py` in the background — it polls the GCP metadata server for preemption notices and logs them. MaxText's built-in checkpointing handles the actual resume.

If a worker reboots while the others don't, the JAX mesh collapses with `SSH 255` errors. The fix is a *rolling reboot of all 8 hosts followed by a 5-minute sync wait* so they all come up at nearly identical uptimes. I have a Makefile target for it now.

### What came out

- **Throughput:** ~8,900 tokens/sec/chip, ~66 TFLOP/s/device
- **Total tokens:** ~57 billion
- **Final loss:** ~2.6 (stable convergence)
- **Bill:** $0 out of pocket. At preemptible/spot pricing this would have been ~$15k (and ~$50k at on-demand list).

---

## Step 4: convert the checkpoint to HuggingFace

This was the single most technically demanding part of the project. I budgeted a day. It took five.

MaxText saves checkpoints in **Orbax** format — a JAX-native serialization with arrays stored as **Zarr** chunks in GCS. HuggingFace wants **safetensors** in PyTorch tensor layout. These aren't just different file formats. They're different *tensor layouts* with different scaling conventions.

```
   MaxText/Orbax checkpoint               HuggingFace safetensors
   ════════════════════════               ═══════════════════════

   Stacked layers in single array.        Per-layer tensors.
   Query scaling baked into               RoPE interleaving differs.
   weights at save time.                  Q scaling applied at runtime.
```

### Write the inspector first

This is the single most important piece of advice in this whole post. **Write the inspector before you write the converter.** You will iterate on the converter 5-10 times. Each iteration is bearable if you can see exactly what's in the source checkpoint.

`inspect_checkpoint.py`:

```python
from orbax.checkpoint import PyTreeCheckpointer

ckpt = PyTreeCheckpointer().restore(checkpoint_dir)

def walk(obj, prefix=""):
    if hasattr(obj, "shape"):
        print(f"{prefix}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{prefix}/{k}")

walk(ckpt)
```

That's the whole thing. Output looks like:

```
/params/decoder/layers/self_attention/query/kernel: shape=(22, 2048, 16, 128)
/params/decoder/layers/self_attention/key/kernel:   shape=(22, 2048, 4, 128)
/params/decoder/embedder/embedding:                 shape=(32768, 2048)
```

Now you can see exactly what's in there. Without it you're debugging blind.

### The five bugs in the converter

Remember the "5 days" in the pipeline diagram? Here's where they went. In rough order of how long they took me to find:

**1. Vocab padding.** MaxText padded my 32k vocab to 32,768 for FSDP alignment. HF needs `vocab_size=32768` explicitly. Easy bug — fails loudly.

**2. Query scaling.** MaxText bakes the `1/√head_dim` attention scaling *into* the query weights at save time. HF applies it at runtime. So when you load MaxText Q weights into an HF model, the scaling gets applied twice. Fix:

```python
import math
q_weight = orbax_q_kernel * math.sqrt(head_dim)
```

**3. GQA tensor splitting.** MaxText stores Q, K, V as stacked tensors with the layer index as a dimension. The shapes differ between Q (16 heads) and KV (4 heads). Splitting them correctly is just careful slicing, but I got it wrong three times before I got it right.

**4. RoPE interleaving.** This is the bug that produces eerily convincing nonsense. MaxText and HuggingFace use different interleaving orders for rotary position embeddings. The model still generates fluent-looking English, but the meaning is gone. The fix is a permutation:

```python
def maxtext_to_hf_rope_perm(w, head_dim):
    w = w.reshape(-1, head_dim)
    perm = list(range(0, head_dim, 2)) + list(range(1, head_dim, 2))
    return w[:, perm].reshape(w.shape[0], head_dim)
```

If your converted model outputs grammatical word salad, this is almost certainly it.

**5. Tokenizer config.** My `tokenizer_config.json` had `"tokenizer_class": "TokenizersBackend"`, which isn't a real class. `AutoTokenizer` was failing *silently*. Change it to `"PreTrainedTokenizerFast"`.

### Running the conversion

You don't need a GPU. CPU with ~8GB RAM is plenty. Then verify with a forward pass — this is the moment of truth:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./hf_1b_model", torch_dtype="bfloat16")
tok = AutoTokenizer.from_pretrained("./hf_1b_model")
out = model.generate(**tok("The square root of 16 is", return_tensors="pt"),
                     max_new_tokens=20)
print(tok.decode(out[0]))
```

Fluent-looking text means your conversion is structurally correct. Garbled bytes mean it's not. Word salad in correct English means RoPE.

---

## What's next

At this point you have a base model that can predict tokens. It doesn't know how to be helpful, it doesn't know to put answers after `<think>` tags, and it'll happily fall into infinite LaTeX loops on hard prompts. That's all post-training, and that's [Part 2](#): SFT, GRPO, evaluation, the full cost breakdown across all five free-credit programs I stacked, and the priority-ordered list of lessons I'd actually pass on.

### Lessons from pretraining alone

1. **Architecture: copy a proven one.** TinyLlama or Llama-3.2 shape. Don't innovate here.
2. **Audit your tokenizer before pretraining.** Especially digit behavior. Cheapest decision to get right, most expensive to fix later.
3. **Treat your data pipeline like real software.** Tests, logs, sampled outputs. The README/code gap is real and you will pay for it.
4. **Read MaxText (or whatever) end-to-end** before you launch the first multi-day training run — even on free credits, your time is the limited resource. Find the config bugs in the documentation, not in your checkpoint.
5. **Write the checkpoint inspector before the checkpoint converter.** You'll iterate the converter 5-10 times. Each iteration is bearable if you can see the source.

Everything is open source: [github.com/himanshu-nakrani/TinyMathReason-1B](https://github.com/himanshu-nakrani/TinyMathReason-1B). The two files that hold most of the real decisions in this half of the pipeline:

- `src/train/maxtext_config.yml` — every pretraining flag
- `src/train/convert_checkpoint.py` — all five conversion bugs and how I fixed them

[Continue to Part 2: SFT, GRPO, and the full cost story →](#)
