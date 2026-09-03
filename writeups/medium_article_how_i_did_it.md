# Here's how I built a 1B language model from scratch



*A friendly tour through the modern LLM stack, with diagrams, the actual decisions I made, and just enough code to be useful.*



---



I'm not going to pretend my model is a leaderboard contender. TinyMathReason-1B gets 2.2% on GSM8K — but for context, peer 1B-class base models on the Open LLM Leaderboard (TinyLlama-1.1B-Chat, Pythia-1B) land in roughly the same low-single-digit band on GSM8K, and TinyLlama did it with ~50× more pretraining tokens than mine. 2.2% on 57B tokens with a digit-unfriendly tokenizer is about what this compute budget buys. The point of this post isn't the score; it's that I built every piece of the pipeline myself — tokenizer up through GRPO reinforcement learning — and the lessons in the pipeline transfer at any scale.



If you've never trained an LLM, you'll come away knowing what each stage *does* and why. If you've shipped one before, skip to whichever section sounds painful and steal the configs.



Here's the whole pipeline at a glance:



```

┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌─────────────┐

│  TOKENIZER   │──▶│    DATA    │──▶│  PRETRAIN    │──▶│   CONVERT   │

│  32k BPE     │   │  ~57B tok  │   │  TPU v4-64   │   │ Orbax → HF  │

│  (1 day)     │   │  (3 days)  │   │  (~10 days)  │   │  (5 days!)  │

└──────────────┘   └────────────┘   └──────────────┘   └─────────────┘

                                                              │

       ┌──────────────────────────────────────────────────────┘

       ▼

┌──────────────┐   ┌────────────┐   ┌──────────────┐

│     SFT      │──▶│   GRPO     │──▶│  EVALUATE    │

│ AMD MI300X   │   │ 3 GPUs (!) │   │  lm-eval     │

│  (15h)       │   │  (13.5h)   │   │  + custom    │

└──────────────┘   └────────────┘   └──────────────┘

```



Seven stages, about 40 days of wall time, and — this is the punchline — **effectively zero out-of-pocket cost**. Every piece of compute came from a research-credit or trial program. Full breakdown at the end. Let's walk through each one.



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

                              │

                              ▼

                      Output logits

```



Key choices:



The diagram covers the standard choices. The three I want to call out, because they're the ones I'd actually defend in a code review:

| Decision | Value | Why it's not the default |
|---|---|---|
| Attention | GQA 4:1 | 4 KV heads vs. 16 query heads — KV cache shrinks 4× at inference |
| Head dim | 128 (not 64) | Llama-3 showed bigger heads capture richer relationships; matters for tracking math variables |
| Vocab | 32k padded to 32,768 | The padding to a power of 2 is for FSDP alignment, not vocab size |



**Why so similar to TinyLlama?** I wanted my *pipeline* to be the variable I was testing, not my architecture. If something went wrong I needed to be able to say "it's not the architecture" with confidence.



The one place I deviated was head dimension. TinyLlama uses 32 query heads at 64-dim each; I used 16 query heads at 128-dim each. Same total attention compute, but Llama-3 showed that bigger heads capture more complex relationships. For math reasoning where you're tracking multiple variables across a derivation, that mattered to me.



> **For beginners:** "Hidden dim" is how wide each layer is. "Heads" are independent attention sub-networks. "GQA 4:1" means 4 keys/values share among 16 queries — saves memory at inference. "RoPE" is how the model knows token position. None of this is magic. Just bookkeeping.



The exact final parameter count: **1,123,117,056**. I computed this by hand before writing any code. If you don't, you'll be staring at a checkpoint six weeks from now wondering why the embedding shape doesn't match.



---



## Step 1: train the tokenizer



The tokenizer is what turns text into integers the model can consume. Sounds simple, hides traps.



A BPE tokenizer learns to merge frequently-co-occurring character sequences into single tokens. "the" becomes one token. "ing" becomes one token. For a math model, the question is: what happens to numbers?



```

                   The tokenizer's job:

                   ═══════════════════



   "The answer is 42"  ──▶  [464, 3280, 318, 5433]  ──▶  model

                            (4 integer tokens)



   But for math:



   "382 + 491 = 873"  ──▶  [???, ???, ???, ...]  ──▶  ?

                              ▲

                              │

                       This is where you can ruin your model.

```



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



I also wrote a small converter (`src/data/convert_tokenizer.py`) to export the trained tokenizer to `.tiktoken` format, because MaxText doesn't load HuggingFace tokenizers natively.



---



## Step 2: build the data pipeline



The boring stage nobody writes about. Roughly a third of the total work.



I rented two Vultr c2-standard-30 CPU instances ($1.50/hour each, paid out of $150 in trial credits Vultr gives new accounts) and ran the pipeline in parallel. Six stages, six Python scripts:



```

┌──────────────────────────────────────────────────────────────────┐

│                       DATA PIPELINE                              │

└──────────────────────────────────────────────────────────────────┘



   ┌────────────────┐

   │  a_download    │   Stream datasets from HuggingFace

   │                │   FineWeb-Edu | OpenWebMath | MathPile | Stack-Edu

   └───────┬────────┘

           │

           ▼

   ┌────────────────┐

   │ b_clean_filter │   word_count > 20, alpha_ratio > 0.3

   │                │   hash-based dedup

   └───────┬────────┘

           │

           ▼

   ┌────────────────┐

   │   c_mix        │   Weighted interleave, dataset folders → ratios

   │                │   Auto-normalizes from disk contents

   └───────┬────────┘

           │

           ▼

   ┌────────────────┐

   │ d_tokenize     │   Tokenize, concat with EOS,

   │ _and_pack      │   pack into 4096-token sequences

   └───────┬────────┘

           │

           ▼

   ┌────────────────┐

   │  e_shards      │   Split into ~50MB .jsonl.zst shards

   └───────┬────────┘

           │

           ▼

   ┌────────────────┐

   │  f_upload      │   rsync to GCS bucket for TPU consumption

   └────────────────┘

```



### The corpus mix (and a happy accident)



The mix I actually ended up training on:



- **FineWeb-Edu: 40%** — general educational web text, the bread and butter

- **OpenWebMath: 35%** — the high-quality math web corpus

- **MathPile: 15%** — curated math textbooks and papers

- **Stack-Edu: 10%** — educational code

- ~~Proof-Pile-2: 15%~~ — *planned, but never downloaded*



*Note: I originally planned a 5-dataset mix that included Proof-Pile-2 (15%). Due to an oversight in the download script, it was never fetched. However, my mixing script dynamically auto-normalized probabilities based on folders present on disk. The result was a "happy accident" that concentrated my high-quality math web data to **35% OpenWebMath** (up from the planned 30% ratio)!*



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

        (each document attends only to itself)

```



Fix: use FlashAttention's varlen mode, or emit segment IDs and add a block-diagonal attention mask.



**Real MinHash, not exact-string dedup.** I claimed MinHash in my docs. The code shipped `hash(text)` exact-string dedup. Result: lots of near-duplicate web pages slipped through. The `datasketch` library makes real MinHash trivial; I just didn't bother.



Total cost for this stage: $0 out of pocket — well inside Vultr's $150 trial credit. Compute isn't the expensive part. The expensive part is your future self debugging your past self's shortcuts.



---



## Step 3: TPU pretraining



> **If you read nothing else in this section, read this.** Two flags in `maxtext_config.yml` are the difference between a working 1B run and a week of wasted compute:
>
> - `scan_layers: False` — `True` is faster but in the MaxText/NNX combo I was on it silently produced checkpoints with **zero transformer layers**. I lost a week to this.
> - `per_device_batch_size: 2` — `8` triggers XLA host-RAM allocations large enough to OOM the host (not the HBM) during compilation.
>
> Set both before you launch. Details and the rest of the config below.



This is the part that *would* cost real money.



List price for a TPU v4-64 is in the thousands per week. I didn't pay any of it. I got access through Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program, which grants free TPU time for open research. Apply with a one-page proposal — they really do approve individuals.



I trained on a TPU v4-64 — 64 chips spread across 8 host VMs — using the preemptible queue that TRC allocations run on. About a week of compute, ~10 days of wall time after preemption restarts.



```

                       TPU v4-64 topology

                       ═══════════════════



              ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐

              │ Host │ │ Host │ │ Host │ │ Host │

              │  0   │ │  1   │ │  2   │ │  3   │

              │ 8 ch │ │ 8 ch │ │ 8 ch │ │ 8 ch │

              └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘

                 │        │        │        │

              ───┴────────┴────────┴────────┴───  high-speed mesh

                 │        │        │        │

              ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐

              │ Host │ │ Host │ │ Host │ │ Host │

              │  4   │ │  5   │ │  6   │ │  7   │

              └──────┘ └──────┘ └──────┘ └──────┘



         64 chips total. Single JAX distributed mesh.

         If one host reboots → entire mesh collapses.

```



### Setup: the parts they don't put in the tutorial



I wrote `scripts/setup_tpu.sh` to do everything on every worker:



1. Install Python 3.12 (MaxText needs it; the default 3.10 segfaults instantly)

2. Create a `venv312` virtualenv per worker

3. Clone MaxText, run their `setup.sh`

4. Apply two **critical** compatibility shims:

   - Alias `jax.Ref → typing.Any` (JAX 0.6.2 removed `jax.Ref`, MaxText still imports it)

   - Register a `MetaPathFinder` that spoofs Google-internal Pallas modules, so `from jax.experimental.pallas.ops.tpu.splash_attention import ...` succeeds without crashing on the public JAX



The Pallas thing in particular ate me alive. Python 3.12's strict import system wouldn't let me stub these modules with simple `MagicMock` — it complains about `__spec__` and `__path__`. The fix is a proper `MetaPathFinder` subclass that returns a real-looking module spec.



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



**Two flags carried more weight than anything else.** Burn them into your retinas:



- `scan_layers: False` — `True` is faster, but in my setup (MaxText `main` ~May 2026, JAX/jaxlib 0.6.2, `pure_nnx_decoder: True`) it silently produced checkpoints with zero transformer layers. I lost a week to this. Worth re-checking on newer MaxText, but until you've verified your layers actually got serialized, leave it `False`.

- `per_device_batch_size: 2` — On the same stack (JAX 0.6.2, MaxText `main` ~May 2026) XLA compilation of a 1B model with batch=8 wanted *>300GB of host RAM* (not HBM, host RAM). Newer JAX may have improved this, but batch=2 fits comfortably and gives an effective batch of 64 sequences × 4096 tokens = **~262k tokens/step**.



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

- **Bill:** $0 out of pocket (TPU Research Cloud grant). At preemptible/spot pricing this would have been ~$15k (and ~$50k at on-demand list).

- **Final checkpoint:** `gs://your-bucket/checkpoints/run11/checkpoints/54362/`



---



## Step 4: convert the checkpoint to HuggingFace



This was the single most technically demanding part of the project. I budgeted a day. It took five.



MaxText saves checkpoints in **Orbax** format — a JAX-native serialization with arrays stored as **Zarr** chunks in GCS. HuggingFace wants **safetensors** in PyTorch tensor layout. These aren't just different file formats. They're different *tensor layouts* with different scaling conventions.



```

   MaxText/Orbax checkpoint               HuggingFace safetensors

   ════════════════════════               ═══════════════════════



   ┌──────────────────────┐               ┌──────────────────────┐

   │ params/              │               │ model.embed_tokens.  │

   │  decoder/            │               │   weight             │

   │   layers/            │               │ model.layers.0.      │

   │    self_attention/   │   convert     │   self_attn.q_proj.  │

   │     query/kernel ◄───┤   ────────▶   │   weight             │

   │     key/kernel       │               │ model.layers.0.      │

   │     value/kernel     │               │   self_attn.k_proj.  │

   │     out/kernel       │               │   weight             │

   │    mlp/...           │               │ ...                  │

   │  embedder/...        │               │                      │

   └──────────────────────┘               └──────────────────────┘

   ▲ Stacked layers in       ▲ Per-layer tensors. RoPE

     single array. Query       interleaving differs. Q

     scaling baked into        scaling applied at runtime.

     weights at save time.

```



### Write the inspector first



This is the single most important piece of advice I can give you here. **Write the inspector before you write the converter.** You will iterate on the converter 5-10 times. Each iteration is bearable if you can see exactly what's in the source checkpoint.



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

...

```



Now you can see exactly what's in there. Without it you're debugging blind.



### The five bugs in the converter



Remember the "5 days" in the pipeline diagram? Here's where they went. In rough order of how long they took me to find:



**1. Vocab padding.** MaxText padded my 32k vocab to 32,768 for FSDP alignment. HF needs `vocab_size=32768` explicitly. Forget this and your embedding matrix is the wrong shape, which fails loudly. Easy bug.



**2. Query scaling.** MaxText bakes the `1/√head_dim` attention scaling *into* the query weights at save time. HF applies it at runtime. So when you load MaxText Q weights into an HF model, the scaling gets applied twice. Fix: multiply Q weights by `√head_dim` during conversion.



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



You don't need a GPU. CPU with ~8GB RAM is plenty.



```bash

python src/train/inspect_checkpoint.py \

    --orbax_dir gs://your-bucket/checkpoints/run11/checkpoints/54362



python src/train/convert_checkpoint.py \

    --orbax_dir gs://your-bucket/checkpoints/run11/checkpoints/54362 \

    --hf_out_dir ./hf_1b_model \

    --tokenizer_path ./tokenizer

```



Then verify with a forward pass — this is the moment of truth:



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



## Step 5: SFT — teaching it to follow instructions



The base model knows how to predict the next token. It doesn't know how to be helpful. SFT (Supervised Fine-Tuning) is where you show it labeled examples of "user asks → model answers correctly" and let it imitate.



I ran SFT on an **AMD MI300X** (192GB VRAM, ~$2/hour list). I got the time through the [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) program — they hand out $100 in free MI300X credits if you apply. HuggingFace Transformers + TRL's `SFTTrainer`.



```

              SFT data flow

              ═════════════



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

  │ You are a math assistant...                      │

  │ <|im_end|>                                       │

  │ <|im_start|>user                                 │

  │ What's 12 × 13?                                  │

  │ <|im_end|>                                       │

  │ <|im_start|>assistant                            │

  │ <think>12 × 13 = 12 × 10 + 12 × 3 = 120 + 36 ... │

  │ </think>156<|im_end|>                            │

  └──────────────────────────────────────────────────┘

               │

               ▼

        SFTTrainer (TRL)

               │

               ▼

        Fine-tuned model

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

         │ Completion 6: ... = 130  ✗  reward = 0.0 │

         │ Completion 7: ... = 156  ✓  reward = 1.0 │

         │ Completion 8: ... = 156  ✓  reward = 1.0 │

         └─────────────────┬────────────────────────┘

                           │

                           ▼

         Mean reward in this group: 0.625

         Advantage per sample: reward - 0.625

                           │

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

        GRPO infrastructure migration

        ═════════════════════════════



  Steps 0 ────────── 12,500 ────────── 19,000 ────────── 22,419

        │              │                  │                │

        ▼              ▼                  ▼                ▼

  ┌──────────┐    ┌──────────┐      ┌──────────┐    ┌──────────┐

  │ AMD      │    │   HF     │      │ AMD      │    │  GCP     │

  │ MI300X   │ ─▶ │   Hub    │ ──▶  │ MI300X   │ ──▶│ 2× L4    │

  │ #1       │    │ (resume) │      │ #2       │    │ (CUDA)   │

  └──────────┘    └──────────┘      └──────────┘    └──────────┘

   quota out      checkpoint         vLLM/ROCm       ROCm→CUDA

   → SIGINT       backed up          conflicts       hop, works!

```



Total wall time: 13h 32m across the three machines. The two MI300X stints came out of the AMD Developer Cloud credit, and the L4 finish came out of free Google Cloud developer credits, so the cash bill on this stage was also $0.



---



## Step 7: evaluate honestly



For evaluation I used a small A100 instance on [Thunder Compute](https://www.thundercompute.com/), paid out of $25 of free trial credit they hand to new accounts. lm-evaluation-harness is light enough that this is plenty.



Run `lm-evaluation-harness` for benchmark numbers:



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

        answer = extract_answer(tokenizer.decode(out[0]))

        answers.append(answer)

    # Majority vote

    return Counter(answers).most_common(1)[0][0]

```



---



## What it actually cost



The honest answer: **$0 out of pocket.** The honest-with-an-asterisk answer: at preemptible/spot pricing this would have run roughly **~$15k**, almost all of it pretraining (on-demand list price would be 3-4× that). Here's what powered each stage and where the credits came from:



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

- **[Vultr free trial](https://www.vultr.com/promo/)** — $150 credit for new accounts, plenty for CPU data work.

- **[Thunder Compute](https://www.thundercompute.com/)** — $25 trial credit, enough for an evaluation pass.



If you're an independent researcher or student, stack these. The "I can't afford to train a model" story is mostly out of date.



---



## What I'd tell someone starting this from scratch



The numbered list version, in priority order:



1. **Architecture: copy a proven one.** TinyLlama or Llama-3.2 shape. Don't innovate here.

2. **Audit your tokenizer before pretraining.** Especially digit behavior. Cheapest decision to get right, most expensive to fix later.

3. **Treat your data pipeline like real software.** Tests, logs, sampled outputs. The README/code gap is real and you will pay for it.

4. **Read MaxText (or whatever) end-to-end** before you launch the first multi-day training run — even on free credits, your time is the limited resource. Find the config bugs in the documentation, not in your checkpoint.

5. **Write the checkpoint inspector before the checkpoint converter.** You'll iterate the converter 5-10 times. Each iteration is bearable if you can see the source.

6. **Use `assistant_only_loss=True` in SFT.** Just do it. It's one flag.

7. **Mean-init new token embeddings**, not random. Three lines of code. Saves you a 4pp regression.

8. **Gate your RL rewards, don't sum them.** Format as precondition, not competing objective.

9. **Have a clean checkpoint resume path** between cloud providers. You'll need it.



The model I shipped isn't a strong math model — 57B tokens just isn't enough at this parameter count, and the tokenizer caps arithmetic. But the *pipeline* is solid, and most of the lessons apply at any scale.



```

                  Final picture

                  ═════════════



         ┌────────────────────────────────────┐

         │  TinyMathReason-1B                 │

         │                                    │

         │  • 1.12B params                    │

         │  • 57B pretraining tokens          │

         │  • SFT + GRPO post-training        │

         │  • 5 cloud providers visited       │

         │  • ~40 days, $0 cash (~$15k spot)  │

         │                                    │

         │  GSM8K: 2.2% (in band for 57B tok) │

         │  Pipeline: shippable               │

         └────────────────────────────────────┘

```



Everything is open source: [github.com/himanshu-nakrani/TinyMathReason-1B](https://github.com/himanshu-nakrani/TinyMathReason-1B). The model weights are on Hugging Face under [himanshunakrani9](https://huggingface.co/himanshunakrani9).



If you want to actually look at the code while reading this article, the four files that hold most of the real decisions are:



- `src/train/maxtext_config.yml` — every pretraining flag

- `src/train/convert_checkpoint.py` — all five conversion bugs and how I fixed them

- `src/sft/train_sft.py` — SFT pipeline

- `src/dpo/train_grpo.py` — the hardened GRPO setup, monkey-patches and all



That's the playbook. Now go build something with it.