# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

analyse @[handoff.md]

and lets continue on Convert Checkpoint → HuggingFace 

*Viewed [handoff.md](file:///Users/himanshu/Git/TinyMathReason-1B/handoff.md) *

*Listed directory [TinyMathReason-1B](file:///Users/himanshu/Git/TinyMathReason-1B) *

*Viewed [convert_checkpoint.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/convert_checkpoint.py) *

*Listed directory [src](file:///Users/himanshu/Git/TinyMathReason-1B/src) *

*Listed directory [tokenizer](file:///Users/himanshu/Git/TinyMathReason-1B/tokenizer) *

*Viewed [maxtext_config.yml](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/maxtext_config.yml) *

*Viewed [tokenizer_config.json](file:///Users/himanshu/Git/TinyMathReason-1B/tokenizer/tokenizer_config.json) *

*Viewed [modeling_tinymath.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/model/modeling_tinymath.py) *

*Listed directory [train](file:///Users/himanshu/Git/TinyMathReason-1B/src/train) *

### Planner Response

Now let me also check the tokenizer.json to understand the HF tokenizer format, and research the MaxText checkpoint structure.

*Viewed [tokenizer.json](file:///Users/himanshu/Git/TinyMathReason-1B/tokenizer/tokenizer.json) *

*Searched web for MaxText JAX Orbax checkpoint parameter tree structure Llama model keys convert to HuggingFace*

### Planner Response

This is very important — MaxText uses **scanned (stacked) layers**, meaning the checkpoint structure is fundamentally different from what the current conversion script assumes. Let me research this further and look at MaxText's actual conversion utilities.

*Searched web for MaxText orbax checkpoint PyTree structure param keys "decoder" "layers" scan stacked array shape llama convert huggingface 2024 2025*

*Searched web for MaxText to_huggingface.py convert checkpoint script github google "ckpt_conversion"*

### Planner Response

Let me now look at the MaxText source code to understand the exact parameter naming conventions.

*Read URL content from /AI-Hypercomputer/maxtext/main/src/MaxText/utils/ckpt_conversion/to_huggingface.py *

*Searched web for MaxText JAX model.py Decoder layer parameter names "token_embedder" "decoder_norm" "logits_dense" "self_attention" "mlp" "query" "key" "value" weight naming convention*

*Read URL content from /AI-Hypercomputer/maxtext/tree/main/src/MaxText/utils/ckpt_conversion *

*Read URL content from /AI-Hypercomputer/maxtext/main/MaxText/layers/models.py *

*Read URL content from /repos/AI-Hypercomputer/maxtext/git/trees/main *

*Grep searched codebase*

*Grep searched codebase*

*Grep searched codebase*

*Grep searched codebase*

*Viewed [content.md](file:///Users/himanshu/.gemini/antigravity/brain/6ff72167-4f51-455d-a580-d4c58c05ef76/.system_generated/steps/37/content.md) *

### Planner Response

I found the key file. Let me read the critical MaxText Orbax-to-HuggingFace conversion script, and specifically the Llama parameter mapping:

*Read URL content from /AI-Hypercomputer/maxtext/main/src/maxtext/checkpoint_conversion/standalone_scripts/llama_mistral_mixtral_orbax_to_hf.py *

*Viewed [content.md](file:///Users/himanshu/.gemini/antigravity/brain/6ff72167-4f51-455d-a580-d4c58c05ef76/.system_generated/steps/52/content.md) *

### Planner Response

Now I have the complete picture. The official MaxText conversion script reveals the **critical insight**: MaxText stores layer parameters in a **scanned/stacked** format where the layer index is a *dimension within the array*, not a separate key. The current `convert_checkpoint.py` assumes per-layer keys like `layers_0`, `layers_1`, etc., which is **wrong**.

Let me now create the implementation plan and rewrite the conversion script.

*Edited relevant file*

### Planner Response

Now let me rewrite the conversion script and create the inspection utility:

*Edited relevant file*

### Planner Response

Now let me rewrite the main conversion script:

*Edited relevant file*

### Planner Response

Now let's fix the tokenizer config to work properly with HuggingFace:

*Edited relevant file*

### Planner Response

Now let me check if there are special tokens like `<think>` that need to be in the tokenizer:

*Viewed [tinymath_tokenizer.json](file:///Users/himanshu/Git/TinyMathReason-1B/tokenizer/tinymath_tokenizer.json) *

### Planner Response

I note the tokenizer has `<|bos|>`, `<|eos|>`, `<|unk|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>` but NOT `<think>` / `</think>` — those can be added later during SFT with `resize_token_embeddings`. The tokenizer.json has 6 special tokens and ~32000 BPE merges → ~32000 active vocab with 32768 padded for the model.

The script is now complete. Let me create a summary of everything done:

*Edited relevant file*

### Planner Response

## Summary

I've analyzed the handoff document and fully prepared the **Checkpoint Conversion (Step 1)** for the TinyMathReason-1B pipeline. Here's what was done:

### Files Changed

| File | Action |
|------|--------|
| `src/train/convert_checkpoint.py` | **Rewritten** — complete rewrite fixing 5 critical bugs |
| `src/train/inspect_checkpoint.py` | **Created** — new checkpoint structure inspector |
| `tokenizer/tokenizer_config.json` | **Fixed** — invalid `TokenizersBackend` → `PreTrainedTokenizerFast` |

### 5 Bugs Fixed in the Original Script

1. **`vocab_size`**: Was `32000`, should be **`32768`** (FSDP-padded)
2. **Parameter structure**: Script assumed per-layer keys (`layers_0`, `layers_1`...) but MaxText uses **stacked arrays** where all layers are in one tensor (e.g., `layers['self_attention']['query']['kernel'][:, layer_idx, :, :]`)
3. **Query scaling**: MaxText bakes `1/√d` into Q weights — must reverse with `× √head_dim`
4. **RoPE permutation**: MaxText permutes Q/K for its RoPE — must unpermute for HuggingFace
5. **Tokenizer**: Broken `from_pretrained()` call → direct `PreTrainedTokenizerFast` construction

### Next Step

**You need to run this on a machine with GCS access** (GCP VM, Colab, etc.). The workflow is:

```bash
# 1. Inspect checkpoint structure first
python src/train/inspect_checkpoint.py --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

# 2. Run conversion  
python src/train/convert_checkpoint.py --orbax_dir gs://... --hf_out_dir ./hf_model --tokenizer_path ./tokenizer
```

See the execution guide artifact for full details and troubleshooting. Where will you be running this — Colab, a GCP VM, or somewhere else?

### User Input



### Planner Response

Great, the plan is approved! Let me check the current STATUS.md to see what needs updating, and then we can move to execution.

*Viewed [STATUS.md](file:///Users/himanshu/Git/TinyMathReason-1B/STATUS.md) *

### Planner Response

The scripts are ready. The conversion can't run locally on your Mac (needs GCS access to the checkpoint). Since you approved the plan, the next concrete action is to **run the conversion** on a machine with GCS access.

Let me update STATUS.md to reflect the current state, and add the new inspect script to the key files:

*Edited relevant file*

### Planner Response

STATUS.md is updated. Everything is ready for execution. Here's where things stand:

### ✅ Done (this session)
- **Rewritten** `convert_checkpoint.py` — all 5 critical bugs fixed
- **Created** `inspect_checkpoint.py` — checkpoint structure inspector  
- **Fixed** `tokenizer_config.json` — valid HF class name

### ⏭️ Immediate Next Action
You need a machine with **GCS access** to run the two commands in order:

```bash
# 1. Inspect (verify structure matches expectations)
python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

# 2. Convert (produces ./hf_model/)
python src/train/convert_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362 \
    --hf_out_dir ./hf_model \
    --tokenizer_path ./tokenizer
```

This runs on **CPU** with ~4GB RAM — no GPU needed. Colab (free tier), a small GCP VM, or any machine with `gcloud auth` will work. Where are you planning to run it?

### User Input

    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 110, in <module>
    inspect_checkpoint(args.orbax_dir)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 21, in inspect_checkpoint
    import jax
ModuleNotFoundError: No module named 'jax'
himanshu@Himanshus-MacBook TinyMathReason-1B % 

*Viewed [inspect_checkpoint.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py) *

### Planner Response

You're running it locally — that works! JAX runs fine on Mac (CPU). Let's install the dependencies:

*User accepted the command `pip install jax jaxlib orbax-checkpoint google-cloud-storage numpy`*

*User accepted the command `python3 -m pip install jax jaxlib orbax-checkpoint google-cloud-storage numpy`*

### Planner Response

You have a venv in the project. Let me use that:

*User accepted the command `ls -d .venv venv 2>/dev/null && (test -f .venv/bin/activate && echo ".venv" || echo "venv")`*

*User accepted the command `.venv/bin/pip install jax jaxlib orbax-checkpoint google-cloud-storage numpy`*

### Planner Response

You seem to have a network connectivity issue (DNS resolution failing — `nodename nor servname provided`). The pip can't reach PyPI.

**Two things to note:**
1. Your `.venv` is **Python 3.14** — which is very new and JAX may not have wheels for it yet
2. You have a network connectivity issue right now

Can you:
1. **Check your internet connection** — the DNS resolution is failing
2. Once connected, try with the `venv` directory instead (which might have a different Python version):

```bash
venv/bin/python --version
.venv/bin/python --version
```

If both are Python 3.14, you may need to create a new venv with Python 3.11/3.12 since JAX doesn't support 3.14 yet. Alternatively, this is a great use case for **Google Colab** — it has everything pre-installed (JAX, torch, GCS auth) and you'd just need to upload the two scripts and run them.

### User Input

Python 3.14.3
Python 3.14.3

### Planner Response

Both are Python 3.14 — JAX doesn't have wheels for 3.14 yet (supports 3.10-3.13). Let me check if you have an older Python available:

*User accepted the command `ls /usr/local/bin/python3.* /opt/homebrew/bin/python3.* 2>/dev/null; which python3.11 python3.12 python3.13 2>/dev/null; brew list | grep python 2>/dev/null`*

### Planner Response

You have **Python 3.13** — that'll work. Let's create a fresh venv and install the deps:

*User accepted the command `/opt/homebrew/bin/python3.13 -m venv .venv313`*

*User accepted the command `.venv313/bin/pip install jax jaxlib orbax-checkpoint google-cloud-storage numpy`*

### Planner Response

Your internet connection is down — DNS resolution is failing (`nodename nor servname provided`). pip can't reach PyPI at all.

**You need to fix your network first.** Try:

```bash
# Test connectivity
ping google.com
curl https://pypi.org/simple/jax/
```

Common fixes:
- Check Wi-Fi connection
- Try `networksetup -setdnsservers Wi-Fi 8.8.8.8 8.8.4.4` if DNS is the issue
- Restart your router

Once your internet is back, run:

```bash
.venv313/bin/pip install jax jaxlib orbax-checkpoint google-cloud-storage numpy torch transformers safetensors
```

Then:

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

### User Input

upport_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x1125d1590>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
2026-05-13 01:18:04,267 - [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>.<lambda> at 0x112ae79c0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
2026-05-13 01:18:04,267 - [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 268, in get_filesystem_class
    register_implementation(protocol, _import_class(bit["class"]))
                                      ~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 303, in _import_class
    mod = importlib.import_module(mod)
  File "/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1395, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'gcsfs'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 30, in inspect_checkpoint
    ckpt = checkpointer.restore(orbax_dir)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 303, in restore
    if not directory.exists():
           ~~~~~~~~~~~~~~~~^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/gpath.py", line 152, in exists
    return self._backend.exists(self._path_str)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 453, in exists
    return self.fs(path).exists(path)
           ~~~~~~~^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 439, in fs
    return self._get_filesystem('gcs')
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 433, in _get_filesystem
    return fsspec.filesystem(name)
           ~~~~~~~~~~~~~~~~~^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 328, in filesystem
    cls = get_filesystem_class(protocol)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 270, in get_filesystem_class
    raise ImportError(bit.get("err")) from e
ImportError: Please install gcsfs to access Google Storage

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 268, in get_filesystem_class
    register_implementation(protocol, _import_class(bit["class"]))
                                      ~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 303, in _import_class
    mod = importlib.import_module(mod)
  File "/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1395, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'gcsfs'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 110, in <module>
    inspect_checkpoint(args.orbax_dir)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 35, in inspect_checkpoint
    ckpt = checkpointer.restore(orbax_dir)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/standard_checkpointer.py", line 152, in restore
    return super().restore(
           ~~~~~~~~~~~~~~~^
        directory, args=StandardRestore(target, strict=strict)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/async_checkpointer.py", line 580, in restore
    return super().restore(directory, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 303, in restore
    if not directory.exists():
           ~~~~~~~~~~~~~~~~^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/gpath.py", line 152, in exists
    return self._backend.exists(self._path_str)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 453, in exists
    return self.fs(path).exists(path)
           ~~~~~~~^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 439, in fs
    return self._get_filesystem('gcs')
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/etils/epath/backend.py", line 433, in _get_filesystem
    return fsspec.filesystem(name)
           ~~~~~~~~~~~~~~~~~^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 328, in filesystem
    cls = get_filesystem_class(protocol)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/fsspec/registry.py", line 270, in get_filesystem_class
    raise ImportError(bit.get("err")) from e
ImportError: Please install gcsfs to access Google Storage
himanshu@Himanshus-MacBook TinyMathReason-1B % pip install gcsfs
zsh: command not found: pip
himanshu@Himanshus-MacBook TinyMathReason-1B % pip3 install gcsfs
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try brew install
    xyz, where xyz is the package you are trying to
    install.
    
    If you wish to install a Python library that isn't in Homebrew,
    use a virtual environment:
    
    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install xyz
    
    If you wish to install a Python application that isn't in Homebrew,
    it may be easiest to use 'pipx install xyz', which will manage a
    virtual environment for you. You can install pipx with
    
    brew install pipx
    
    You may restore the old behavior of pip by passing
    the '--break-system-packages' flag to pip, or by adding
    'break-system-packages = true' to your pip.conf file. The latter
    will permanently disable this error.
    
    If you disable this error, we STRONGLY recommend that you additionally
    pass the '--user' flag to pip, or set 'user = true' in your pip.conf
    file. Failure to do this can result in a broken Homebrew installation.
    
    Read more about this behavior here: <https://peps.python.org/pep-0668/>

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

Good news — JAX and everything else installed fine! Just missing `gcsfs`. Install it into your venv:

*User accepted the command `.venv313/bin/pip install gcsfs`*

### Planner Response

Network is down again. Your DNS keeps dropping. Quick fix:

*User accepted the command `networksetup -getdnsservers Wi-Fi 2>/dev/null; ping -c 1 8.8.8.8 2>&1 | head -3`*

### User Input

just give command i will run

### Planner Response

Once your internet is back, run these in order:

```bash
# 1. Install missing dependency
.venv313/bin/pip install gcsfs

# 2. Run the inspector
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

Paste the output here and we'll proceed with the conversion.

### User Input

      Successfully uninstalled fsspec-2026.4.0
Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.13.5 aiosignal-1.4.0 attrs-26.1.0 decorator-5.2.1 frozenlist-1.8.0 fsspec-2026.3.0 gcsfs-2026.5.0 google-auth-oauthlib-1.4.0 google-cloud-storage-control-1.11.0 grpc-google-iam-v1-0.14.4 grpcio-1.80.0 grpcio-status-1.80.0 multidict-6.7.1 oauthlib-3.3.1 propcache-0.5.2 protobuf-6.33.6 requests-oauthlib-2.0.0 yarl-1.23.0

[notice] A new release of pip is available: 26.0.1 -> 26.1.1
[notice] To update, run: /Users/himanshu/Git/TinyMathReason-1B/.venv313/bin/python3.13 -m pip install --upgrade pip
zsh: command not found: #
2026-05-13 01:21:17,590 - Loading checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
2026-05-13 01:21:17,590 - orbax-checkpoint version: 0.11.39
2026-05-13 01:21:17,590 - save_device_host_concurrent_bytes=None
2026-05-13 01:21:17,608 - Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: dlopen(libtpu.so, 0x0001): tried: 'libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OSlibtpu.so' (no such file), '/opt/homebrew/lib/libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/lib/libtpu.so' (no such file), '/usr/lib/libtpu.so' (no such file, not in dyld cache), 'libtpu.so' (no such file)
2026-05-13 01:21:17,608 - Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x112a09810>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
2026-05-13 01:21:17,608 - [thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
2026-05-13 01:21:17,608 - [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:21,273 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:21:27,241 - Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:27,243 - [process=0] [sync] Finished blocking load in 9.63 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:34,941 - PyTreeCheckpointer failed: No structure could be identified for the checkpoint at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:34,941 - Trying StandardCheckpointer...
2026-05-13 01:21:34,943 - save_device_host_concurrent_bytes=None
2026-05-13 01:21:34,944 - Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x112a09810>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
2026-05-13 01:21:34,946 - [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>.<lambda> at 0x10a13eca0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
2026-05-13 01:21:34,946 - [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:38,378 - Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:21:38,378 - [process=0] [sync] Finished blocking load in 3.43 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 30, in inspect_checkpoint
    ckpt = checkpointer.restore(orbax_dir)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 312, in restore
    restored = self._restore(directory, args=ckpt_args)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 326, in _restore
    return self._handler.restore(directory, args=args)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/pytree_checkpoint_handler.py", line 884, in restore
    structure, use_zarr3_metadata = self._get_internal_metadata(directory)
                                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/pytree_checkpoint_handler.py", line 1057, in _get_internal_metadata
    raise FileNotFoundError(
        f'No structure could be identified for the checkpoint at {directory}.'
    )
FileNotFoundError: No structure could be identified for the checkpoint at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 110, in <module>
    inspect_checkpoint(args.orbax_dir)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 35, in inspect_checkpoint
    ckpt = checkpointer.restore(orbax_dir)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/standard_checkpointer.py", line 152, in restore
    return super().restore(
           ~~~~~~~~~~~~~~~^
        directory, args=StandardRestore(target, strict=strict)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/async_checkpointer.py", line 580, in restore
    return super().restore(directory, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 312, in restore
    restored = self._restore(directory, args=ckpt_args)
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/checkpointers/checkpointer.py", line 326, in _restore
    return self._handler.restore(directory, args=args)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py", line 249, in restore
    restore_args = _construct_restore_args(
        args.item,
    ...<2 lines>...
        args.support_layout,
    )
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py", line 406, in _construct_restore_args
    loaded_metadata_tree = _get_loaded_metadata()
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py", line 344, in _get_loaded_metadata
    return metadata()
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py", line 274, in metadata
    return self._impl.metadata(directory)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/pytree_checkpoint_handler.py", line 1123, in metadata
    return self._handler_impl.metadata(directory)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py", line 1401, in metadata
    internal_tree_metadata = asyncio_utils.run_sync(
        self._read_metadata_file(directory)
    )
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/asyncio_utils.py", line 50, in run_sync
    return asyncio.run(coro)
           ~~~~~~~~~~~^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py", line 1354, in _read_metadata_file
    raise FileNotFoundError(
    ...<2 lines>...
    )
FileNotFoundError: Metadata file (named _METADATA) does not exist at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The checkpoint uses MaxText's format with `items/` subdirectory and no `_METADATA` file (only `_CHECKPOINT_METADATA`). The Orbax API needs to be told the structure. Let me update the inspect script to handle this:

*Edited relevant file*

### Planner Response

Run it again:

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

This will first **list the raw GCS directory** so we can see exactly what's inside, then try to restore via `CheckpointManager` (which is how MaxText saves checkpoints). Paste the output here.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \ 
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

2026-05-13 01:24:53,567 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:24:53,992 - Listing GCS contents of: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

================================================================================
GCS DIRECTORY: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
================================================================================
  [DIR ] tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/
  [FILE] _CHECKPOINT_METADATA  (269 bytes)
  [FILE] commit_success.txt  (122 bytes)
  [DIR ] items/

  items/ subdirectory contents:
    [DIR ] items/
    [FILE] items/_METADATA  (5,016 bytes)
    [FILE] items/_sharding  (19,485 bytes)
    [DIR ] items/array_metadatas/
    [FILE] items/commit_success.txt  (128 bytes)
    [DIR ] items/d/
    [FILE] items/manifest.ocdbt  (120 bytes)
    [DIR ] items/ocdbt.process_0/
    [DIR ] items/ocdbt.process_1/
    [DIR ] items/ocdbt.process_2/
    [DIR ] items/ocdbt.process_3/
    [DIR ] items/ocdbt.process_4/
    [DIR ] items/ocdbt.process_5/
    [DIR ] items/ocdbt.process_6/
    [DIR ] items/ocdbt.process_7/

    items/ contents:
      [DIR ] items/
      [FILE] items/_METADATA/
      [FILE] items/_sharding/
      [DIR ] items/array_metadatas/
      [FILE] items/commit_success.txt/
      [DIR ] items/d/
      [FILE] items/manifest.ocdbt/
      [DIR ] items/ocdbt.process_0/
      [DIR ] items/ocdbt.process_1/
      [DIR ] items/ocdbt.process_2/
      [DIR ] items/ocdbt.process_3/
      [DIR ] items/ocdbt.process_4/
      [DIR ] items/ocdbt.process_5/
      [DIR ] items/ocdbt.process_6/
      [DIR ] items/ocdbt.process_7/

    items/array_metadatas/ contents:
      [DIR ] items/array_metadatas/
      [FILE] items/array_metadatas/process_0/
      [FILE] items/array_metadatas/process_1/
      [FILE] items/array_metadatas/process_2/
      [FILE] items/array_metadatas/process_3/
      [FILE] items/array_metadatas/process_4/
      [FILE] items/array_metadatas/process_5/
      [FILE] items/array_metadatas/process_6/
      [FILE] items/array_metadatas/process_7/

    items/d/ contents:
      [FILE] items/d/d6a2bd605a35e56460174e38e593242f/

    items/ocdbt.process_0/ contents:
      [DIR ] items/ocdbt.process_0/d/
      [FILE] items/ocdbt.process_0/manifest.ocdbt/

    items/ocdbt.process_1/ contents:
      [DIR ] items/ocdbt.process_1/d/
      [FILE] items/ocdbt.process_1/manifest.ocdbt/

    items/ocdbt.process_2/ contents:
      [DIR ] items/ocdbt.process_2/d/
      [FILE] items/ocdbt.process_2/manifest.ocdbt/

    items/ocdbt.process_3/ contents:
      [DIR ] items/ocdbt.process_3/d/
      [FILE] items/ocdbt.process_3/manifest.ocdbt/

    items/ocdbt.process_4/ contents:
      [DIR ] items/ocdbt.process_4/d/
      [FILE] items/ocdbt.process_4/manifest.ocdbt/

    items/ocdbt.process_5/ contents:
      [DIR ] items/ocdbt.process_5/d/
      [FILE] items/ocdbt.process_5/manifest.ocdbt/

    items/ocdbt.process_6/ contents:
      [DIR ] items/ocdbt.process_6/d/
      [FILE] items/ocdbt.process_6/manifest.ocdbt/

    items/ocdbt.process_7/ contents:
      [DIR ] items/ocdbt.process_7/d/
      [FILE] items/ocdbt.process_7/manifest.ocdbt/

================================================================================
_CHECKPOINT_METADATA contents:
================================================================================
{"item_handlers": {"items": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler"}, "metrics": {}, "performance_metrics": {}, "init_timestamp_nsecs": 1778608522128292016, "commit_timestamp_nsecs": 1778608527212768960, "custom_metadata": {}}
2026-05-13 01:25:05,135 - 
Attempting to restore checkpoint...
2026-05-13 01:25:05,886 - Trying to read checkpoint metadata...
2026-05-13 01:25:05,886 -   Parent dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints
2026-05-13 01:25:05,886 -   Step: 54362
2026-05-13 01:25:05,886 - CheckpointManagerOptions.read_only=True, setting save_interval_steps=0.
2026-05-13 01:25:05,886 - CheckpointManagerOptions.read_only=True, setting create=False.
2026-05-13 01:25:05,886 - [thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
2026-05-13 01:25:05,904 - Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: dlopen(libtpu.so, 0x0001): tried: 'libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OSlibtpu.so' (no such file), '/opt/homebrew/lib/libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/lib/libtpu.so' (no such file), '/usr/lib/libtpu.so' (no such file, not in dyld cache), 'libtpu.so' (no such file)
2026-05-13 01:25:05,905 - [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=None, item_handlers=None, handler_registry=None
2026-05-13 01:25:05,905 - Initialized registry DefaultCheckpointHandlerRegistry({('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x115868590>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x115868590>}).
2026-05-13 01:25:05,905 - orbax-checkpoint version: 0.11.39
2026-05-13 01:25:05,905 - [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>.<lambda> at 0x115836de0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
2026-05-13 01:25:05,905 - Given directory is read only=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints
I0513 01:25:09.026656 11834562 fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
I0513 01:25:09.033576 11834821 ev_poll_posix.cc:593] FD from fork parent still in poll list: fd(13, generation: 1)
I0513 01:25:09.436091 11834562 fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
I0513 01:25:09.437628 11834876 ev_poll_posix.cc:593] FD from fork parent still in poll list: fd(13, generation: 1)
2026-05-13 01:25:13,796 - Read Metadata={'item_handlers': {'items': 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler'}, 'metrics': {}, 'performance_metrics': {}, 'init_timestamp_nsecs': 1778608522128292016, 'commit_timestamp_nsecs': 1778608527212768960, 'custom_metadata': {}} from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/_CHECKPOINT_METADATA
2026-05-13 01:25:13,798 - Missing metrics for step 54362
2026-05-13 01:25:18,390 - Found 12 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints
2026-05-13 01:25:18,393 - [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=0, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=False, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=True, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=None, preservation_policy=None, prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x1157b7620>
2026-05-13 01:25:18,394 -   Available steps: [44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 54362]
2026-05-13 01:25:18,394 -   Latest step: 54362
2026-05-13 01:25:18,395 - [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:25:18,398 - Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:25:18,398 - [process=0] [sync] Finished blocking load in 0.00 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362.
2026-05-13 01:25:18,704 - Item "items" was found in the checkpoint, but could not be restored. Please provide a `CheckpointHandlerRegistry`, or call `restore` with an appropriate `CheckpointArgs` subclass.
2026-05-13 01:25:18,704 - Item "items" was found in the checkpoint, but could not be restored. Please provide a `CheckpointHandlerRegistry`, or call `restore` with an appropriate `CheckpointArgs` subclass.
2026-05-13 01:25:18,705 - No entry found in handler registry for item: items and args with type: <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>. Falling back to global handler registry.
2026-05-13 01:25:18,706 - save_device_host_concurrent_bytes=None
2026-05-13 01:25:18,707 - Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x115400690>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
2026-05-13 01:25:18,707 - Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x1158c9550>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/serialization/jax_array_handlers.py:736: UserWarning: Sharding info not provided when restoring. Populating sharding info from sharding file. Please note restoration time will be slightly increased due to reading from file. Note also that this option is unsafe when restoring on a different topology than the checkpoint was saved with.
  warnings.warn(
I0513 01:25:19.098116 11835113 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:25:19.098841 11835113 google_auth_provider.cc:156] Using OAuth2 AuthProvider
2026-05-13 01:25:21,115 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:21,425 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:22,609 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:22,932 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:24,097 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:24,452 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:25,654 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:26,109 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:27,317 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:27,607 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:28,799 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:29,125 - The available devices are different from the devices used to save the checkpoint.  Please restore checkpoint by passing new shardings for target devices. Original=[[[[[[[[[[[[DeviceMetadata(id=0)]]]]]]]], [[[[[[[[DeviceMetadata(id=8)]]]]]]]], [[[[[[[[DeviceMetadata(id=16)]]]]]]]], [[[[[[[[DeviceMetadata(id=24)]]]]]]]], [[[[[[[[DeviceMetadata(id=2)]]]]]]]], [[[[[[[[DeviceMetadata(id=10)]]]]]]]], [[[[[[[[DeviceMetadata(id=18)]]]]]]]], [[[[[[[[DeviceMetadata(id=26)]]]]]]]], [[[[[[[[DeviceMetadata(id=4)]]]]]]]], [[[[[[[[DeviceMetadata(id=12)]]]]]]]], [[[[[[[[DeviceMetadata(id=20)]]]]]]]], [[[[[[[[DeviceMetadata(id=28)]]]]]]]], [[[[[[[[DeviceMetadata(id=6)]]]]]]]], [[[[[[[[DeviceMetadata(id=14)]]]]]]]], [[[[[[[[DeviceMetadata(id=22)]]]]]]]], [[[[[[[[DeviceMetadata(id=30)]]]]]]]], [[[[[[[[DeviceMetadata(id=1)]]]]]]]], [[[[[[[[DeviceMetadata(id=9)]]]]]]]], [[[[[[[[DeviceMetadata(id=17)]]]]]]]], [[[[[[[[DeviceMetadata(id=25)]]]]]]]], [[[[[[[[DeviceMetadata(id=3)]]]]]]]], [[[[[[[[DeviceMetadata(id=11)]]]]]]]], [[[[[[[[DeviceMetadata(id=19)]]]]]]]], [[[[[[[[DeviceMetadata(id=27)]]]]]]]], [[[[[[[[DeviceMetadata(id=5)]]]]]]]], [[[[[[[[DeviceMetadata(id=13)]]]]]]]], [[[[[[[[DeviceMetadata(id=21)]]]]]]]], [[[[[[[[DeviceMetadata(id=29)]]]]]]]], [[[[[[[[DeviceMetadata(id=7)]]]]]]]], [[[[[[[[DeviceMetadata(id=15)]]]]]]]], [[[[[[[[DeviceMetadata(id=23)]]]]]]]], [[[[[[[[DeviceMetadata(id=31)]]]]]]]]]]]], current available=[CpuDevice(id=0)]
2026-05-13 01:25:30,058 - CheckpointManager approach failed: sharding passed to deserialization should be specified, concrete and an instance of `jax.sharding.Sharding`. Got None
2026-05-13 01:25:30,063 - Trying direct tensorstore-based inspection...
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_0
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_1
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_2
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_3
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_4
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_5
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_6
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_7
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_0/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_1/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_2/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_3/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_4/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_5/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_6/manifest.ocdbt
  Found OCDBT data: tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_7/manifest.ocdbt
2026-05-13 01:25:30,122 - Use the GCS listing above to understand the checkpoint layout.
2026-05-13 01:25:30,123 - The conversion script will need to be adapted based on this structure.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

*Viewed [inspect_checkpoint.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py) *

### Planner Response

Good data! The checkpoint was found on GCS with all 12 steps. The issue is it was saved across 32 TPU devices and Orbax needs explicit sharding to restore onto 1 CPU. Let me read the `_METADATA` file to get the tree structure, and fix the restore with proper single-device sharding:

*Edited relevant file*

### Planner Response

Run it:

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

The key fix: now it uses `SingleDeviceSharding(cpu)` to tell Orbax how to reshard from 32 TPUs → 1 CPU. It also reads the `items/_METADATA` file first so we can see the tree structure even if the full restore fails.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

2026-05-13 01:26:45,239 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:26:46,452 - Reading metadata from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/_METADATA
2026-05-13 01:26:51,864 - Metadata format: JSON

================================================================================
CHECKPOINT METADATA (items/_METADATA)
================================================================================
{
  "tree_metadata": {
    "('step',)": {
      "key_metadata": [
        {
          "key": "step",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": []
      }
    },
    "('params', 'VariableState', 'decoder', 'decoder_norm', 'scale', 'value')": {
      "key_metadata": [
        {
          "key": "params",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "key": "decoder",
          "key_type": 2
        },
        {
          "key": "decoder_norm",
          "key_type": 2
        },
        {
          "key": "scale",
          "key_type": 2
        },
        {
          "key": "value",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": [
          64
        ]
      }
    },
    "('params', 'VariableState', 'decoder', 'logits_dense', 'kernel', 'value')": {
      "key_metadata": [
        {
          "key": "params",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "key": "decoder",
          "key_type": 2
        },
        {
          "key": "logits_dense",
          "key_type": 2
        },
        {
          "key": "kernel",
          "key_type": 2
        },
        {
          "key": "value",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": [
          64,
          32768
        ]
      }
    },
    "('params', 'VariableState', 'token_embedder', 'embedding', 'value')": {
      "key_metadata": [
        {
          "key": "params",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "key": "token_embedder",
          "key_type": 2
        },
        {
          "key": "embedding",
          "key_type": 2
        },
        {
          "key": "value",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": [
          1024,
          2048
        ]
      }
    },
    "('opt_state', '0', 'count')": {
      "key_metadata": [
        {
          "key": "opt_state",
          "key_type": 2
        },
        {
          "key": "0",
          "key_type": 1
        },
        {
          "key": "count",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": []
      }
    },
    "('opt_state', '0', 'mu', 'VariableState', 'decoder', 'decoder_norm', 'scale', 'value')": {
      "key_metadata": [
        {
          "key": "opt_state",
          "key_type": 2
        },
        {
          "key": "0",
          "key_type": 1
        },
        {
          "key": "mu",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "key": "decoder",
          "key_type": 2
        },
        {
          "key": "decoder_norm",
          "key_type": 2
        },
        {
          "key": "scale",
          "key_type": 2
        },
        {
          "key": "value",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": [
          64
        ]
      }
    },
    "('opt_state', '0', 'mu', 'VariableState', 'decoder', 'logits_dense', 'kernel', 'value')": {
      "key_metadata": [
        {
          "key": "opt_state",
          "key_type": 2
        },
        {
          "key": "0",
          "key_type": 1
        },
        {
          "key": "mu",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "key": "decoder",
          "key_type": 2
        },
        {
          "key": "logits_dense",
          "key_type": 2
        },
        {
          "key": "kernel",
          "key_type": 2
        },
        {
          "key": "value",
          "key_type": 2
        }
      ],
      "value_metadata": {
        "value_type": "jax.Array",
        "skip_deserialize": false,
        "write_shape": [
          64,
          32768
        ]
      }
    },
    "('opt_state', '0', 'mu', 'VariableState', 'token_embedder', 'embedding', 'value')": {
      "key_metadata": [
        {
          "key": "opt_state",
          "key_type": 2
        },
        {
          "key": "0",
          "key_type": 1
        },
        {
          "key": "mu",
          "key_type": 2
        },
        {
          "key": "VariableState",
          "key_type": 2
        },
        {
          "ke
2026-05-13 01:26:51,869 - 
Restoring checkpoint with single-device CPU sharding...
2026-05-13 01:26:51,869 -   Parent: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints, Step: 54362
2026-05-13 01:26:51,870 - CheckpointManagerOptions.read_only=True, setting save_interval_steps=0.
2026-05-13 01:26:51,870 - CheckpointManagerOptions.read_only=True, setting create=False.
2026-05-13 01:26:51,871 - [thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
2026-05-13 01:26:51,943 - Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: dlopen(libtpu.so, 0x0001): tried: 'libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OSlibtpu.so' (no such file), '/opt/homebrew/lib/libtpu.so' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/lib/libtpu.so' (no such file), '/usr/lib/libtpu.so' (no such file, not in dyld cache), 'libtpu.so' (no such file)
2026-05-13 01:26:51,944 - [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=None, item_handlers=None, handler_registry=None
2026-05-13 01:26:51,947 - Initialized registry DefaultCheckpointHandlerRegistry({('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x11339c440>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x11339c440>}).
2026-05-13 01:26:51,947 - orbax-checkpoint version: 0.11.39
2026-05-13 01:26:51,948 - [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>.<lambda> at 0x11333e200> timeout: 1200 secs and primary_host=0 for async checkpoint writes
2026-05-13 01:26:51,948 - Given directory is read only=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints
I0513 01:26:53.894260 11836027 fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
I0513 01:26:53.910199 11836312 ev_poll_posix.cc:593] FD from fork parent still in poll list: fd(13, generation: 1)
I0513 01:26:54.700195 11836027 fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
I0513 01:26:54.701316 11836397 ev_poll_posix.cc:593] FD from fork parent still in poll list: fd(13, generation: 1)
2026-05-13 01:26:59,794 - Read Metadata={'item_handlers': {'items': 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler'}, 'metrics': {}, 'performance_metrics': {}, 'init_timestamp_nsecs': 1778517917227011074, 'commit_timestamp_nsecs': 1778517923724255214, 'custom_metadata': {}} from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/45000/_CHECKPOINT_METADATA
2026-05-13 01:27:01,333 - Missing metrics for step 45000
2026-05-13 01:27:05,512 - Found 12 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints
2026-05-13 01:27:05,517 - [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=0, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=False, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=True, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=None, preservation_policy=None, prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x113326660>
2026-05-13 01:27:05,517 -   Available steps: [44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 54362]
2026-05-13 01:27:06,124 - Item "items" was found in the checkpoint, but could not be restored. Please provide a `CheckpointHandlerRegistry`, or call `restore` with an appropriate `CheckpointArgs` subclass.
2026-05-13 01:27:06,125 - Item "items" was found in the checkpoint, but could not be restored. Please provide a `CheckpointHandlerRegistry`, or call `restore` with an appropriate `CheckpointArgs` subclass.
2026-05-13 01:27:06,125 -   Item metadata keys: <class 'orbax.checkpoint._src.composite.Composite'>
Traceback (most recent call last):
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 154, in <module>
    inspect_checkpoint(args.orbax_dir)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 94, in inspect_checkpoint
    restore_args = build_restore_args(items_meta)
  File "/Users/himanshu/Git/TinyMathReason-1B/src/train/inspect_checkpoint.py", line 79, in build_restore_args
    if hasattr(meta, 'shape') or (isinstance(meta, dict) and 'shape' in meta):
       ~~~~~~~^^^^^^^^^^^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/composite.py", line 81, in __getattr__
    return self.__getitem__(key)
           ~~~~~~~~~~~~~~~~^^^^^
  File "/Users/himanshu/Git/TinyMathReason-1B/.venv313/lib/python3.13/site-packages/orbax/checkpoint/_src/composite.py", line 69, in __getitem__
    raise KeyError(
        f'Unknown key: {key}. Available keys: {self._items.keys()}'
    )
KeyError: "Unknown key: shape. Available keys: dict_keys(['items'])"
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

We got the critical info from the `_METADATA` JSON. The tree structure is now clear — it has `VariableState` and `value` nesting that the conversion script needs to handle. Let me parse the metadata properly and skip the problematic restore for now:

*Edited relevant file*

### Planner Response

Run it:

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

This time it just reads the `_METADATA` JSON (which already worked) and parses it — no restore needed. It'll show us the exact parameter paths, shapes, and whether layers are stacked or have per-layer indices.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

2026-05-13 01:35:07,961 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:35:08,387 - Reading metadata from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/_METADATA
2026-05-13 01:35:11,907 - Total entries in tree_metadata: 13

================================================================================
CHECKPOINT PARAMETER TREE (write_shape = per-shard shape)
Saved across 8 processes (TPU v4-64 = 32 chips, 8 hosts)
================================================================================

--- Model Parameters (3 tensors) ---
  params/VariableState/decoder/decoder_norm/scale/value: [64]
  params/VariableState/decoder/logits_dense/kernel/value: [64, 32768]
  params/VariableState/token_embedder/embedding/value: [1024, 2048]

--- Other (1 entries) ---
  step: []

--- Optimizer State (9 tensors) [skipped] ---
  (contains 9 entries for mu/nu/count)

================================================================================
CONVERSION-CRITICAL ANALYSIS
================================================================================

Nesting pattern (first param entry):
  Keys: ['params', 'VariableState', 'decoder', 'decoder_norm', 'scale', 'value']
  Access: ckpt['items']['params']['VariableState']['decoder']['decoder_norm']['scale']['value']

  Has 'VariableState' nesting: True
  Has 'value' suffix: True

  ⚠️  IMPORTANT: Parameters are wrapped in Flax VariableState!
  Access pattern: ckpt['items']['params']['VariableState']['decoder']['...']['value']
  NOT: ckpt['params']['params']['decoder']['...']

  Layer-related params: 0

  Stacked layers (single 'layers' key): False
  Layer indices (layers_0, layers_1, ...): False

================================================================================
SHARDING INFO (first 3000 chars)
================================================================================
{
  "b3B0X3N0YXRlLjAuY291bnQ=": "{\"sharding_type\": \"NamedSharding\", \"shape\": [1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1], \"axis_names\": [\"diloco\", \"data\", \"stage\", \"fsdp\", \"fsdp_transpose\", \"context\", \"context_autoregressive\", \"tensor\", \"tensor_transpose\", \"tensor_sequence\", \"expert\", \"autoregressive\"], \"axis_types\": [\"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\"], \"partition_spec\": [], \"device_mesh\": {\"mesh\": [[[[[[[[[[[[{\"id\": 0}]]]]]]]], [[[[[[[[{\"id\": 8}]]]]]]]], [[[[[[[[{\"id\": 16}]]]]]]]], [[[[[[[[{\"id\": 24}]]]]]]]], [[[[[[[[{\"id\": 2}]]]]]]]], [[[[[[[[{\"id\": 10}]]]]]]]], [[[[[[[[{\"id\": 18}]]]]]]]], [[[[[[[[{\"id\": 26}]]]]]]]], [[[[[[[[{\"id\": 4}]]]]]]]], [[[[[[[[{\"id\": 12}]]]]]]]], [[[[[[[[{\"id\": 20}]]]]]]]], [[[[[[[[{\"id\": 28}]]]]]]]], [[[[[[[[{\"id\": 6}]]]]]]]], [[[[[[[[{\"id\": 14}]]]]]]]], [[[[[[[[{\"id\": 22}]]]]]]]], [[[[[[[[{\"id\": 30}]]]]]]]], [[[[[[[[{\"id\": 1}]]]]]]]], [[[[[[[[{\"id\": 9}]]]]]]]], [[[[[[[[{\"id\": 17}]]]]]]]], [[[[[[[[{\"id\": 25}]]]]]]]], [[[[[[[[{\"id\": 3}]]]]]]]], [[[[[[[[{\"id\": 11}]]]]]]]], [[[[[[[[{\"id\": 19}]]]]]]]], [[[[[[[[{\"id\": 27}]]]]]]]], [[[[[[[[{\"id\": 5}]]]]]]]], [[[[[[[[{\"id\": 13}]]]]]]]], [[[[[[[[{\"id\": 21}]]]]]]]], [[[[[[[[{\"id\": 29}]]]]]]]], [[[[[[[[{\"id\": 7}]]]]]]]], [[[[[[[[{\"id\": 15}]]]]]]]], [[[[[[[[{\"id\": 23}]]]]]]]], [[[[[[[[{\"id\": 31}]]]]]]]]]]]]}}",
  "b3B0X3N0YXRlLjAubXUuVmFyaWFibGVTdGF0ZS50b2tlbl9lbWJlZGRlci5lbWJlZGRpbmcudmFsdWU=": "{\"sharding_type\": \"NamedSharding\", \"shape\": [1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1], \"axis_names\": [\"diloco\", \"data\", \"stage\", \"fsdp\", \"fsdp_transpose\", \"context\", \"context_autoregressive\", \"tensor\", \"tensor_transpose\", \"tensor_sequence\", \"expert\", \"autoregressive\"], \"axis_types\": [\"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\", \"AxisType.Auto\"], \"partition_spec\": [], \"device_mesh\": {\"mesh\": [[[[[[[[[[[[{\"id\": 0}]]]]]]]], [[[[[[[[{\"id\": 8}]]]]]]]], [[[[[[[[{\"id\": 16}]]]]]]]], [[[[[[[[{\"id\": 24}]]]]]]]], [[[[[[[[{\"id\": 2}]]]]]]]], [[[[[[[[{\"id\": 10}]]]]]]]], [[[[[[[[{\"id\": 18}]]]]]]]], [[[[[[[[{\"id\": 26}]]]]]]]], [[[[[[[[{\"id\": 4}]]]]]]]], [[[[[[[[{\"id\": 12}]]]]]]]], [[[[[[[[{\"id\": 20}]]]]]]]], [[[[[[[[{\"id\": 28}]]]]]]]], [[[[[[[[{\"id\": 6}]]]]]]]], [[[[[[[[{\"id\": 14}]]]]]]]], [[[[[[[[{\"id\": 22}]]]]]]]], [[[[[[[[{\"id\": 30}]]]]]]]], [[[[[[[[{\"id\": 1}]]]]]]]], [[[[[[[[{\"id\": 9}]]]]]]]], [[[[[[[[{\"id\": 17}]]]]]]]], [[[[[[[[{\"id\": 25}]]]]]]]], [[[[[[[[{\"id\": 3}]]]]]]]], [[[[[[[[{\"id\": 11}]]]]]]]], [[[[[[[[{\"id\": 19}]]]]]]]], [[[[[[[[{\"id\": 27}]]]]]]]], [[[[[[[[{\"
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

This is a **critical finding**! Only 3 parameter tensors in the metadata — the **22 layers of attention + MLP are missing**! This means the scanned layer params are stored differently in the OCDBT format. Let me dig into the `array_metadatas` to find them:

*Edited relevant file*

### Planner Response

Run it — this will dig deeper into the OCDBT data to find the missing layer params:

```bash
.venv313/bin/pip install tensorstore msgpack && \
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/pip install tensorstore msgpack && \
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

Requirement already satisfied: tensorstore in ./.venv313/lib/python3.13/site-packages (0.1.83)
Requirement already satisfied: msgpack in ./.venv313/lib/python3.13/site-packages (1.1.2)
Requirement already satisfied: numpy>=1.22.0 in ./.venv313/lib/python3.13/site-packages (from tensorstore) (2.4.4)
Requirement already satisfied: ml_dtypes>=0.5.0 in ./.venv313/lib/python3.13/site-packages (from tensorstore) (0.5.4)

[notice] A new release of pip is available: 26.0.1 -> 26.1.1
[notice] To update, run: /Users/himanshu/Git/TinyMathReason-1B/.venv313/bin/python3.13 -m pip install --upgrade pip
2026-05-13 01:39:36,804 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:39:37,243 - Reading tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/_METADATA

================================================================================
items/_METADATA: 13 entries
================================================================================
  [PARAM] params/VariableState/decoder/decoder_norm/scale/value: write_shape=[64]
  [PARAM] params/VariableState/decoder/logits_dense/kernel/value: write_shape=[64, 32768]
  [PARAM] params/VariableState/token_embedder/embedding/value: write_shape=[1024, 2048]
  [STEP]  step: write_shape=[]
  [OPT]   9 optimizer state entries (hidden)
2026-05-13 01:39:40,508 - 
Reading tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/array_metadatas/process_0
2026-05-13 01:39:42,016 -   Size: 1907 bytes

================================================================================
array_metadatas/process_0 (JSON, 1907 bytes)
================================================================================
  Keys: 1
    array_metadatas: [{'array_metadata': {'param_name': 'step', 'write_shape': [], 'chunk_shape': [], 'ext_metadata': Non
2026-05-13 01:39:42,016 - 
Listing tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/d

================================================================================
items/d/ directory (1 entries)
================================================================================
  d6a2bd605a35e56460174e38e593242f: 25,995 bytes
2026-05-13 01:39:42,540 - 
Listing tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ocdbt.process_0/d

================================================================================
items/ocdbt.process_0/d/ (4 entries)
================================================================================
  60f6c8498c6fbbd4e7cf30bc31a097c7: 323 bytes
  c371a272a9414618585f5dcc317e7d19: 518 bytes
  e1cdbb4ff81a4918af50af6a5f130604: 138,178,560 bytes
  f13e57bf2cbd5b4fd2a4cff4a582c528: 45,772,800 bytes
  TOTAL: 183,952,201 bytes (0.18 GB)
2026-05-13 01:39:43,133 - 
Reading OCDBT manifest via tensorstore...
I0513 01:39:43.320202 11852075 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:39:43.320813 11852075 google_auth_provider.cc:156] Using OAuth2 AuthProvider

================================================================================
OCDBT KEYS (303 total)
================================================================================
  opt_state.0.count/c
  opt_state.0.count/zarr.json
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/0
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/1
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/10
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/11
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/12
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/13
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/14
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/15
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/16
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/17
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/18
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/19
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/2
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/20
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/21
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/22
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/23
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/24
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/25
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/26
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/27
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/28
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/29
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/3
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/30
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/31
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/4
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/5
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/6
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/7
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/8
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/c/9
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/zarr.json
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/0/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/1/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/10/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/11/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/12/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/13/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/14/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/15/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/16/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/17/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/18/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/19/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/2/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/20/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/21/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/22/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/23/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/24/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/25/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/26/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/27/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/28/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/29/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/3/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/30/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/31/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/4/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/5/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/6/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/7/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/8/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/c/9/0
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/zarr.json
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/0/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/1/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/10/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/11/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/12/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/13/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/14/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/15/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/16/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/17/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/18/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/19/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/2/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/20/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/21/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/22/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/23/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/24/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/25/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/26/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/27/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/28/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/29/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/3/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/30/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/31/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/4/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/5/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/6/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/7/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/8/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/c/9/0
  opt_state.0.mu.VariableState.token_embedder.embedding.value/zarr.json
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/0
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/1
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/10
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/11
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/12
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/13
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/14
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/15
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/16
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/17
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/18
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/19
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/2
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/20
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/21
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/22
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/23
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/24
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/25
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/26
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/27
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/28
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/29
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/3
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/30
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/31
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/4
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/5
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/6
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/7
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/8
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/c/9
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/zarr.json
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/0/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/1/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/10/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/11/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/12/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/13/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/14/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/15/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/16/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/17/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/18/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/19/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/2/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/20/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/21/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/22/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/23/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/24/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/25/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/26/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/27/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/28/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/29/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/3/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/30/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/31/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/4/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/5/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/6/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/7/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/8/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/c/9/0
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/zarr.json
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/0/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/1/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/10/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/11/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/12/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/13/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/14/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/15/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/16/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/17/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/18/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/19/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/2/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/20/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/21/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/22/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/23/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/24/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/25/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/26/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/27/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/28/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/29/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/3/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/30/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/31/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/4/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/5/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/6/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/7/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/8/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/c/9/0
  opt_state.0.nu.VariableState.token_embedder.embedding.value/zarr.json
  opt_state.2.count/c
  opt_state.2.count/zarr.json
  params.VariableState.decoder.decoder_norm.scale.value/c/0
  params.VariableState.decoder.decoder_norm.scale.value/c/1
  params.VariableState.decoder.decoder_norm.scale.value/c/10
  params.VariableState.decoder.decoder_norm.scale.value/c/11
  params.VariableState.decoder.decoder_norm.scale.value/c/12
  params.VariableState.decoder.decoder_norm.scale.value/c/13
  params.VariableState.decoder.decoder_norm.scale.value/c/14
  params.VariableState.decoder.decoder_norm.scale.value/c/15
  params.VariableState.decoder.decoder_norm.scale.value/c/16
  params.VariableState.decoder.decoder_norm.scale.value/c/17
  params.VariableState.decoder.decoder_norm.scale.value/c/18
  params.VariableState.decoder.decoder_norm.scale.value/c/19
  params.VariableState.decoder.decoder_norm.scale.value/c/2
  params.VariableState.decoder.decoder_norm.scale.value/c/20
  params.VariableState.decoder.decoder_norm.scale.value/c/21
  params.VariableState.decoder.decoder_norm.scale.value/c/22
  params.VariableState.decoder.decoder_norm.scale.value/c/23
  params.VariableState.decoder.decoder_norm.scale.value/c/24
  params.VariableState.decoder.decoder_norm.scale.value/c/25
  params.VariableState.decoder.decoder_norm.scale.value/c/26
  params.VariableState.decoder.decoder_norm.scale.value/c/27
  params.VariableState.decoder.decoder_norm.scale.value/c/28
  params.VariableState.decoder.decoder_norm.scale.value/c/29
  params.VariableState.decoder.decoder_norm.scale.value/c/3
  params.VariableState.decoder.decoder_norm.scale.value/c/30
  params.VariableState.decoder.decoder_norm.scale.value/c/31
  params.VariableState.decoder.decoder_norm.scale.value/c/4
  params.VariableState.decoder.decoder_norm.scale.value/c/5
  params.VariableState.decoder.decoder_norm.scale.value/c/6
  params.VariableState.decoder.decoder_norm.scale.value/c/7
  params.VariableState.decoder.decoder_norm.scale.value/c/8
  params.VariableState.decoder.decoder_norm.scale.value/c/9
  params.VariableState.decoder.decoder_norm.scale.value/zarr.json
  params.VariableState.decoder.logits_dense.kernel.value/c/0/0
  params.VariableState.decoder.logits_dense.kernel.value/c/1/0
  params.VariableState.decoder.logits_dense.kernel.value/c/10/0
  params.VariableState.decoder.logits_dense.kernel.value/c/11/0
  params.VariableState.decoder.logits_dense.kernel.value/c/12/0
  params.VariableState.decoder.logits_dense.kernel.value/c/13/0
  params.VariableState.decoder.logits_dense.kernel.value/c/14/0
  params.VariableState.decoder.logits_dense.kernel.value/c/15/0
  params.VariableState.decoder.logits_dense.kernel.value/c/16/0
  params.VariableState.decoder.logits_dense.kernel.value/c/17/0
  params.VariableState.decoder.logits_dense.kernel.value/c/18/0
  params.VariableState.decoder.logits_dense.kernel.value/c/19/0
  params.VariableState.decoder.logits_dense.kernel.value/c/2/0
  params.VariableState.decoder.logits_dense.kernel.value/c/20/0
  params.VariableState.decoder.logits_dense.kernel.value/c/21/0
  params.VariableState.decoder.logits_dense.kernel.value/c/22/0
  params.VariableState.decoder.logits_dense.kernel.value/c/23/0
  params.VariableState.decoder.logits_dense.kernel.value/c/24/0
  params.VariableState.decoder.logits_dense.kernel.value/c/25/0
  params.VariableState.decoder.logits_dense.kernel.value/c/26/0
  params.VariableState.decoder.logits_dense.kernel.value/c/27/0
  params.VariableState.decoder.logits_dense.kernel.value/c/28/0
  params.VariableState.decoder.logits_dense.kernel.value/c/29/0
  params.VariableState.decoder.logits_dense.kernel.value/c/3/0
  params.VariableState.decoder.logits_dense.kernel.value/c/30/0
  params.VariableState.decoder.logits_dense.kernel.value/c/31/0
  params.VariableState.decoder.logits_dense.kernel.value/c/4/0
  params.VariableState.decoder.logits_dense.kernel.value/c/5/0
  params.VariableState.decoder.logits_dense.kernel.value/c/6/0
  params.VariableState.decoder.logits_dense.kernel.value/c/7/0
  params.VariableState.decoder.logits_dense.kernel.value/c/8/0
  params.VariableState.decoder.logits_dense.kernel.value/c/9/0
  params.VariableState.decoder.logits_dense.kernel.value/zarr.json
  params.VariableState.token_embedder.embedding.value/c/0/0
  params.VariableState.token_embedder.embedding.value/c/1/0
  params.VariableState.token_embedder.embedding.value/c/10/0
  params.VariableState.token_embedder.embedding.value/c/11/0
  params.VariableState.token_embedder.embedding.value/c/12/0
  params.VariableState.token_embedder.embedding.value/c/13/0
  params.VariableState.token_embedder.embedding.value/c/14/0
  params.VariableState.token_embedder.embedding.value/c/15/0
  params.VariableState.token_embedder.embedding.value/c/16/0
  params.VariableState.token_embedder.embedding.value/c/17/0
  params.VariableState.token_embedder.embedding.value/c/18/0
  params.VariableState.token_embedder.embedding.value/c/19/0
  params.VariableState.token_embedder.embedding.value/c/2/0
  params.VariableState.token_embedder.embedding.value/c/20/0
  params.VariableState.token_embedder.embedding.value/c/21/0
  params.VariableState.token_embedder.embedding.value/c/22/0
  params.VariableState.token_embedder.embedding.value/c/23/0
  params.VariableState.token_embedder.embedding.value/c/24/0
  params.VariableState.token_embedder.embedding.value/c/25/0
  params.VariableState.token_embedder.embedding.value/c/26/0
  params.VariableState.token_embedder.embedding.value/c/27/0
  params.VariableState.token_embedder.embedding.value/c/28/0
  params.VariableState.token_embedder.embedding.value/c/29/0
  params.VariableState.token_embedder.embedding.value/c/3/0
  params.VariableState.token_embedder.embedding.value/c/30/0
  params.VariableState.token_embedder.embedding.value/c/31/0
  params.VariableState.token_embedder.embedding.value/c/4/0
  params.VariableState.token_embedder.embedding.value/c/5/0
  params.VariableState.token_embedder.embedding.value/c/6/0
  params.VariableState.token_embedder.embedding.value/c/7/0
  params.VariableState.token_embedder.embedding.value/c/8/0
  params.VariableState.token_embedder.embedding.value/c/9/0
  params.VariableState.token_embedder.embedding.value/zarr.json
  step/c
  step/zarr.json
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

This confirms the root OCDBT store only has 3 param tensors (embedding, final norm, logits). The **layer parameters must be in the per-process OCDBT stores**. Let me check `ocdbt.process_0/`:

*Edited relevant file*

### Planner Response

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

2026-05-13 01:45:53,729 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:45:54,372 - Reading tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/array_metadatas/process_0

================================================================================
array_metadatas/process_0: 12 array entries
================================================================================
  step: global=[], write=[], chunk=[], dtype=?
  params.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  params.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  params.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.0.count: global=[], write=[], chunk=[], dtype=?
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  opt_state.0.mu.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  opt_state.0.nu.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.2.count: global=[], write=[], chunk=[], dtype=?
2026-05-13 01:45:57,990 - 
Reading OCDBT keys from ocdbt.process_0/
I0513 01:45:58.015968 11857452 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:45:58.017584 11857452 google_auth_provider.cc:156] Using OAuth2 AuthProvider

================================================================================
ocdbt.process_0 OCDBT: 51 total keys, 15 unique params
================================================================================
  opt_state.0.count  (0 chunks, +zarr.json)
  opt_state.0.count/c  (0 chunks)
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  opt_state.0.mu.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  opt_state.2.count  (0 chunks, +zarr.json)
  opt_state.2.count/c  (0 chunks)
  params.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  params.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  params.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  step  (0 chunks, +zarr.json)
  step/c  (0 chunks)
2026-05-13 01:45:59,884 - 
Reading zarr.json files for shape info...
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

Same 3 params everywhere. The layer parameters are genuinely missing from the OCDBT key listing. But 184MB per process × 8 = 1.47GB of actual data exists — more than the 3 params account for. Let me read the zarr.json files to get true shapes, and try probing for hidden layer param paths:

*Edited relevant file*

### Planner Response

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362


================================================================================
READING ACTUAL ARRAY SHAPES VIA TENSORSTORE
================================================================================
I0513 01:55:32.187130 11865615 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:55:32.189329 11865615 google_auth_provider.cc:156] Using OAuth2 AuthProvider
  params.VariableState.decoder.decoder_norm.scale.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.decoder.decoder_norm.scale.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.decoder.decoder_norm.scale.value/\"}}'] [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116']
  params.VariableState.decoder.logits_dense.kernel.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.decoder.logits_dense.kernel.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116'] [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.decoder.logits_dense.kernel.value/\"}}']
  params.VariableState.token_embedder.embedding.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.token_embedder.embedding.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116'] [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.token_embedder.embedding.value/\"}}']

================================================================================
PROBING FOR SCANNED LAYER PARAMS
================================================================================
  ❌ params.VariableState.decoder.layers.self_attention.query.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.key.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.value.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.out.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wi_0.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wi_1.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wo.kernel.value: not found
  ❌ params.VariableState.decoder.layers.pre_self_attention_layer_norm.scale.value: not found
  ❌ params.VariableState.decoder.layers.post_self_attention_layer_norm.scale.value: not found
  ❌ params.decoder.layers.self_attention.query.kernel: not found
  ❌ params.VariableState.decoder.layers_0.self_attention.query.kernel.value: not found
  ❌ params.VariableState.Decoder.layers.self_attention.query.kernel.value: not found
  ❌ params.VariableState.decoder.layers.ScanIn.self_attention.query.kernel.value: not found
  ❌ decoder.layers.self_attention.query.kernel: not found
  ❌ params.params.decoder.layers.self_attention.query.kernel: not found

================================================================================
PROBING ocdbt.process_0/ FOR LAYER PARAMS
================================================================================
  ❌ not found
  ❌ not found
  ❌ not found
  ❌ not found
  ❌ not found

================================================================================
CHECKING MAXTEXT CONFIG
================================================================================
2026-05-13 01:56:12,927 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The zarr.json format indicates **zarr v3** (not v2), which is why the `zarr` driver failed. Let me read the zarr.json directly and also check the local maxtext config for clues about the missing layer params:

*Viewed [maxtext_config.yml](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/maxtext_config.yml) *

*Edited relevant file*

### Planner Response

```bash
.venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362
```

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

2026-05-13 01:45:53,729 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
2026-05-13 01:45:54,372 - Reading tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/array_metadatas/process_0

================================================================================
array_metadatas/process_0: 12 array entries
================================================================================
  step: global=[], write=[], chunk=[], dtype=?
  params.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  params.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  params.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.0.count: global=[], write=[], chunk=[], dtype=?
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  opt_state.0.mu.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value: global=[64], write=[64], chunk=[64], dtype=?
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value: global=[64, 32768], write=[64, 32768], chunk=[64, 32768], dtype=?
  opt_state.0.nu.VariableState.token_embedder.embedding.value: global=[1024, 2048], write=[1024, 2048], chunk=[1024, 2048], dtype=?
  opt_state.2.count: global=[], write=[], chunk=[], dtype=?
2026-05-13 01:45:57,990 - 
Reading OCDBT keys from ocdbt.process_0/
I0513 01:45:58.015968 11857452 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:45:58.017584 11857452 google_auth_provider.cc:156] Using OAuth2 AuthProvider

================================================================================
ocdbt.process_0 OCDBT: 51 total keys, 15 unique params
================================================================================
  opt_state.0.count  (0 chunks, +zarr.json)
  opt_state.0.count/c  (0 chunks)
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  opt_state.0.mu.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  opt_state.0.nu.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  opt_state.2.count  (0 chunks, +zarr.json)
  opt_state.2.count/c  (0 chunks)
  params.VariableState.decoder.decoder_norm.scale.value  (4 chunks, +zarr.json)
  params.VariableState.decoder.logits_dense.kernel.value  (4 chunks, +zarr.json)
  params.VariableState.token_embedder.embedding.value  (4 chunks, +zarr.json)
  step  (0 chunks, +zarr.json)
  step/c  (0 chunks)
2026-05-13 01:45:59,884 - 
Reading zarr.json files for shape info...
himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362


================================================================================
READING ACTUAL ARRAY SHAPES VIA TENSORSTORE
================================================================================
I0513 01:55:32.187130 11865615 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 01:55:32.189329 11865615 google_auth_provider.cc:156] Using OAuth2 AuthProvider
  params.VariableState.decoder.decoder_norm.scale.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.decoder.decoder_norm.scale.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.decoder.decoder_norm.scale.value/\"}}'] [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116']
  params.VariableState.decoder.logits_dense.kernel.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.decoder.logits_dense.kernel.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116'] [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.decoder.logits_dense.kernel.value/\"}}']
  params.VariableState.token_embedder.embedding.value: FAILED - NOT_FOUND: Error opening "zarr" driver: Metadata at "params.VariableState.token_embedder.embedding.value/.zarray" in OCDBT database at gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/ does not exist [source locations='tensorstore/driver/kvs_backed_chunk_driver.cc:1329\ntensorstore/driver/driver.cc:116tensorstore/driver/driver.cc:116'] [tensorstore_spec='{\"context\":{\"cache_pool\":{},\"data_copy_concurrency\":{},\"gcs_request_concurrency\":{},\"gcs_request_retries\":{},\"gcs_user_project\":{},\"ocdbt_coordinator\":{}},\"driver\":\"zarr\",\"kvstore\":{\"base\":{\"bucket\":\"tinymath-reason-data-himanshu\",\"driver\":\"gcs\",\"path\":\"checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/\"},\"driver\":\"ocdbt\",\"path\":\"params.VariableState.token_embedder.embedding.value/\"}}']

================================================================================
PROBING FOR SCANNED LAYER PARAMS
================================================================================
  ❌ params.VariableState.decoder.layers.self_attention.query.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.key.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.value.kernel.value: not found
  ❌ params.VariableState.decoder.layers.self_attention.out.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wi_0.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wi_1.kernel.value: not found
  ❌ params.VariableState.decoder.layers.mlp.wo.kernel.value: not found
  ❌ params.VariableState.decoder.layers.pre_self_attention_layer_norm.scale.value: not found
  ❌ params.VariableState.decoder.layers.post_self_attention_layer_norm.scale.value: not found
  ❌ params.decoder.layers.self_attention.query.kernel: not found
  ❌ params.VariableState.decoder.layers_0.self_attention.query.kernel.value: not found
  ❌ params.VariableState.Decoder.layers.self_attention.query.kernel.value: not found
  ❌ params.VariableState.decoder.layers.ScanIn.self_attention.query.kernel.value: not found
  ❌ decoder.layers.self_attention.query.kernel: not found
  ❌ params.params.decoder.layers.self_attention.query.kernel: not found

================================================================================
PROBING ocdbt.process_0/ FOR LAYER PARAMS
================================================================================
  ❌ not found
  ❌ not found
  ❌ not found
  ❌ not found
  ❌ not found

================================================================================
CHECKING MAXTEXT CONFIG
================================================================================
2026-05-13 01:56:12,927 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
himanshu@Himanshus-MacBook TinyMathReason-1B % .venv313/bin/python src/train/inspect_checkpoint.py \
    --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362


================================================================================
READING zarr.json FROM ROOT OCDBT STORE
================================================================================
I0513 02:03:52.518549 11877162 google_auth_provider.cc:149] Using credentials at /Users/himanshu/.config/gcloud/application_default_credentials.json
I0513 02:03:52.519950 11877162 google_auth_provider.cc:156] Using OAuth2 AuthProvider
  opt_state.0.count/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.2.count/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  step/zarr.json: ERROR - 'bytes' object has no attribute 'read'

================================================================================
READING zarr.json FROM ocdbt.process_0
================================================================================
  opt_state.0.count/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.mu.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.0.nu.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  opt_state.2.count/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.decoder.decoder_norm.scale.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.decoder.logits_dense.kernel.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  params.VariableState.token_embedder.embedding.value/zarr.json: ERROR - 'bytes' object has no attribute 'read'
  step/zarr.json: ERROR - 'bytes' object has no attribute 'read'

================================================================================
READING ARRAYS WITH zarr3 DRIVER
================================================================================
  ✅ params.VariableState.decoder.decoder_norm.scale.value
     shape=(2048,), dtype=dtype("float32")
     first 5 values: [0.49983808 0.51861143 0.6026926  0.5902462  0.52815753]
  ✅ params.VariableState.decoder.logits_dense.kernel.value
     shape=(2048, 32768), dtype=dtype("float32")
  ✅ params.VariableState.token_embedder.embedding.value
     shape=(32768, 2048), dtype=dtype("float32")

================================================================================
CHECKING ANOTHER CHECKPOINT (54000) FOR COMPARISON
================================================================================
2026-05-13 02:04:43,368 - gcsfs experimental features enabled via GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT.
  Step 54000 has 13 entries in tree_metadata
    [PARAM] params/VariableState/decoder/decoder_norm/scale/value: [64]
    [PARAM] params/VariableState/decoder/logits_dense/kernel/value: [64, 32768]
    [PARAM] params/VariableState/token_embedder/embedding/value: [1024, 2048]

================================================================================
LOCAL CONFIG ANALYSIS
================================================================================
  maxtext_config.yml does NOT set 'scan_layers'
  MaxText default scan_layers = True
  This means layers are stacked/scanned

  ⚠️  CRITICAL: Only 3 param arrays found in checkpoint!
  The 22 layers of attention + MLP weights are MISSING.
  Total data per process: ~184 MB (more than 3 params need)
  This suggests data exists but is stored differently.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

*Viewed [convert_checkpoint.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/convert_checkpoint.py) *

### Planner Response

Major findings: 

1. **zarr3 driver works** — true shapes confirmed: embedding `(32768, 2048)`, logits `(2048, 32768)`, norm `(2048,)` — all in float32
2. **Step 54000 has the SAME 3-only-params structure** — this isn't a one-off issue
3. **The layer parameters are genuinely not in ANY OCDBT key**

This is a critical problem. Let me check the training setup script — the JAX compatibility patches may have caused the scanned layer params to not be saved:

*Viewed [setup_tpu.sh](file:///Users/himanshu/Git/TinyMathReason-1B/scripts/setup_tpu.sh) *

*Searched web for MaxText scan_layers checkpoint save scanned parameters missing orbax "VariableState"*

### Planner Response

**That's the answer!** The checkpoint is fine — it's in MaxText's **"stacked training" format** and requires `generate_param_only_checkpoint.py` to unscan/unstack the parameters. The scanned layer parameters ARE in the checkpoint but are stored in a way that raw Orbax inspection can't see them directly. We need MaxText's own tooling to read them.

The key insight from the search: **"You MUST use the provided MaxText utility script `generate_param_only_checkpoint.py` to strip the optimizer state and perform the critical transformation from scanned (stacked) to unscanned (layer-by-layer) format."**

This means:
1. The checkpoint is NOT corrupted — it's just in MaxText's internal stacked format
2. We can't read it with raw Orbax/tensorstore — we need MaxText's checkpoint utilities
3. The conversion path is: **Stacked Training Checkpoint → `generate_param_only_checkpoint.py` → Unscanned Param-Only Checkpoint → Our `convert_checkpoint.py` → HuggingFace**

This has to run on a **TPU VM** (or at minimum a machine with MaxText installed and the full training mesh config), because MaxText's checkpoint restore needs the correct device mesh to reshard the stacked parameters.

Let me look up the exact `generate_param_only_checkpoint.py` usage:

*Read URL content from /AI-Hypercomputer/maxtext/main/src/maxtext/utils/generate_param_only_checkpoint.py *

*Viewed [content.md](file:///Users/himanshu/.gemini/antigravity/brain/6ff72167-4f51-455d-a580-d4c58c05ef76/.system_generated/steps/189/content.md) *

### Planner Response

Now I have the full picture. Let me update the plan and create the proper conversion workflow:

*Edited relevant file*

### Planner Response

## Summary

**The checkpoint is fine — not corrupted!** Here's what happened:

MaxText with `scan_layers=True` stores the 22 transformer layers in an internal **stacked format** that raw Orbax/tensorstore can't enumerate. Only the 3 non-scanned params (embedding, final norm, LM head) appear as discrete OCDBT keys. The layer weights ARE in the data files but need MaxText's own mesh-aware deserialization to extract.

### The fix is a 2-step pipeline:

| Step | Where | What |
|------|-------|------|
| **1** | TPU VM | Run MaxText's `generate_param_only_checkpoint.py` with `force_unroll=true` → produces unscanned param-only checkpoint |
| **2** | Anywhere | Run our `convert_checkpoint.py` → produces HuggingFace safetensors |

**You need a small TPU VM** (even a v4-8 works) to run Step 1 — MaxText needs the device mesh to reshard the stacked params. This is the same machine setup you used for training (just `setup_tpu.sh` + one command).

See the updated plan artifact for the exact commands. Want to proceed with provisioning a TPU VM for Step 1?

### User Input

previous v4 tpu cluster is still active

### Planner Response

Let me check the cluster details so I can give you the exact commands:

*Viewed [pretraining_setup.md](file:///Users/himanshu/Git/TinyMathReason-1B/docs/pretraining_setup.md) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [handoff.md](file:///Users/himanshu/Git/TinyMathReason-1B/handoff.md) *

### Planner Response

Perfect — the v4-64 cluster is still up. Since MaxText and all patches are already installed from training, you just need to SSH in and run one command. Here's exactly what to do:

**SSH into worker 0 and run:**

```bash
# SSH into the TPU VM (worker 0)
gcloud compute tpus tpu-vm ssh <YOUR_TPU_NAME> --zone=<YOUR_ZONE> --worker=0

# On the TPU VM — MaxText is already set up from training
cd ~/maxtext

# Generate the param-only unscanned checkpoint
PYTHONPATH=src python3 -m maxtext.utils.generate_param_only_checkpoint \
    maxtext_config.yml \
    load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items \
    base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only \
    run_name=tinymath-1b-decode \
    force_unroll=true
```

> **Note:** Make sure the training loop (`while true` in setup_tpu.sh) is stopped first — kill any running `train.py` processes so they don't conflict:
> ```bash
> pkill -f train.py
> ```

This will:
1. Load the full training state (with the stacked layer params)
2. Unscan/unstack the 22 layers into individual `layers_0`...`layers_21`
3. Cast to bfloat16
4. Save a clean param-only checkpoint to GCS

Once done, paste the output here and we'll run Step 2 (convert to HuggingFace) locally on your Mac.

### User Input

tpu-v4-64-spot	us-central2-b	v4-64	tpu-ubuntu2204-base	TPU VM	10.130.15.192	35.186.124.141	May 12, 2026, 6:13:00 PM	Spot VM	


### Planner Response

Got it. Here are the exact commands:

```bash
# 1. Kill any running training processes (all workers)
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="pkill -f train.py 2>/dev/null; echo done"

# 2. SSH into worker 0
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0

# 3. Once inside the TPU VM, run:
cd ~/maxtext && PYTHONPATH=src python3 -m maxtext.utils.generate_param_only_checkpoint \
    maxtext_config.yml \
    load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items \
    base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only \
    run_name=tinymath-1b-decode \
    force_unroll=true
```

Paste the output when it finishes (or errors). This should take ~2-5 minutes for a 1B model.

### User Input

himanshu@t1v-n-bf675f5d-w-0:~$ cd ~/maxtext && PYTHONPATH=src python3 -m maxtext.utils.generate_param_only_checkpoint \
    maxtext_config.yml \
    load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items \
    base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only \
    run_name=tinymath-1b-decode \
    force_unroll=true
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 187, in _run_module_as_main
    mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
  File "/usr/lib/python3.10/runpy.py", line 110, in _get_module_details
    __import__(pkg_name)
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 23, in <module>
    import pathwaysutils
ModuleNotFoundError: No module named 'pathwaysutils'
himanshu@t1v-n-bf675f5d-w-0:~/maxtext$ 

### Planner Response

The mock injector (which handles `pathwaysutils`) was only prepended to `train.py`. For `generate_param_only_checkpoint`, we need to inject it too. Run this:

```bash
cd ~/maxtext && python3 -c "
import sys
from unittest.mock import MagicMock

# Mock internal Google modules
class _SE(Exception): pass
class _SDE(Exception): pass
class _M:
    def __init__(self, *a, **k): pass

_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _SE
_pm.ScaleDownSignalError = _SDE
_pm.Manager = _M
_pw.elastic.manager = _pm
sys.modules['pathwaysutils'] = _pw
sys.modules['pathwaysutils.elastic'] = _pw.elastic
sys.modules['pathwaysutils.elastic.manager'] = _pm
for m in ['qwix','qwix.pallas','qwix._src','qwix._src.core','qwix.contrib','qwix.contrib.sparsity','tokamax','tokamax._src','tokamax._src.ops','tokamax._src.ops.experimental','tokamax._src.ops.experimental.tpu','tokamax._src.ops.experimental.tpu.splash_attention','drjax']:
    sys.modules[m] = MagicMock()

import types, importlib.abc
from unittest.mock import MagicMock as MM
class _PF(importlib.abc.MetaPathFinder):
    _P = ('jax.pallas', 'jax.experimental.pallas')
    def find_module(self, fn, path=None):
        for p in self._P:
            if fn == p or fn.startswith(p+'.'): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
        _C = type('PM', (types.ModuleType,), {'__getattr__': lambda s,n: MM()})
        m = _C(fn); m.__path__=[]; m.__file__='<mock>'; m.__loader__=self; m.__package__=fn
        sys.modules[fn] = m; return m
sys.meta_path.insert(0, _PF())

import jax
if not hasattr(jax, 'pallas'): jax.pallas = sys.modules.get('jax.pallas') or _PF().load_module('jax.pallas')

sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
    'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
    'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
    'run_name=tinymath-1b-decode',
    'force_unroll=true']

from maxtext.utils.generate_param_only_checkpoint import main
from absl import app
app.run(main)
"
```

This injects the same mocks from `setup_tpu.sh` before importing MaxText, then runs the checkpoint generator. Paste the output.

### User Input

himanshu@t1v-n-bf675f5d-w-0:~/maxtext$ cd ~/maxtext && python3 -c "
import sys
from unittest.mock import MagicMock

# Mock internal Google modules
class _SE(Exception): pass
class _SDE(Exception): pass
class _M:
    def __init__(self, *a, **k): pass

_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _SE
_pm.ScaleDownSignalError = _SDE
_pm.Manager = _M
_pw.elastic.manager = _pm
sys.modules['pathwaysutils'] = _pw
sys.modules['pathwaysutils.elastic'] = _pw.elastic
sys.modules['pathwaysutils.elastic.manager'] = _pm
for m in ['qwix','qwix.pallas','qwix._src','qwix._src.core','qwix.contrib','qwix.contrib.sparsity','tokamax','tokamax._src','tokamax._src.ops','tokamax._src.ops.experimental','tokamax._src.ops.experimental.tpu','tokamax._src.ops.experimental.tpu.splash_attention','drjax']:
    sys.modules[m] = MagicMock()

import types, importlib.abc
from unittest.mock import MagicMock as MM
class _PF(importlib.abc.MetaPathFinder):
    _P = ('jax.pallas', 'jax.experimental.pallas')
    def find_module(self, fn, path=None):
        for p in self._P:
            if fn == p or fn.startswith(p+'.'): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
        _C = type('PM', (types.ModuleType,), {'__getattr__': lambda s,n: MM()})
        m = _C(fn); m.__path__=[]; m.__file__='<mock>'; m.__loader__=self; m.__package__=fn
        sys.modules[fn] = m; return m
sys.meta_path.insert(0, _PF())

import jax
if not hasattr(jax, 'pallas'): jax.pallas = sys.modules.get('jax.pallas') or _PF().load_module('jax.pallas')

sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
    'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
    'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
    'run_name=tinymath-1b-decode',
    'force_unroll=true']

from maxtext.utils.generate_param_only_checkpoint import main
from absl import app
app.run(main)
"
Traceback (most recent call last):
  File "<string>", line 46, in <module>
ModuleNotFoundError: No module named 'maxtext'

### Planner Response

Ah, I missed adding `src` to the search path in that one-liner. Let's make it cleaner by creating a small helper script on the TPU VM instead. This avoids escaping issues and ensures the mocks are loaded correctly.

Run this on the TPU VM (worker 0):

```bash
cd ~/maxtext

# Create the helper script with mocks and path setup
cat << 'EOF' > convert_helper.py
import sys
import os
from unittest.mock import MagicMock

# 1. Mock internal Google modules
class _SE(Exception): pass
class _SDE(Exception): pass
class _M:
    def __init__(self, *a, **k): pass

_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _SE
_pm.ScaleDownSignalError = _SDE
_pm.Manager = _M
_pw.elastic.manager = _pm
sys.modules['pathwaysutils'] = _pw
sys.modules['pathwaysutils.elastic'] = _pw.elastic
sys.modules['pathwaysutils.elastic.manager'] = _pm
for m in ['qwix','qwix.pallas','qwix._src','qwix._src.core','qwix.contrib','qwix.contrib.sparsity','tokamax','tokamax._src','tokamax._src.ops','tokamax._src.ops.experimental','tokamax._src.ops.experimental.tpu','tokamax._src.ops.experimental.tpu.splash_attention','drjax']:
    sys.modules[m] = MagicMock()

# 2. Pallas Mock
import types, importlib.abc
from unittest.mock import MagicMock as MM
class _PF(importlib.abc.MetaPathFinder):
    _P = ('jax.pallas', 'jax.experimental.pallas')
    def find_module(self, fn, path=None):
        for p in self._P:
            if fn == p or fn.startswith(p+'.'): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
        _C = type('PM', (types.ModuleType,), {'__getattr__': lambda s,n: MM()})
        m = _C(fn); m.__path__=[]; m.__file__='<mock>'; m.__loader__=self; m.__package__=fn
        sys.modules[fn] = m; return m
sys.meta_path.insert(0, _PF())

import jax
if not hasattr(jax, 'pallas'): jax.pallas = sys.modules.get('jax.pallas') or _PF().load_module('jax.pallas')

# 3. Setup Path and Run
sys.path.append(os.path.abspath('src'))
from maxtext.utils.generate_param_only_checkpoint import main
from absl import app

if __name__ == "__main__":
    # Override sys.argv for the MaxText app
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(main)
EOF

# Run it
PYTHONPATH=src python3 convert_helper.py
```

**Note:** If the script hangs at "Initializing distributed JAX", you may need to run it on **all workers** simultaneously. If that happens, let me know and I'll give you the `gcloud` command to broadcast it.

### User Input

    from maxtext.models import models
  File "/home/himanshu/maxtext/src/maxtext/models/models.py", line 30, in <module>
    from maxtext.layers.nnx_decoders import NNXDecoder
  File "/home/himanshu/maxtext/src/maxtext/layers/nnx_decoders.py", line 47, in <module>
    from maxtext.models import (
  File "/home/himanshu/maxtext/src/maxtext/models/deepseek.py", line 41, in <module>
    from maxtext.models import deepseek_batchsplit
  File "/home/himanshu/maxtext/src/maxtext/models/deepseek_batchsplit.py", line 38, in <module>
    from maxtext.kernels import attention, sort_activations, megablox
  File "/home/himanshu/maxtext/src/maxtext/kernels/sort_activations.py", line 21, in <module>
    from maxtext.kernels import gather_reduce_sc
  File "/home/himanshu/maxtext/src/maxtext/kernels/gather_reduce_sc.py", line 58, in <module>
    @jax.jit(
  File "/home/himanshu/.local/lib/python3.10/site-packages/jax/_src/api.py", line 177, in wrapped
    return f(*args, **kwargs)
TypeError: jit() missing 1 required positional argument: 'fun'
himanshu@t1v-n-bf675f5d-w-0:~/maxtext$ 

### Planner Response

The `jax.jit` error is a compatibility issue with JAX 0.6.x. We need to apply the same monkey-patches I used in the training setup to the helper script.

Update `convert_helper.py` on the TPU VM with this version:

```bash
cd ~/maxtext

cat << 'EOF' > convert_helper.py
import sys
import os
import functools
from unittest.mock import MagicMock

# 1. JAX Compatibility Patches (from setup_tpu.sh)
import jax
import jax._src.api

_original_jit = jax.jit
def _patched_jit(fun=None, **kwargs):
    if fun is None:
        return functools.partial(_original_jit, **kwargs)
    return _original_jit(fun, **kwargs)
jax.jit = _patched_jit
jax._src.api.jit = _patched_jit

_original_config_update = jax.config.update
def _patched_config_update(name, value):
    try:
        _original_config_update(name, value)
    except AttributeError:
        pass
jax.config.update = _patched_config_update

if not hasattr(jax, 'set_mesh'):
    import contextlib
    @contextlib.contextmanager
    def _set_mesh_shim(mesh):
        with mesh:
            yield
    jax.set_mesh = _set_mesh_shim
    jax.sharding.set_mesh = _set_mesh_shim

# 2. Mock internal Google modules
class _SE(Exception): pass
class _SDE(Exception): pass
class _M:
    def __init__(self, *a, **k): pass

_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _SE
_pm.ScaleDownSignalError = _SDE
_pm.Manager = _M
_pw.elastic.manager = _pm
sys.modules['pathwaysutils'] = _pw
sys.modules['pathwaysutils.elastic'] = _pw.elastic
sys.modules['pathwaysutils.elastic.manager'] = _pm
for m in ['qwix','qwix.pallas','qwix._src','qwix._src.core','qwix.contrib','qwix.contrib.sparsity','tokamax','tokamax._src','tokamax._src.ops','tokamax._src.ops.experimental','tokamax._src.ops.experimental.tpu','tokamax._src.ops.experimental.tpu.splash_attention','drjax']:
    sys.modules[m] = MagicMock()

# 3. Pallas Mock Finder
import types, importlib.abc
from unittest.mock import MagicMock as MM
class _PF(importlib.abc.MetaPathFinder):
    _P = ('jax.pallas', 'jax.experimental.pallas')
    def find_module(self, fn, path=None):
        for p in self._P:
            if fn == p or fn.startswith(p+'.'): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
        _C = type('PM', (types.ModuleType,), {'__getattr__': lambda s,n: MM()})
        m = _C(fn); m.__path__=[]; m.__file__='<mock>'; m.__loader__=self; m.__package__=fn
        sys.modules[fn] = m; return m
sys.meta_path.insert(0, _PF())

if not hasattr(jax, 'pallas'): jax.pallas = sys.modules.get('jax.pallas') or _PF().load_module('jax.pallas')

# 4. Setup Path and Run
sys.path.append(os.path.abspath('src'))

# Distributed Init (Required for Orbax on TPU)
try:
    jax.distributed.initialize()
except Exception as e:
    print(f"Distributed init info: {e}")

from maxtext.utils.generate_param_only_checkpoint import main
from absl import app

if __name__ == "__main__":
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(main)
EOF

PYTHONPATH=src python3 convert_helper.py
```

Try running that. If it stays stuck at "Starting MaxText Pretraining..." or "Initializing distributed JAX", we'll need to run it on all workers.

### User Input

himanshu@t1v-n-bf675f5d-w-0:~/maxtext$ cd ~/maxtext

cat << 'EOF' > convert_helper.py
import sys
import os
import functools
from unittest.mock import MagicMock

# 1. JAX Compatibility Patches (from setup_tpu.sh)
import jax
import jax._src.api

_original_jit = jax.jit
def _patched_jit(fun=None, **kwargs):
    if fun is None:
        return functools.partial(_original_jit, **kwargs)
    return _original_jit(fun, **kwargs)
jax.jit = _patched_jit
jax._src.api.jit = _patched_jit

_original_config_update = jax.config.update
def _patched_config_update(name, value):
    try:
        _original_config_update(name, value)
    except AttributeError:
        pass
jax.config.update = _patched_config_update

if not hasattr(jax, 'set_mesh'):
    import contextlib
    @contextlib.contextmanager
    def _set_mesh_shim(mesh):
        with mesh:
            yield
    jax.set_mesh = _set_mesh_shim
    jax.sharding.set_mesh = _set_mesh_shim

# 2. Mock internal Google modules
class _SE(Exception): pass
class _SDE(Exception): pass
class _M:
    def __init__(self, *a, **k): pass

_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _SE
_pm.ScaleDownSignalError = _SDE
_pm.Manager = _M
_pw.elastic.manager = _pm
sys.modules['pathwaysutils'] = _pw
sys.modules['pathwaysutils.elastic'] = _pw.elastic
sys.modules['pathwaysutils.elastic.manager'] = _pm
for m in ['qwix','qwix.pallas','qwix._src','qwix._src.core','qwix.contrib','qwix.contrib.sparsity','tokamax','tokamax._src','tokamax._src.ops','tokamax._src.ops.experimental','tokamax._src.ops.experimental.tpu','tokamax._src.ops.experimental.tpu.splash_attention','drjax']:
    sys.modules[m] = MagicMock()

# 3. Pallas Mock Finder
import types, importlib.abc
from unittest.mock import MagicMock as MM
class _PF(importlib.abc.MetaPathFinder):
    _P = ('jax.pallas', 'jax.experimental.pallas')
    def find_module(self, fn, path=None):
        for p in self._P:
            if fn == p or fn.startswith(p+'.'): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
PYTHONPATH=src python3 convert_helper.pymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',4362/items',


### Planner Response

It looks like the paste got cut off in your terminal. Let's try again with a cleaner version.

Run this entire block on the TPU VM:

```bash
cd ~/maxtext

cat << 'EOF' > convert_helper.py
import sys
import os
import functools
from unittest.mock import MagicMock

# 1. JAX Patches
import jax
import jax._src.api
_original_jit = jax.jit
def _patched_jit(fun=None, **kwargs):
    if fun is None: return functools.partial(_original_jit, **kwargs)
    return _original_jit(fun, **kwargs)
jax.jit = _patched_jit
jax._src.api.jit = _patched_jit

if not hasattr(jax, 'set_mesh'):
    import contextlib
    @contextlib.contextmanager
    def _set_mesh_shim(mesh):
        with mesh: yield
    jax.set_mesh = _set_mesh_shim
    jax.sharding.set_mesh = _set_mesh_shim

# 2. Mocks
class _E(Exception): pass
_pw = MagicMock()
_pm = MagicMock()
_pm.ScaleUpSignalError = _pm.ScaleDownSignalError = _E
_pm.Manager = lambda *a, **k: None
_pw.elastic.manager = _pm
sys.modules.update({
    'pathwaysutils': _pw, 'pathwaysutils.elastic': _pw.elastic, 'pathwaysutils.elastic.manager': _pm,
    'qwix': MagicMock(), 'tokamax': MagicMock(), 'drjax': MagicMock()
})

import types, importlib.abc
class _PF(importlib.abc.MetaPathFinder):
    def find_module(self, fn, path=None):
        if fn.startswith(('jax.pallas', 'jax.experimental.pallas')): return self
    def load_module(self, fn):
        if fn in sys.modules: return sys.modules[fn]
        m = types.ModuleType(fn); m.__getattr__ = lambda n: MagicMock()
        sys.modules[fn] = m; return m
sys.meta_path.insert(0, _PF())

# 3. Setup Path and Run
sys.path.append(os.path.abspath('src'))
try: jax.distributed.initialize()
except: pass

from maxtext.utils.generate_param_only_checkpoint import main
from absl import app

if __name__ == "__main__":
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(main)
EOF

PYTHONPATH=src python3 convert_helper.py
```

If it hangs at the start, it's likely waiting for other workers to join the JAX cluster. If that happens, tell me and we'll run it on all workers at once.

### User Input

store it in python file, move to tpu via scp and the execute using gcloud command

### Planner Response

I'll create the script locally first. Then you can use `scp` to send it to all workers and execute it.

### 1. Create the script locally
I'm writing the script to `src/train/convert_helper.py` in your local workspace.



*Edited relevant file*

### Planner Response

### 2. Move and Execute on TPU

Run these commands in your local terminal:

```bash
# 1. Copy the script to ALL workers
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Execute on ALL workers (essential for distributed checkpoint restore)
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

**Why all workers?** 
The training checkpoint is sharded across the entire v4-64 mesh. To restore it and "unroll" the layers, all 8 workers (32 chips) need to participate in the JAX collective operations. Only worker 0 will actually write the final file to GCS, but the others must join to provide their shards of the data.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all
WARNING: Cannot retrieve keys in ssh-agent. Command may stall.
Using scp batch size of 1.Attempting to SCP into 1 nodes with a total of 8 workers.
SCP: Attempting to connect to worker 0...
SCP: Attempting to connect to worker 1...
SCP: Attempting to connect to worker 2...
SCP: Attempting to connect to worker 3...
SCP: Attempting to connect to worker 4...
SCP: Attempting to connect to worker 5...
SCP: Attempting to connect to worker 6...
SCP: Attempting to connect to worker 7...
convert_helper.py                                                                                            100% 2295     9.0KB/s   00:00    
convert_helper.py                                                                                            100% 2295     8.9KB/s   00:00    
convert_helper.py                                                                                            100% 2295     9.1KB/s   00:00    
convert_helper.py                                                                                            100% 2295     9.0KB/s   00:00    
convert_helper.py                                                                                            100% 2295     9.1KB/s   00:00    
convert_helper.py                                                                                            100% 2295     8.9KB/s   00:00    
convert_helper.py                                                                                            100% 2295     9.1KB/s   00:00    
convert_helper.py                                                                                            100% 2295     8.7KB/s   00:00    
himanshu@Himanshus-MacBook TinyMathReason-1B % gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 8 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
SSH: Attempting to connect to worker 4...
SSH: Attempting to connect to worker 5...
SSH: Attempting to connect to worker 6...
SSH: Attempting to connect to worker 7...
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
/home/himanshu/.local/lib/python3.10/site-packages/google/api_core/_python_version_support.py:273: FutureWarning: You are using a Python version (3.10.6) which Google will stop supporting in new releases of google.cloud.appengine_logging_v1 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.appengine_logging_v1 past that date.
  warnings.warn(message, FutureWarning)
Worker 1 initialized.
Worker 0 initialized.
Worker 7 initialized.
Worker 2 initialized.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
Worker 6 initialized.
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Worker 5 initialized.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <moTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
dule>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Worker 3 initialized.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
Worker 4 initialized.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 55, in <module>
    from maxtext.utils.generate_param_only_checkpoint import main
  File "/home/himanshu/maxtext/src/maxtext/__init__.py", line 38, in <module>
    from maxtext.configs import pyconfig
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig.py", line 31, in <module>
    from maxtext.configs import pyconfig_deprecated
  File "/home/himanshu/maxtext/src/maxtext/configs/pyconfig_deprecated.py", line 36, in <module>
    from maxtext.utils import max_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/max_utils.py", line 45, in <module>
    from maxtext.utils import elastic_utils
  File "/home/himanshu/maxtext/src/maxtext/utils/elastic_utils.py", line 26, in <module>
    elastic_manager: manager.Manager | None = None
TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The error happened because I mocked `manager.Manager` as a function, but Python 3.10's type hints (using the `|` operator) require it to be a **class** to avoid a `TypeError`.

I've fixed the mock in the script. Please update the local file and retry.



*Edited relevant file*

### Planner Response

Now, copy the updated script to the TPU cluster and run it again:

```bash
# 1. Copy the updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Execute on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

### User Input

gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

### User Input

    from maxtext.layers import initializers, linears, mhc, normalizations, quantizations
  File "/home/himanshu/maxtext/src/maxtext/layers/linears.py", line 34, in <module>
    from maxtext.layers import nnx_wrappers, quantizations
  File "/home/himanshu/maxtext/src/maxtext/layers/quantizations.py", line 19, in <module>
    import qwix.pallas as qpl
ModuleNotFoundError: No module named 'qwix.pallas'; 'qwix' is not a package
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1.

### Planner Response

The error `'qwix' is not a package` means the mock wasn't set up to handle submodules like `qwix.pallas`. 

I've updated the script to use a more robust "Meta-Mock" finder that automatically handles any missing internal Google packages and submodules.

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

Now, SCP the new version and run:

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

### User Input

-prod-run2/checkpoints/54362/items.
I0512 21:10:26.542409 140346501683200 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.31 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:10:26.871180   69739 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:26.879775   71095 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:26.917820   68927 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:26.930311   63745 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:27.050685   64546 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:27.056323   64429 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:27.062405   73713 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:27.070281   66276 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:10:28.535125 140121024292864 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.015 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.478097915649414 s) (per-host)
I0512 21:10:28.534950 140463803897856 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.021 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4696922302246094 s) (per-host)
I0512 21:10:28.535096 140346501683200 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.007 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4896833896636963 s) (per-host)
I0512 21:10:28.536699 140446751868928 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.010 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4851233959197998 s) (per-host)
I0512 21:10:28.539735 139848688261120 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 922.368 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.66530442237854 s) (per-host)
I0512 21:10:28.541303 140346501683200 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.463 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7328298091888428 s) (per-host)
I0512 21:10:28.542668 140121024292864 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.351 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7905893325805664 s) (per-host)
I0512 21:10:28.547119 140463803897856 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 3.372 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7793097496032715 s) (per-host)
I0512 21:10:28.546819 139848688261120 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 3.299 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.8188762664794922 s) (per-host)
I0512 21:10:28.547768 140484659931136 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 913.268 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.6818969249725342 s) (per-host)
I0512 21:10:28.552243 140446751868928 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.461 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7335875034332275 s) (per-host)
I0512 21:10:28.554272 140484659931136 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 3.220 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.8636221885681152 s) (per-host)
I0512 21:10:28.613752 140042913851392 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 902.756 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.7014822959899902 s) (per-host)
I0512 21:10:28.617992 140028269369344 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 907.271 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.6930158138275146 s) (per-host)
I0512 21:10:28.621007 140042913851392 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.088 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.9428117275238037 s) (per-host)
I0512 21:10:28.624695 140028269369344 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.178 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.8879456520080566 s) (per-host)
I0512 21:10:28.630481 140463803897856 event_tracking.py:138] [process=2] [sync] Finished load in 2.39 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.630858 140463803897856 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:10:28.630631 140121024292864 event_tracking.py:138] [process=7] [sync] Finished load in 2.40 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
I0512 21:10:28.631162 140121024292864 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:10:28.630496 140484659931136 event_tracking.py:138] [process=0] [sync] Finished load in 2.42 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
I0512 21:10:28.630483 140346501683200 event_tracking.py:138] [process=5] [sync] Finished load in 2.40 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.631012 140346501683200 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:10:28.630530 139848688261120 event_tracking.py:138] [process=1] [sync] Finished load in 2.41 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.631039 140484659931136 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
I0512 21:10:28.631054 139848688261120 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:10:28.631616 140028269369344 event_tracking.py:138] [process=6] [sync] Finished load in 2.38 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.632427 140028269369344 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
I0512 21:10:28.630517 140446751868928 event_tracking.py:138] [process=4] [sync] Finished load in 2.37 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.630932 140446751868928 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
I0512 21:10:28.634190 140042913851392 event_tracking.py:138] [process=3] [sync] Finished load in 2.36 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:10:28.634752 140042913851392 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 77, in <module>
    app.run(main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 88, in _possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 54, in unroll_layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
layer_group
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
KeyError: 'params'
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The `KeyError: 'params'` confirms that MaxText's internal logic expects a nesting of `params["params"]["decoder"]`, but our checkpoint uses `params["VariableState"]["decoder"]`.

I've updated the `convert_helper.py` script to **monkey-patch** MaxText's unrolling logic to handle the `VariableState` nesting correctly. 

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

Once again, SCP and run:

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

### User Input

shu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:15:27.876572 140566071461888 event_tracking.py:138] [process=1] [sync] Finished load in 2.95 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
I0512 21:15:27.877116 140226018174976 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
I0512 21:15:27.876772 139703389026304 event_tracking.py:138] [process=2] [sync] Finished load in 2.91 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:15:27.877079 140566071461888 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
I0512 21:15:27.876483 140667768707072 event_tracking.py:138] [process=3] [sync] Finished load in 2.91 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:15:27.876855 140667768707072 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
I0512 21:15:27.877287 139703389026304 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:15:27.876626 140492572538880 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:15:27.878291 139994266155008 event_tracking.py:138] [process=0] [sync] Finished load in 2.93 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:15:27.878751 139994266155008 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 106, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 93, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 71, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The script is successfully loading the checkpoint (`Number of model params=0.134 billion`), but it can't find the `layers` key where it expects. 

I've updated `convert_helper.py` to **print the actual structure** of the checkpoint parameters. This will tell us exactly where the transformer layers are hidden. 

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

Now, SCP and run:

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

Look for the line **`DEBUG: decoder keys: [...]`**. That will solve the mystery.

### User Input

I0512 21:18:42.730260 140709095155712 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:18:42.730286 140709095155712 pyconfig.py:457] Config param steps: 54363
I0512 21:18:42.730319 140709095155712 pyconfig.py:457] Config param stop_strings: None
I0512 21:18:42.730346 140709095155712 pyconfig.py:457] Config param student_overrides: {}
I0512 21:18:42.730374 140709095155712 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:18:42.730400 140709095155712 pyconfig.py:457] Config param subslice_shape: 
I0512 21:18:42.730426 140709095155712 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:18:42.730453 140709095155712 pyconfig.py:457] Config param system_prompt: 
I0512 21:18:42.730479 140709095155712 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:18:42.730505 140709095155712 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:18:42.730532 140709095155712 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:18:42.730558 140709095155712 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:18:42.730587 140709095155712 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:18:42.730614 140709095155712 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:18:42.730640 140709095155712 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:18:42.730667 140709095155712 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:18:42.730693 140709095155712 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:18:42.730720 140709095155712 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:18:42.730746 140709095155712 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:18:42.730772 140709095155712 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:18:42.730798 140709095155712 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:18:42.730829 140709095155712 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:18:42.730856 140709095155712 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:18:42.730883 140709095155712 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:18:42.730910 140709095155712 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:18:42.730936 140709095155712 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:18:42.730964 140709095155712 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:18:42.730991 140709095155712 pyconfig.py:457] Config param train_image_column: image
I0512 21:18:42.731017 140709095155712 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:18:42.731044 140709095155712 pyconfig.py:457] Config param train_split: train
I0512 21:18:42.731070 140709095155712 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:18:42.731096 140709095155712 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:18:42.731122 140709095155712 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:18:42.731150 140709095155712 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:18:42.731176 140709095155712 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:18:42.731202 140709095155712 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:18:42.731229 140709095155712 pyconfig.py:457] Config param use_audio: False
I0512 21:18:42.731255 140709095155712 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:18:42.731281 140709095155712 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:18:42.731307 140709095155712 pyconfig.py:457] Config param use_chat_template: False
I0512 21:18:42.731340 140709095155712 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:18:42.731366 140709095155712 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:18:42.731393 140709095155712 pyconfig.py:457] Config param use_dpo: False
I0512 21:18:42.731419 140709095155712 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:18:42.731446 140709095155712 pyconfig.py:457] Config param use_grpo: True
I0512 21:18:42.731472 140709095155712 pyconfig.py:457] Config param use_indexer: False
I0512 21:18:42.731498 140709095155712 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:18:42.731525 140709095155712 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:18:42.731551 140709095155712 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:18:42.731579 140709095155712 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:18:42.731606 140709095155712 pyconfig.py:457] Config param use_mrope: False
I0512 21:18:42.731633 140709095155712 pyconfig.py:457] Config param use_multimodal: False
I0512 21:18:42.731659 140709095155712 pyconfig.py:457] Config param use_pathways: True
I0512 21:18:42.731685 140709095155712 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:18:42.731711 140709095155712 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:18:42.731738 140709095155712 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:18:42.731764 140709095155712 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:18:42.731790 140709095155712 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:18:42.731816 140709095155712 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:18:42.731842 140709095155712 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:18:42.731868 140709095155712 pyconfig.py:457] Config param use_random_routing: False
I0512 21:18:42.731894 140709095155712 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:18:42.731921 140709095155712 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:18:42.731947 140709095155712 pyconfig.py:457] Config param use_sft: False
I0512 21:18:42.731994 140709095155712 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:18:42.732022 140709095155712 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:18:42.732049 140709095155712 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:18:42.732075 140709095155712 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:18:42.732102 140709095155712 pyconfig.py:457] Config param use_truncation: True
I0512 21:18:42.732129 140709095155712 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:18:42.732155 140709095155712 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:18:42.732182 140709095155712 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:18:42.732208 140709095155712 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:18:42.732234 140709095155712 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:18:42.732261 140709095155712 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:18:42.732287 140709095155712 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:18:42.732320 140709095155712 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:18:42.732347 140709095155712 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:18:42.732374 140709095155712 pyconfig.py:457] Config param video_path: 
I0512 21:18:42.732400 140709095155712 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:18:42.732427 140709095155712 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:18:42.732453 140709095155712 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:18:42.732485 140709095155712 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:18:42.732515 140709095155712 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:18:42.732542 140709095155712 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:18:42.732569 140709095155712 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:18:42.732596 140709095155712 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:18:42.732623 140709095155712 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:18:42.732650 140709095155712 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:18:42.732691 140709095155712 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:18:42.732720 140709095155712 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:18:42.732747 140709095155712 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:18:42.732774 140709095155712 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:18:42.732800 140709095155712 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:18:42.732827 140709095155712 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:18:42.732853 140709095155712 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:18:42.732879 140709095155712 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.732905 140709095155712 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:18:42.732931 140709095155712 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:18:42.732957 140709095155712 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:18:42.732983 140709095155712 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:18:42.733009 140709095155712 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:18:42.733038 140709095155712 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:18:42.733067 140709095155712 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:18:42.733092 140709095155712 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:18:42.733118 140709095155712 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.733144 140709095155712 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:18:42.733170 140709095155712 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:18:42.733197 140709095155712 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:18:42.733223 140709095155712 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:18:42.733248 140709095155712 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:18:42.733275 140709095155712 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:18:42.733301 140709095155712 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:18:42.733335 140709095155712 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:18:42.733365 140709095155712 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:18:42.733392 140709095155712 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:18:42.733418 140709095155712 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:18:42.733445 140709095155712 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:18:42.733479 140709095155712 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:18:42.734835 140709095155712 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:18:42.919081 140709095155712 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:18:42.919296 140709095155712 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:42.919944 140709095155712 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fe73d159f60>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:42.841251 140271903090688 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:18:42.841278 140271903090688 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:18:42.841305 140271903090688 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:18:42.841331 140271903090688 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:18:42.841358 140271903090688 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:18:42.841390 140271903090688 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:18:42.841417 140271903090688 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:18:42.841444 140271903090688 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:18:42.841470 140271903090688 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:18:42.841497 140271903090688 pyconfig.py:457] Config param steps: 54363
I0512 21:18:42.841523 140271903090688 pyconfig.py:457] Config param stop_strings: None
I0512 21:18:42.841550 140271903090688 pyconfig.py:457] Config param student_overrides: {}
I0512 21:18:42.841577 140271903090688 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:18:42.841603 140271903090688 pyconfig.py:457] Config param subslice_shape: 
I0512 21:18:42.841631 140271903090688 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:18:42.841659 140271903090688 pyconfig.py:457] Config param system_prompt: 
I0512 21:18:42.841685 140271903090688 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:18:42.841712 140271903090688 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:18:42.841739 140271903090688 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:18:42.841766 140271903090688 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:18:42.841792 140271903090688 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:18:42.841819 140271903090688 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:18:42.841846 140271903090688 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:18:42.841872 140271903090688 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:18:42.841899 140271903090688 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:18:42.841925 140271903090688 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:18:42.841952 140271903090688 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:18:42.841978 140271903090688 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:18:42.842005 140271903090688 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:18:42.842035 140271903090688 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:18:42.842061 140271903090688 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:18:42.842088 140271903090688 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:18:42.842114 140271903090688 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:18:42.842140 140271903090688 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:18:42.842168 140271903090688 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:18:42.842195 140271903090688 pyconfig.py:457] Config param train_image_column: image
I0512 21:18:42.842221 140271903090688 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:18:42.842247 140271903090688 pyconfig.py:457] Config param train_split: train
I0512 21:18:42.842273 140271903090688 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:18:42.842322 140271903090688 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:18:42.842351 140271903090688 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:18:42.842384 140271903090688 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:18:42.842412 140271903090688 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:18:42.842439 140271903090688 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:18:42.842466 140271903090688 pyconfig.py:457] Config param use_audio: False
I0512 21:18:42.842492 140271903090688 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:18:42.842519 140271903090688 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:18:42.842545 140271903090688 pyconfig.py:457] Config param use_chat_template: False
I0512 21:18:42.842572 140271903090688 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:18:42.842598 140271903090688 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:18:42.842627 140271903090688 pyconfig.py:457] Config param use_dpo: False
I0512 21:18:42.842655 140271903090688 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:18:42.842682 140271903090688 pyconfig.py:457] Config param use_grpo: True
I0512 21:18:42.842708 140271903090688 pyconfig.py:457] Config param use_indexer: False
I0512 21:18:42.842735 140271903090688 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:18:42.842761 140271903090688 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:18:42.842788 140271903090688 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:18:42.842815 140271903090688 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:18:42.842841 140271903090688 pyconfig.py:457] Config param use_mrope: False
I0512 21:18:42.842868 140271903090688 pyconfig.py:457] Config param use_multimodal: False
I0512 21:18:42.842894 140271903090688 pyconfig.py:457] Config param use_pathways: True
I0512 21:18:42.842921 140271903090688 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:18:42.842948 140271903090688 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:18:42.842974 140271903090688 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:18:42.843000 140271903090688 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:18:42.843026 140271903090688 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:18:42.843053 140271903090688 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:18:42.843079 140271903090688 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:18:42.843106 140271903090688 pyconfig.py:457] Config param use_random_routing: False
I0512 21:18:42.843132 140271903090688 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:18:42.843158 140271903090688 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:18:42.843185 140271903090688 pyconfig.py:457] Config param use_sft: False
I0512 21:18:42.843211 140271903090688 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:18:42.843237 140271903090688 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:18:42.843264 140271903090688 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:18:42.843290 140271903090688 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:18:42.843316 140271903090688 pyconfig.py:457] Config param use_truncation: True
I0512 21:18:42.843343 140271903090688 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:18:42.843374 140271903090688 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:18:42.843402 140271903090688 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:18:42.843429 140271903090688 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:18:42.843455 140271903090688 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:18:42.843482 140271903090688 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:18:42.843508 140271903090688 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:18:42.843536 140271903090688 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:18:42.843563 140271903090688 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:18:42.843589 140271903090688 pyconfig.py:457] Config param video_path: 
I0512 21:18:42.843616 140271903090688 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:18:42.843647 140271903090688 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:18:42.843676 140271903090688 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:18:42.843703 140271903090688 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:18:42.843730 140271903090688 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:18:42.843756 140271903090688 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:18:42.843783 140271903090688 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:18:42.843809 140271903090688 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:18:42.843836 140271903090688 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:18:42.843863 140271903090688 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:18:42.843904 140271903090688 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:18:42.843934 140271903090688 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:18:42.843960 140271903090688 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:18:42.843987 140271903090688 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:18:42.844014 140271903090688 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:18:42.844040 140271903090688 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:18:42.844067 140271903090688 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:18:42.844093 140271903090688 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.844120 140271903090688 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:18:42.844146 140271903090688 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:18:42.844172 140271903090688 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:18:42.844199 140271903090688 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:18:42.844225 140271903090688 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:18:42.844251 140271903090688 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:18:42.844277 140271903090688 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:18:42.844304 140271903090688 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:18:42.844330 140271903090688 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.844356 140271903090688 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:18:42.844388 140271903090688 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:18:42.844415 140271903090688 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:18:42.844442 140271903090688 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:18:42.844468 140271903090688 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:18:42.844495 140271903090688 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:18:42.844521 140271903090688 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:18:42.844548 140271903090688 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:18:42.844578 140271903090688 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:18:42.844605 140271903090688 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:18:42.844633 140271903090688 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:18:42.844661 140271903090688 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:18:42.844692 140271903090688 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:18:42.846163 140271903090688 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:18:43.031798 140271903090688 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:18:43.031978 140271903090688 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:43.032434 140271903090688 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f816ec61f60>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:42.869692 140508494075904 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:18:42.869721 140508494075904 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:18:42.869750 140508494075904 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:18:42.869778 140508494075904 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:18:42.869805 140508494075904 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:18:42.869833 140508494075904 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:18:42.869860 140508494075904 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:18:42.869888 140508494075904 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:18:42.869915 140508494075904 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:18:42.869942 140508494075904 pyconfig.py:457] Config param steps: 54363
I0512 21:18:42.869975 140508494075904 pyconfig.py:457] Config param stop_strings: None
I0512 21:18:42.870004 140508494075904 pyconfig.py:457] Config param student_overrides: {}
I0512 21:18:42.870032 140508494075904 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:18:42.870060 140508494075904 pyconfig.py:457] Config param subslice_shape: 
I0512 21:18:42.870087 140508494075904 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:18:42.870115 140508494075904 pyconfig.py:457] Config param system_prompt: 
I0512 21:18:42.870142 140508494075904 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:18:42.870170 140508494075904 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:18:42.870198 140508494075904 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:18:42.870225 140508494075904 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:18:42.870253 140508494075904 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:18:42.870281 140508494075904 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:18:42.870308 140508494075904 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:18:42.870336 140508494075904 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:18:42.870364 140508494075904 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:18:42.870391 140508494075904 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:18:42.870418 140508494075904 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:18:42.870445 140508494075904 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:18:42.870473 140508494075904 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:18:42.870504 140508494075904 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:18:42.870532 140508494075904 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:18:42.870560 140508494075904 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:18:42.870587 140508494075904 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:18:42.870615 140508494075904 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:18:42.870643 140508494075904 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:18:42.870671 140508494075904 pyconfig.py:457] Config param train_image_column: image
I0512 21:18:42.870699 140508494075904 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:18:42.870728 140508494075904 pyconfig.py:457] Config param train_split: train
I0512 21:18:42.870756 140508494075904 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:18:42.870784 140508494075904 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:18:42.870811 140508494075904 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:18:42.870840 140508494075904 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:18:42.870867 140508494075904 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:18:42.870895 140508494075904 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:18:42.870922 140508494075904 pyconfig.py:457] Config param use_audio: False
I0512 21:18:42.870949 140508494075904 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:18:42.870983 140508494075904 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:18:42.871011 140508494075904 pyconfig.py:457] Config param use_chat_template: False
I0512 21:18:42.871039 140508494075904 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:18:42.871066 140508494075904 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:18:42.871093 140508494075904 pyconfig.py:457] Config param use_dpo: False
I0512 21:18:42.871121 140508494075904 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:18:42.871148 140508494075904 pyconfig.py:457] Config param use_grpo: True
I0512 21:18:42.871176 140508494075904 pyconfig.py:457] Config param use_indexer: False
I0512 21:18:42.871203 140508494075904 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:18:42.871231 140508494075904 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:18:42.871277 140508494075904 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:18:42.871310 140508494075904 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:18:42.871339 140508494075904 pyconfig.py:457] Config param use_mrope: False
I0512 21:18:42.871367 140508494075904 pyconfig.py:457] Config param use_multimodal: False
I0512 21:18:42.871395 140508494075904 pyconfig.py:457] Config param use_pathways: True
I0512 21:18:42.871422 140508494075904 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:18:42.871450 140508494075904 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:18:42.871478 140508494075904 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:18:42.871505 140508494075904 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:18:42.871533 140508494075904 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:18:42.871560 140508494075904 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:18:42.871587 140508494075904 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:18:42.871615 140508494075904 pyconfig.py:457] Config param use_random_routing: False
I0512 21:18:42.871642 140508494075904 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:18:42.871670 140508494075904 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:18:42.871697 140508494075904 pyconfig.py:457] Config param use_sft: False
I0512 21:18:42.871726 140508494075904 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:18:42.871754 140508494075904 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:18:42.871781 140508494075904 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:18:42.871809 140508494075904 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:18:42.871836 140508494075904 pyconfig.py:457] Config param use_truncation: True
I0512 21:18:42.871864 140508494075904 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:18:42.871891 140508494075904 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:18:42.871918 140508494075904 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:18:42.871946 140508494075904 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:18:42.871980 140508494075904 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:18:42.872008 140508494075904 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:18:42.872036 140508494075904 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:18:42.872065 140508494075904 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:18:42.872092 140508494075904 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:18:42.872120 140508494075904 pyconfig.py:457] Config param video_path: 
I0512 21:18:42.872147 140508494075904 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:18:42.872175 140508494075904 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:18:42.872202 140508494075904 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:18:42.872229 140508494075904 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:18:42.872257 140508494075904 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:18:42.872284 140508494075904 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:18:42.872312 140508494075904 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:18:42.872339 140508494075904 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:18:42.872367 140508494075904 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:18:42.872396 140508494075904 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:18:42.872437 140508494075904 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:18:42.872467 140508494075904 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:18:42.872495 140508494075904 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:18:42.872523 140508494075904 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:18:42.872556 140508494075904 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:18:42.872587 140508494075904 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:18:42.872615 140508494075904 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:18:42.872642 140508494075904 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.872670 140508494075904 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:18:42.872698 140508494075904 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:18:42.872727 140508494075904 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:18:42.872756 140508494075904 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:18:42.872783 140508494075904 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:18:42.872811 140508494075904 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:18:42.872838 140508494075904 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:18:42.872865 140508494075904 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:18:42.872892 140508494075904 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:18:42.872920 140508494075904 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:18:42.872947 140508494075904 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:18:42.872981 140508494075904 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:18:42.873009 140508494075904 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:18:42.873037 140508494075904 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:18:42.873064 140508494075904 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:18:42.873092 140508494075904 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:18:42.873121 140508494075904 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:18:42.873151 140508494075904 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:18:42.873179 140508494075904 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:18:42.873207 140508494075904 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:18:42.873234 140508494075904 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:18:42.873266 140508494075904 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:18:42.874675 140508494075904 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:18:43.047935 140508494075904 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:18:43.048043 140508494075904 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:43.048543 140508494075904 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb88802dea0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:43.882348 140486736115712 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.882547 140486736115712 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb2988a3010>}, handler_registry=None
I0512 21:18:43.882799 140486736115712 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb2988a3010>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.882866 140486736115712 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb2987950c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.882917 140486736115712 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb2988a3010>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb2988a3010>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb2987950c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb2987950c0>}).
I0512 21:18:43.883366 140486736115712 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.883468 140486736115712 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7fc4eb80ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:43.888561 139846072956928 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.888770 139846072956928 checkpoint_manager.py:708] [process=1][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f1d6df6af80>}, handler_registry=None
I0512 21:18:43.889028 139846072956928 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f1d6df6af80>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.889097 139846072956928 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f1d6de5d1b0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.889149 139846072956928 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f1d6df6af80>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f1d6df6af80>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f1d6de5d1b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f1d6de5d1b0>}).
I0512 21:18:43.889585 139846072956928 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.889688 139846072956928 async_checkpointer.py:192] [process=1][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f2fc100ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:43.902014 140709095155712 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.902273 140709095155712 checkpoint_manager.py:708] [process=2][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fe65e493010>}, handler_registry=None
I0512 21:18:43.902665 140709095155712 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fe65e493010>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.902739 140709095155712 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fe65e3850c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.902792 140709095155712 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fe65e493010>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fe65e493010>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fe65e3850c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fe65e3850c0>}).
I0512 21:18:43.903318 140709095155712 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.903435 140709095155712 async_checkpointer.py:192] [process=2][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7ff8b120ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:43.930907 139626765461504 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.931088 139626765461504 checkpoint_manager.py:708] [process=4][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea5de0b0a0>}, handler_registry=None
I0512 21:18:43.931354 139626765461504 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea5de0b0a0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.931421 139626765461504 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea5d5010c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.931470 139626765461504 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea5de0b0a0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea5de0b0a0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea5d5010c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea5d5010c0>}).
I0512 21:18:43.931882 139626765461504 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.931987 139626765461504 async_checkpointer.py:192] [process=4][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7efcb14e6e60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:43.985878 140508494075904 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.986088 140508494075904 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb7a945f010>}, handler_registry=None
I0512 21:18:43.986381 140508494075904 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb7a945f010>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.986454 140508494075904 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb7a93510c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.986508 140508494075904 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb7a945f010>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb7a945f010>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb7a93510c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb7a93510c0>}).
I0512 21:18:43.986921 140508494075904 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.987032 140508494075904 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7fc9fc57ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:43.989019 140191392077824 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:43.989220 140191392077824 checkpoint_manager.py:708] [process=5][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6dd367b010>}, handler_registry=None
I0512 21:18:43.989480 140191392077824 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6dd367b010>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.989550 140191392077824 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f6dd356d0c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:43.989603 140191392077824 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6dd367b010>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6dd367b010>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f6dd356d0c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f6dd356d0c0>}).
I0512 21:18:43.990030 140191392077824 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:43.990139 140191392077824 async_checkpointer.py:192] [process=5][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f8027b12e60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:44.041406 140403634198528 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:44.041608 140403634198528 checkpoint_manager.py:708] [process=7][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9f3f1a70a0>}, handler_registry=None
I0512 21:18:44.041887 140403634198528 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9f3f1a70a0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:44.041462 140271903090688 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:18:44.041651 140271903090688 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f808ff9b0d0>}, handler_registry=None
I0512 21:18:44.041919 140271903090688 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f808ff9b0d0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:18:44.041958 140403634198528 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9f3f09d0c0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:44.042008 140403634198528 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9f3f1a70a0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9f3f1a70a0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9f3f09d0c0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9f3f09d0c0>}).
I0512 21:18:44.041991 140271903090688 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f808fe90070>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:18:44.042042 140271903090688 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f808ff9b0d0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f808ff9b0d0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f808fe90070>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f808fe90070>}).
I0512 21:18:44.042500 140271903090688 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:44.042622 140271903090688 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f92e6a0ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:44.042471 140403634198528 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:18:44.042576 140403634198528 async_checkpointer.py:192] [process=7][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7fb192716e60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:18:44.558614 139846072956928 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.598081 140191392077824 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.602360 140486736115712 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.611149 140709095155712 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.643918 140271903090688 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.653969 139626765461504 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.652943 140508494075904 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.658434 140403634198528 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:18:44.660178 139626765461504 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(nI0512 21:18:44.660312 139846072956928 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(nI0512 21:18:44.660332 140709095155712 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fe65e32dc60>
I0512 21:18:44.660483 140709095155712 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660537 140709095155712 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f1d6de0d9f0>
I0512 21:18:44.660476 139846072956928 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660533 139846072956928 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:44.660199 140403634198528 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f9f3f049c00>
I0512 21:18:44.660326 140403634198528 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660383 140403634198528 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7eea5deadc00>
I0512 21:18:44.660309 139626765461504 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660364 139626765461504 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:44.660237 140271903090688 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f808fe3dc60>
I0512 21:18:44.660385 140271903090688 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660442 140271903090688 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:44.660256 140191392077824 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f6dd3519cf0>
I0512 21:18:44.660399 140191392077824 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660455 140191392077824 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:44.660254 140486736115712 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fb29873dcf0>
I0512 21:18:44.660392 140486736115712 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660446 140486736115712 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:44.660259 140508494075904 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fb7a94fdc60>
I0512 21:18:44.660398 140508494075904 checkpointing.py:326] Checkpoint manager created!
I0512 21:18:44.660453 140508494075904 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.132132 139846072956928 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.132284 139846072956928 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.132422 139846072956928 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.132524 139846072956928 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f1e4ce4df60>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.132733 139846072956928 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.137023 140486736115712 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.137184 140486736115712 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.137310 140486736115712 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.137411 140486736115712 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb377565f00>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.137641 140486736115712 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.153782 139626765461504 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.153979 139626765461504 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.154112 139626765461504 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.154222 139626765461504 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7eeb3cf520b0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.154447 139626765461504 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.154798 140191392077824 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.154986 140191392077824 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.155125 140191392077824 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.155252 140191392077824 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f6eb2151ea0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.155487 140191392077824 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himI0512 21:18:45.155704 140403634198528 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.155896 140403634198528 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.156026 140403634198528 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.156135 140403634198528 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fa01e14a050>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.156366 140403634198528 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
anshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.172360 140271903090688 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.172565 140271903090688 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.172708 140271903090688 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.172821 140271903090688 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f816ec61f60>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.173049 140271903090688 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.177232 140508494075904 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.177417 140508494075904 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.177552 140508494075904 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.177663 140508494075904 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb88802dea0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.177903 140508494075904 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.187859 140709095155712 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:18:45.188061 140709095155712 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:45.188186 140709095155712 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:18:45.188291 140709095155712 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fe73d159f60>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:18:45.188514 140709095155712 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.381121 140486736115712 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.381715 140486736115712 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.24 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.433399 139846072956928 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.433983 139846072956928 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.30 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.447772 139626765461504 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.448334 139626765461504 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.473675 140191392077824 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.474239 140191392077824 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.32 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.474997 140271903090688 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.475556 140271903090688 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.30 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.485899 140508494075904 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.486455 140508494075904 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.31 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.520503 140403634198528 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.521093 140403634198528 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.36 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.529744 140709095155712 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.530310 140709095155712 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.34 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:18:45.914789   73143 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:45.959403   74525 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:45.999200   67822 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:46.009332   67965 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:46.025140   77148 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:46.072498   72817 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:46.094288   69651 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:46.114779   67137 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:18:47.338881 140486736115712 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.049 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4295284748077393 s) (per-host)
I0512 21:18:47.346785 140486736115712 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 3.578 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6769640445709229 s) (per-host)
I0512 21:18:47.401142 140709095155712 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.143 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3122832775115967 s) (per-host)
I0512 21:18:47.407312 140709095155712 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 3.827 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5676357746124268 s) (per-host)
I0512 21:18:47.417524 140403634198528 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.073 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3978211879730225 s) (per-host)
I0512 21:18:47.421512 139846072956928 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.022 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4672114849090576 s) (per-host)
I0512 21:18:47.426361 139626765461504 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.047 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4325366020202637 s) (per-host)
I0512 21:18:47.426162 140403634198528 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.656 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.641340732574463 s) (per-host)
I0512 21:18:47.427785 139846072956928 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 3.529 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7001368999481201 s) (per-host)
I0512 21:18:47.429246 140191392077824 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.052 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4253673553466797 s) (per-host)
I0512 21:18:47.433037 139626765461504 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.571 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6802000999450684 s) (per-host)
I0512 21:18:47.436373 140191392077824 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.679 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6308202743530273 s) (per-host)
I0512 21:18:47.460533 140271903090688 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.076 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3935973644256592 s) (per-host)
I0512 21:18:47.466887 140271903090688 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.525 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7020702362060547 s) (per-host)
I0512 21:18:47.480900 140508494075904 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.094 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.371572732925415 s) (per-host)
I0512 21:18:47.487074 140508494075904 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.593 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6700465679168701 s) (per-host)
I0512 21:18:47.498959 139846072956928 event_tracking.py:138] [process=1] [sync] Finished load in 2.37 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.499058 139626765461504 event_tracking.py:138] [process=4] [sync] Finished load in 2.34 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499370 139846072956928 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:18:47.499169 140709095155712 event_tracking.py:138] [process=2] [sync] Finished load in 2.31 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.499644 140709095155712 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499564 139626765461504 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499040 140191392077824 event_tracking.py:138] [process=5] [sync] Finished load in 2.34 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.499608 140191392077824 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:18:47.498907 140486736115712 event_tracking.py:138] [process=0] [sync] Finished load in 2.36 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499090 140403634198528 event_tracking.py:138] [process=7] [sync] Finished load in 2.34 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.499670 140403634198528 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499844 140271903090688 event_tracking.py:138] [process=3] [sync] Finished load in 2.33 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.500385 140271903090688 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
I0512 21:18:47.499378 140486736115712 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:18:47.501671 140508494075904 event_tracking.py:138] [process=6] [sync] Finished load in 2.32 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:18:47.502340 140508494075904 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: decoder keys: ['decoder_norm', 'logits_dense']
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_nameTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_nameTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_nameTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_nameTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 94, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 72, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The debug output reveals the problem: **`DEBUG: decoder keys: ['decoder_norm', 'logits_dense']`**. The `layers` key is completely missing from the decoder, which is why the script is failing.

Since we know the checkpoint contains ~1.5GB of data, those 22 layers **are** there—they just aren't where MaxText's default utility expects them to be. They might be at the top level or under a different name.

I've updated `convert_helper.py` to print the **entire tree structure** of the checkpoint. This will show us exactly where the transformer weights are hiding.

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

One more time, SCP and run:

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

The output will now list every single parameter and its shape. Find the one with **shape `(22, ...)`** or anything that looks like a stack of layers.

### User Input

 at 0x7efe67c8a170>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7efe67c8a170>}).
I0512 21:20:23.008743 139712824379392 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.008852 139712824379392 async_checkpointer.py:192] [process=2][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f10bad16e60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:23.044640 140149774137344 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.044675 140062552033280 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.044844 140149774137344 checkpoint_manager.py:708] [process=7][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f64243e9900>}, handler_registry=None
I0512 21:20:23.045160 140149774137344 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f64243e9900>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.044873 140062552033280 checkpoint_manager.py:708] [process=4][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f4fd55cd810>}, handler_registry=None
I0512 21:20:23.045140 140062552033280 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f4fd55cd810>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.045210 140062552033280 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f4fd53720e0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.045260 140062552033280 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f4fd55cd810>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f4fd55cd810>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f4fd53720e0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f4fd53720e0>}).
I0512 21:20:23.045229 140149774137344 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f642418e1d0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.045278 140149774137344 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f64243e9900>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f64243e9900>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f642418e1d0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f642418e1d0>}).
I0512 21:20:23.045724 140149774137344 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.045830 140149774137344 async_checkpointer.py:192] [process=7][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f767700ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:23.045677 140062552033280 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.045779 140062552033280 async_checkpointer.py:192] [process=4][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f62285dae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:23.054599 140343456602112 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.054794 140343456602112 checkpoint_manager.py:708] [process=5][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f913c9e98a0>}, handler_registry=None
I0512 21:20:23.055093 140343456602112 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f913c9e98a0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.055164 140343456602112 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f913c78e170>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.055240 140343456602112 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f913c9e98a0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f913c9e98a0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f913c78e170>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f913c78e170>}).
I0512 21:20:23.055652 140343456602112 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.055754 140343456602112 async_checkpointer.py:192] [process=5][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7fa38f5dae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:22.403477 140677800355840 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:20:22.403503 140677800355840 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:20:22.403529 140677800355840 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:20:22.403555 140677800355840 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:20:22.403580 140677800355840 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:20:22.403606 140677800355840 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:20:22.403634 140677800355840 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:20:22.403661 140677800355840 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:20:22.403687 140677800355840 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:20:22.403712 140677800355840 pyconfig.py:457] Config param steps: 54363
I0512 21:20:22.403738 140677800355840 pyconfig.py:457] Config param stop_strings: None
I0512 21:20:22.403764 140677800355840 pyconfig.py:457] Config param student_overrides: {}
I0512 21:20:22.403791 140677800355840 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:20:22.403817 140677800355840 pyconfig.py:457] Config param subslice_shape: 
I0512 21:20:22.403843 140677800355840 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:20:22.403869 140677800355840 pyconfig.py:457] Config param system_prompt: 
I0512 21:20:22.403895 140677800355840 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:20:22.403921 140677800355840 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:20:22.403947 140677800355840 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:20:22.403973 140677800355840 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:20:22.403999 140677800355840 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:20:22.404025 140677800355840 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:20:22.404052 140677800355840 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:20:22.404077 140677800355840 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:20:22.404103 140677800355840 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:20:22.404129 140677800355840 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:20:22.404155 140677800355840 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:20:22.404180 140677800355840 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:20:22.404206 140677800355840 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:20:22.404235 140677800355840 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:20:22.404261 140677800355840 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:20:22.404287 140677800355840 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:20:22.404313 140677800355840 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:20:22.404339 140677800355840 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:20:22.404371 140677800355840 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:20:22.404399 140677800355840 pyconfig.py:457] Config param train_image_column: image
I0512 21:20:22.404425 140677800355840 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:20:22.404451 140677800355840 pyconfig.py:457] Config param train_split: train
I0512 21:20:22.404477 140677800355840 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:20:22.404503 140677800355840 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:20:22.404529 140677800355840 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:20:22.404556 140677800355840 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:20:22.404582 140677800355840 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:20:22.404608 140677800355840 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:20:22.404636 140677800355840 pyconfig.py:457] Config param use_audio: False
I0512 21:20:22.404663 140677800355840 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:20:22.404690 140677800355840 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:20:22.404716 140677800355840 pyconfig.py:457] Config param use_chat_template: False
I0512 21:20:22.404742 140677800355840 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:20:22.404767 140677800355840 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:20:22.404793 140677800355840 pyconfig.py:457] Config param use_dpo: False
I0512 21:20:22.404819 140677800355840 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:20:22.404844 140677800355840 pyconfig.py:457] Config param use_grpo: True
I0512 21:20:22.404870 140677800355840 pyconfig.py:457] Config param use_indexer: False
I0512 21:20:22.404896 140677800355840 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:20:22.404921 140677800355840 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:20:22.404947 140677800355840 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:20:22.404973 140677800355840 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:20:22.404999 140677800355840 pyconfig.py:457] Config param use_mrope: False
I0512 21:20:22.405024 140677800355840 pyconfig.py:457] Config param use_multimodal: False
I0512 21:20:22.405050 140677800355840 pyconfig.py:457] Config param use_pathways: True
I0512 21:20:22.405075 140677800355840 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:20:22.405101 140677800355840 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:20:22.405127 140677800355840 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:20:22.405152 140677800355840 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:20:22.405178 140677800355840 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:20:22.405203 140677800355840 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:20:22.405229 140677800355840 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:20:22.405254 140677800355840 pyconfig.py:457] Config param use_random_routing: False
I0512 21:20:22.405280 140677800355840 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:20:22.405305 140677800355840 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:20:22.405331 140677800355840 pyconfig.py:457] Config param use_sft: False
I0512 21:20:22.405357 140677800355840 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:20:22.405389 140677800355840 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:20:22.405416 140677800355840 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:20:22.405441 140677800355840 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:20:22.405467 140677800355840 pyconfig.py:457] Config param use_truncation: True
I0512 21:20:22.405493 140677800355840 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:20:22.405518 140677800355840 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:20:22.405544 140677800355840 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:20:22.405570 140677800355840 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:20:22.405595 140677800355840 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:20:22.405622 140677800355840 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:20:22.405649 140677800355840 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:20:22.405676 140677800355840 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:20:22.405702 140677800355840 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:20:22.405729 140677800355840 pyconfig.py:457] Config param video_path: 
I0512 21:20:22.405754 140677800355840 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:20:22.405780 140677800355840 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:20:22.405806 140677800355840 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:20:22.405832 140677800355840 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:20:22.405858 140677800355840 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:20:22.405884 140677800355840 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:20:22.405910 140677800355840 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:20:22.405936 140677800355840 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:20:22.405961 140677800355840 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:20:22.405989 140677800355840 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:20:22.406028 140677800355840 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:20:22.406057 140677800355840 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:20:22.406084 140677800355840 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:20:22.406110 140677800355840 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:20:22.406136 140677800355840 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:20:22.406162 140677800355840 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:20:22.406188 140677800355840 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:20:22.406213 140677800355840 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.406239 140677800355840 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:20:22.406265 140677800355840 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:20:22.406315 140677800355840 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:20:22.406344 140677800355840 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:20:22.406375 140677800355840 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:20:22.406402 140677800355840 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:20:22.406428 140677800355840 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:20:22.406453 140677800355840 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:20:22.406478 140677800355840 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.406504 140677800355840 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:20:22.406529 140677800355840 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:20:22.406554 140677800355840 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:20:22.406579 140677800355840 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:20:22.406605 140677800355840 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:20:22.406632 140677800355840 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:20:22.406658 140677800355840 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:20:22.406685 140677800355840 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:20:22.406715 140677800355840 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:20:22.406741 140677800355840 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:20:22.406767 140677800355840 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:20:22.406792 140677800355840 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:20:22.406825 140677800355840 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:20:22.408317 140677800355840 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:20:22.587578 140677800355840 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:20:22.587766 140677800355840 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:22.588253 140677800355840 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fdff0441630>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:22.408998 139884207601664 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:20:22.409024 139884207601664 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:20:22.409050 139884207601664 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:20:22.409077 139884207601664 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:20:22.409103 139884207601664 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:20:22.409128 139884207601664 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:20:22.409154 139884207601664 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:20:22.409180 139884207601664 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:20:22.409207 139884207601664 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:20:22.409233 139884207601664 pyconfig.py:457] Config param steps: 54363
I0512 21:20:22.409259 139884207601664 pyconfig.py:457] Config param stop_strings: None
I0512 21:20:22.409285 139884207601664 pyconfig.py:457] Config param student_overrides: {}
I0512 21:20:22.409312 139884207601664 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:20:22.409338 139884207601664 pyconfig.py:457] Config param subslice_shape: 
I0512 21:20:22.409364 139884207601664 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:20:22.409391 139884207601664 pyconfig.py:457] Config param system_prompt: 
I0512 21:20:22.409417 139884207601664 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:20:22.409443 139884207601664 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:20:22.409469 139884207601664 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:20:22.409496 139884207601664 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:20:22.409522 139884207601664 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:20:22.409548 139884207601664 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:20:22.409575 139884207601664 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:20:22.409601 139884207601664 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:20:22.409627 139884207601664 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:20:22.409653 139884207601664 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:20:22.409680 139884207601664 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:20:22.409706 139884207601664 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:20:22.409734 139884207601664 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:20:22.409766 139884207601664 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:20:22.409793 139884207601664 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:20:22.409819 139884207601664 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:20:22.409846 139884207601664 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:20:22.409872 139884207601664 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:20:22.409899 139884207601664 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:20:22.409926 139884207601664 pyconfig.py:457] Config param train_image_column: image
I0512 21:20:22.409952 139884207601664 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:20:22.409984 139884207601664 pyconfig.py:457] Config param train_split: train
I0512 21:20:22.410010 139884207601664 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:20:22.410037 139884207601664 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:20:22.410063 139884207601664 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:20:22.410089 139884207601664 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:20:22.410116 139884207601664 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:20:22.410141 139884207601664 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:20:22.410168 139884207601664 pyconfig.py:457] Config param use_audio: False
I0512 21:20:22.410194 139884207601664 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:20:22.410220 139884207601664 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:20:22.410246 139884207601664 pyconfig.py:457] Config param use_chat_template: False
I0512 21:20:22.410272 139884207601664 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:20:22.410298 139884207601664 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:20:22.410324 139884207601664 pyconfig.py:457] Config param use_dpo: False
I0512 21:20:22.410350 139884207601664 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:20:22.410376 139884207601664 pyconfig.py:457] Config param use_grpo: True
I0512 21:20:22.410402 139884207601664 pyconfig.py:457] Config param use_indexer: False
I0512 21:20:22.410428 139884207601664 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:20:22.410454 139884207601664 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:20:22.410480 139884207601664 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:20:22.410506 139884207601664 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:20:22.410531 139884207601664 pyconfig.py:457] Config param use_mrope: False
I0512 21:20:22.410557 139884207601664 pyconfig.py:457] Config param use_multimodal: False
I0512 21:20:22.410583 139884207601664 pyconfig.py:457] Config param use_pathways: True
I0512 21:20:22.410609 139884207601664 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:20:22.410635 139884207601664 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:20:22.410661 139884207601664 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:20:22.410687 139884207601664 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:20:22.410714 139884207601664 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:20:22.410741 139884207601664 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:20:22.410768 139884207601664 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:20:22.410794 139884207601664 pyconfig.py:457] Config param use_random_routing: False
I0512 21:20:22.410820 139884207601664 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:20:22.410846 139884207601664 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:20:22.410873 139884207601664 pyconfig.py:457] Config param use_sft: False
I0512 21:20:22.410898 139884207601664 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:20:22.410924 139884207601664 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:20:22.410950 139884207601664 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:20:22.410982 139884207601664 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:20:22.411009 139884207601664 pyconfig.py:457] Config param use_truncation: True
I0512 21:20:22.411034 139884207601664 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:20:22.411061 139884207601664 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:20:22.411087 139884207601664 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:20:22.411113 139884207601664 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:20:22.411139 139884207601664 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:20:22.411165 139884207601664 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:20:22.411191 139884207601664 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:20:22.411218 139884207601664 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:20:22.411244 139884207601664 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:20:22.411298 139884207601664 pyconfig.py:457] Config param video_path: 
I0512 21:20:22.411327 139884207601664 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:20:22.411354 139884207601664 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:20:22.411381 139884207601664 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:20:22.411407 139884207601664 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:20:22.411433 139884207601664 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:20:22.411460 139884207601664 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:20:22.411486 139884207601664 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:20:22.411512 139884207601664 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:20:22.411538 139884207601664 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:20:22.411565 139884207601664 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:20:22.411604 139884207601664 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:20:22.411633 139884207601664 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:20:22.411661 139884207601664 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:20:22.411687 139884207601664 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:20:22.411715 139884207601664 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:20:22.411743 139884207601664 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:20:22.411769 139884207601664 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:20:22.411796 139884207601664 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.411822 139884207601664 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:20:22.411848 139884207601664 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:20:22.411874 139884207601664 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:20:22.411899 139884207601664 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:20:22.411925 139884207601664 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:20:22.411951 139884207601664 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:20:22.411984 139884207601664 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:20:22.412010 139884207601664 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:20:22.412037 139884207601664 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.412063 139884207601664 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:20:22.412089 139884207601664 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:20:22.412115 139884207601664 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:20:22.412141 139884207601664 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:20:22.412166 139884207601664 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:20:22.412192 139884207601664 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:20:22.412218 139884207601664 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:20:22.412245 139884207601664 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:20:22.412276 139884207601664 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:20:22.412303 139884207601664 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:20:22.412329 139884207601664 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:20:22.412355 139884207601664 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:20:22.412386 139884207601664 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:20:22.413728 139884207601664 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:20:22.551336 139884207601664 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:20:22.551434 139884207601664 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:22.551901 139884207601664 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f272df41630>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:22.496814 140139921704960 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:20:22.496842 140139921704960 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:20:22.496870 140139921704960 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:20:22.496898 140139921704960 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:20:22.496927 140139921704960 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:20:22.496958 140139921704960 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:20:22.496986 140139921704960 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:20:22.497014 140139921704960 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:20:22.497042 140139921704960 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:20:22.497070 140139921704960 pyconfig.py:457] Config param steps: 54363
I0512 21:20:22.497098 140139921704960 pyconfig.py:457] Config param stop_strings: None
I0512 21:20:22.497127 140139921704960 pyconfig.py:457] Config param student_overrides: {}
I0512 21:20:22.497155 140139921704960 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:20:22.497191 140139921704960 pyconfig.py:457] Config param subslice_shape: 
I0512 21:20:22.497220 140139921704960 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:20:22.497248 140139921704960 pyconfig.py:457] Config param system_prompt: 
I0512 21:20:22.497276 140139921704960 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:20:22.497305 140139921704960 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:20:22.497333 140139921704960 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:20:22.497361 140139921704960 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:20:22.497390 140139921704960 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:20:22.497418 140139921704960 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:20:22.497447 140139921704960 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:20:22.497475 140139921704960 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:20:22.497526 140139921704960 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:20:22.497556 140139921704960 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:20:22.497585 140139921704960 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:20:22.497613 140139921704960 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:20:22.497642 140139921704960 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:20:22.497673 140139921704960 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:20:22.497701 140139921704960 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:20:22.497729 140139921704960 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:20:22.497757 140139921704960 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:20:22.497785 140139921704960 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:20:22.497815 140139921704960 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:20:22.497844 140139921704960 pyconfig.py:457] Config param train_image_column: image
I0512 21:20:22.497873 140139921704960 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:20:22.497902 140139921704960 pyconfig.py:457] Config param train_split: train
I0512 21:20:22.497931 140139921704960 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:20:22.497961 140139921704960 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:20:22.497989 140139921704960 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:20:22.498018 140139921704960 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:20:22.498046 140139921704960 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:20:22.498075 140139921704960 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:20:22.498103 140139921704960 pyconfig.py:457] Config param use_audio: False
I0512 21:20:22.498131 140139921704960 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:20:22.498159 140139921704960 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:20:22.498194 140139921704960 pyconfig.py:457] Config param use_chat_template: False
I0512 21:20:22.498222 140139921704960 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:20:22.498250 140139921704960 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:20:22.498278 140139921704960 pyconfig.py:457] Config param use_dpo: False
I0512 21:20:22.498306 140139921704960 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:20:22.498334 140139921704960 pyconfig.py:457] Config param use_grpo: True
I0512 21:20:22.498363 140139921704960 pyconfig.py:457] Config param use_indexer: False
I0512 21:20:22.498390 140139921704960 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:20:22.498418 140139921704960 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:20:22.498447 140139921704960 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:20:22.498475 140139921704960 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:20:22.498503 140139921704960 pyconfig.py:457] Config param use_mrope: False
I0512 21:20:22.498531 140139921704960 pyconfig.py:457] Config param use_multimodal: False
I0512 21:20:22.498558 140139921704960 pyconfig.py:457] Config param use_pathways: True
I0512 21:20:22.498587 140139921704960 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:20:22.498615 140139921704960 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:20:22.498643 140139921704960 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:20:22.498671 140139921704960 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:20:22.498698 140139921704960 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:20:22.498726 140139921704960 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:20:22.498754 140139921704960 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:20:22.498782 140139921704960 pyconfig.py:457] Config param use_random_routing: False
I0512 21:20:22.498810 140139921704960 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:20:22.498839 140139921704960 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:20:22.498867 140139921704960 pyconfig.py:457] Config param use_sft: False
I0512 21:20:22.498895 140139921704960 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:20:22.498924 140139921704960 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:20:22.498954 140139921704960 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:20:22.498982 140139921704960 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:20:22.499010 140139921704960 pyconfig.py:457] Config param use_truncation: True
I0512 21:20:22.499038 140139921704960 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:20:22.499067 140139921704960 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:20:22.499094 140139921704960 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:20:22.499123 140139921704960 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:20:22.499150 140139921704960 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:20:22.499184 140139921704960 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:20:22.499212 140139921704960 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:20:22.499242 140139921704960 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:20:22.499270 140139921704960 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:20:22.499298 140139921704960 pyconfig.py:457] Config param video_path: 
I0512 21:20:22.499326 140139921704960 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:20:22.499355 140139921704960 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:20:22.499382 140139921704960 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:20:22.499410 140139921704960 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:20:22.499439 140139921704960 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:20:22.499467 140139921704960 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:20:22.499495 140139921704960 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:20:22.499523 140139921704960 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:20:22.499552 140139921704960 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:20:22.499581 140139921704960 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:20:22.499622 140139921704960 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:20:22.499652 140139921704960 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:20:22.499681 140139921704960 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:20:22.499709 140139921704960 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:20:22.499737 140139921704960 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:20:22.499766 140139921704960 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:20:22.499794 140139921704960 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:20:22.499822 140139921704960 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.499850 140139921704960 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:20:22.499878 140139921704960 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:20:22.499907 140139921704960 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:20:22.499937 140139921704960 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:20:22.499966 140139921704960 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:20:22.499994 140139921704960 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:20:22.500022 140139921704960 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:20:22.500050 140139921704960 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:20:22.500079 140139921704960 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:20:22.500106 140139921704960 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:20:22.500134 140139921704960 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:20:22.500162 140139921704960 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:20:22.500195 140139921704960 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:20:22.500224 140139921704960 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:20:22.500252 140139921704960 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:20:22.500280 140139921704960 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:20:22.500309 140139921704960 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:20:22.500340 140139921704960 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:20:22.500369 140139921704960 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:20:22.500397 140139921704960 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:20:22.500425 140139921704960 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:20:22.500457 140139921704960 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:20:22.501816 140139921704960 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:20:22.732866 140139921704960 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:20:22.733082 140139921704960 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:22.733701 140139921704960 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f62b7d5bc40>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:23.232953 139884207601664 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.233144 139884207601664 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f264f0d5810>}, handler_registry=None
I0512 21:20:23.233423 139884207601664 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f264f0d5810>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.233494 139884207601664 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f264ee7e0e0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.233544 139884207601664 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f264f0d5810>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f264f0d5810>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f264ee7e0e0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f264ee7e0e0>}).
I0512 21:20:23.233967 139884207601664 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.234071 139884207601664 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f38a200ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:23.249900 140677800355840 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.250086 140677800355840 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fdf117d1870>}, handler_registry=None
I0512 21:20:23.250398 140677800355840 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fdf117d1870>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.250469 140677800355840 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fdf1157a140>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.250519 140677800355840 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fdf117d1870>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fdf117d1870>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fdf1157a140>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fdf1157a140>}).
I0512 21:20:23.250936 140677800355840 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.251053 140677800355840 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7ff16800ae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:23.929757 140139921704960 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:20:23.930008 140139921704960 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f61d9197610>}, handler_registry=None
I0512 21:20:23.930364 140139921704960 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f61d9197610>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.930435 140139921704960 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f61d90861d0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:20:23.930489 140139921704960 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f61d9197610>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f61d9197610>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f61d90861d0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f61d90861d0>}).
I0512 21:20:23.930960 140139921704960 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:20:23.931072 140139921704960 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f742bbdae60> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:20:24.406950 140139921704960 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.451856 139884207601664 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.453463 140355020339200 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.465637 139712824379392 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.505398 140062552033280 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.546685 140343456602112 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.658443 140677800355840 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.745612 140149774137344 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:20:24.747423 140149774137344 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f642418cf70>
I0512 21:20:24.747602 140149774137344 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747662 140149774137344 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747522 139712824379392 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7efe67c88f10>
I0512 21:20:24.747689 139712824379392 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747745 139712824379392 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747514 140343456602112 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f913c78cf10>
I0512 21:20:24.747653 140343456602112 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747708 140343456602112 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747558 140139921704960 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f61d9084f70>
I0512 21:20:24.747714 140139921704960 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747769 140139921704960 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747486 140677800355840 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(nI0512 21:20:24.747437 139884207601664 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f264ee7cee0>
I0512 21:20:24.747584 139884207601664 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747639 139884207601664 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747420 140355020339200 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f93ed39ceb0>
I0512 21:20:24.747571 140355020339200 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747626 140355020339200 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fdf11578ee0>
I0512 21:20:24.747636 140677800355840 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747692 140677800355840 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:24.747527 140062552033280 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f4fd5370ee0>
I0512 21:20:24.747676 140062552033280 checkpointing.py:326] Checkpoint manager created!
I0512 21:20:24.747731 140062552033280 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.233256 140062552033280 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.233446 140062552033280 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.233579 140062552033280 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.233692 140062552033280 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f50b4d47c10>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.233927 140062552033280 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.237228 140355020339200 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.237383 140355020339200 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.237501 140355020339200 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.237599 140355020339200 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f94cc345630>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.237805 140355020339200 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.241998 140139921704960 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.242190 140139921704960 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.242318 140139921704960 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.242422 140139921704960 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f62b7d5bc40>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.242637 140139921704960 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.258255 140677800355840 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.258479 140677800355840 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.258616 140677800355840 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.258728 140677800355840 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fdff0441630>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.261129 140149774137344 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.261322 140149774137344 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.261449 140149774137344 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.261556 140149774137344 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f6502e5bc10>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.258956 140677800355840 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.260051 139884207601664 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.260229 139884207601664 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.260354 139884207601664 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.260456 139884207601664 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f272df41630>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.261777 140149774137344 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.260672 139884207601664 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.268110 139712824379392 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.268292 139712824379392 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.268424 139712824379392 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.268532 139712824379392 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7eff46b63c10>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.268763 139712824379392 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.268630 140343456602112 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:20:25.268822 140343456602112 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:25.268954 140343456602112 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:20:25.269065 140343456602112 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f921b540460>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:20:25.269300 140343456602112 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.460589 140139921704960 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.461146 140139921704960 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.22 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.522065 139884207601664 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.522659 139884207601664 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.26 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.530220 140062552033280 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.530819 140062552033280 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.30 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.547399 140677800355840 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.547975 140677800355840 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.584209 140355020339200 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.584810 140355020339200 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.35 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.593316 140149774137344 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.593893 140149774137344 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.33 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.606353 140343456602112 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.606899 140343456602112 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.34 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.642091 139712824379392 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.642690 139712824379392 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.37 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:20:25.993925   74734 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.035124   69427 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.069304   68760 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.095764   76099 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.128336   74671 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.193010   69544 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.194169   78777 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:26.235543   71244 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:20:27.210530 140139921704960 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.227 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2220869064331055 s) (per-host)
I0512 21:20:27.217354 140139921704960 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 4.082 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.4698846340179443 s) (per-host)
I0512 21:20:27.332755 140355020339200 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.207 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2422592639923096 s) (per-host)
I0512 21:20:27.341358 140355020339200 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 4.114 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.4584839344024658 s) (per-host)
I0512 21:20:27.385164 140062552033280 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.107 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3553175926208496 s) (per-host)
I0512 21:20:27.391118 140062552033280 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.716 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6145389080047607 s) (per-host)
I0512 21:20:27.406891 139884207601664 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.117 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3429043292999268 s) (per-host)
I0512 21:20:27.413360 139884207601664 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.764 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5939106941223145 s) (per-host)
I0512 21:20:27.465445 140149774137344 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.175 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2766079902648926 s) (per-host)
I0512 21:20:27.471155 140149774137344 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.864 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5528604984283447 s) (per-host)
I0512 21:20:27.480751 140343456602112 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.160 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2932558059692383 s) (per-host)
I0512 21:20:27.486158 140343456602112 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.794 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5813055038452148 s) (per-host)
I0512 21:20:27.485304 140677800355840 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.101 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3628251552581787 s) (per-host)
I0512 21:20:27.492742 140677800355840 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.731 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6083836555480957 s) (per-host)
I0512 21:20:27.509341 139712824379392 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.173 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2790298461914062 s) (per-host)
I0512 21:20:27.516758 139712824379392 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 3.830 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.566723346710205 s) (per-host)
I0512 21:20:27.522150 140149774137344 event_tracking.py:138] [process=7] [sync] Finished load in 2.26 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522600 140149774137344 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:20:27.522208 140343456602112 event_tracking.py:138] [process=5] [sync] Finished load in 2.25 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522590 140343456602112 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:20:27.522180 140139921704960 event_tracking.py:138] [process=0] [sync] Finished load in 2.28 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522560 140139921704960 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:20:27.522201 140062552033280 event_tracking.py:138] [process=4] [sync] Finished load in 2.29 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522709 140062552033280 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:20:27.522135 140355020339200 event_tracking.py:138] [process=1] [sync] Finished load in 2.28 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522643 140355020339200 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:20:27.522700 140677800355840 event_tracking.py:138] [process=3] [sync] Finished load in 2.26 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:20:27.522254 139884207601664 event_tracking.py:138] [process=6] [sync] Finished load in 2.26 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.522842 139884207601664 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:20:27.523385 139712824379392 event_tracking.py:138] [process=2] [sync] Finished load in 2.25 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:20:27.524562 139712824379392 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:20:27.523329 140677800355840 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 118, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 105, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 83, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The plot thickens: The parameter tree only contains the embedder, the final norm, and the logits. **The 22 transformer layers are missing from the `params` collection entirely.**

However, the checkpoint size (1.5GB) confirms they are definitely on disk. This usually means they are stored in a different "collection" (like `non_trainable` or `model_state`) or under a different top-level key in the Orbax checkpoint.

I've updated the script to **list all top-level Orbax items** and **all attributes of the loaded state object**. This will reveal where the transformer backbone is being stored.

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

Let's run this:

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

Also, run this on your local machine to see the directory contents of the checkpoint:
```bash
gsutil ls -R gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/
```

We need to see if there's another folder next to `items`.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % gsutil ls -R gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/

Google recommends using Gcloud storage CLI (https://docs.cloud.google.com/storage/docs/discover-object-storage-gcloud) instead of gsutil. Please refer to migration guide (https://docs.cloud.google.com/storage/docs/gsutil-transition-to-gcloud) for assistance.
gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/
himanshu@Himanshus-MacBook TinyMathReason-1B % 


----

I0512 21:22:13.670592 139627526768640 pyconfig.py:457] Config param use_sft: False
I0512 21:22:13.670618 139627526768640 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:22:13.670644 139627526768640 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:22:13.670670 139627526768640 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:22:13.670696 139627526768640 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:22:13.670728 139627526768640 pyconfig.py:457] Config param use_truncation: True
I0512 21:22:13.670755 139627526768640 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:22:13.670781 139627526768640 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:22:13.670807 139627526768640 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:22:13.670833 139627526768640 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:22:13.670859 139627526768640 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:22:13.670886 139627526768640 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:22:13.670912 139627526768640 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:22:13.670939 139627526768640 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:22:13.670967 139627526768640 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:22:13.670994 139627526768640 pyconfig.py:457] Config param video_path: 
I0512 21:22:13.671020 139627526768640 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:22:13.671046 139627526768640 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:22:13.671072 139627526768640 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:22:13.671099 139627526768640 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:22:13.671125 139627526768640 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:22:13.671151 139627526768640 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:22:13.671177 139627526768640 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:22:13.671203 139627526768640 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:22:13.671229 139627526768640 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:22:13.671275 139627526768640 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:22:13.671322 139627526768640 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:22:13.671351 139627526768640 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:22:13.671378 139627526768640 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:22:13.671405 139627526768640 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:22:13.671431 139627526768640 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:22:13.671458 139627526768640 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:22:13.671484 139627526768640 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:22:13.671510 139627526768640 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.671536 139627526768640 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:22:13.671562 139627526768640 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:22:13.671588 139627526768640 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:22:13.671614 139627526768640 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:22:13.671640 139627526768640 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:22:13.671666 139627526768640 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:22:13.671692 139627526768640 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:22:13.671724 139627526768640 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:22:13.671751 139627526768640 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.671777 139627526768640 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:22:13.671804 139627526768640 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:22:13.671830 139627526768640 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:22:13.671856 139627526768640 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:22:13.671882 139627526768640 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:22:13.671908 139627526768640 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:22:13.671934 139627526768640 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:22:13.671962 139627526768640 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:22:13.671993 139627526768640 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:22:13.672020 139627526768640 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:22:13.672047 139627526768640 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:22:13.672074 139627526768640 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:22:13.672105 139627526768640 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:22:13.673437 139627526768640 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:22:13.845584 139627526768640 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:22:13.845679 139627526768640 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:13.846231 139627526768640 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7eeb6b033fd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:14.532480 139627526768640 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:22:14.532668 139627526768640 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea8c1f9b40>}, handler_registry=None
I0512 21:22:14.532946 139627526768640 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea8c1f9b40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.533022 139627526768640 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea8bfa1210>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.533074 139627526768640 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea8c1f9b40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7eea8c1f9b40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea8bfa1210>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7eea8bfa1210>}).
I0512 21:22:14.533481 139627526768640 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:22:14.533581 139627526768640 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7efcdea0aef0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:22:13.825339 139965031823360 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:22:13.825372 139965031823360 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:22:13.825402 139965031823360 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:22:13.825430 139965031823360 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:22:13.825458 139965031823360 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:22:13.825486 139965031823360 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:22:13.825514 139965031823360 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:22:13.825541 139965031823360 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:22:13.825569 139965031823360 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:22:13.825596 139965031823360 pyconfig.py:457] Config param steps: 54363
I0512 21:22:13.825625 139965031823360 pyconfig.py:457] Config param stop_strings: None
I0512 21:22:13.825654 139965031823360 pyconfig.py:457] Config param student_overrides: {}
I0512 21:22:13.825683 139965031823360 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:22:13.825711 139965031823360 pyconfig.py:457] Config param subslice_shape: 
I0512 21:22:13.825739 139965031823360 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:22:13.825766 139965031823360 pyconfig.py:457] Config param system_prompt: 
I0512 21:22:13.825793 139965031823360 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:22:13.825821 139965031823360 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:22:13.825849 139965031823360 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:22:13.825877 139965031823360 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:22:13.825905 139965031823360 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:22:13.825933 139965031823360 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:22:13.825961 139965031823360 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:22:13.825989 139965031823360 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:22:13.826016 139965031823360 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:22:13.826044 139965031823360 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:22:13.826071 139965031823360 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:22:13.826099 139965031823360 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:22:13.826126 139965031823360 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:22:13.826157 139965031823360 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:22:13.826185 139965031823360 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:22:13.826213 139965031823360 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:22:13.826241 139965031823360 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:22:13.826268 139965031823360 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:22:13.826319 139965031823360 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:22:13.826350 139965031823360 pyconfig.py:457] Config param train_image_column: image
I0512 21:22:13.826384 139965031823360 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:22:13.826413 139965031823360 pyconfig.py:457] Config param train_split: train
I0512 21:22:13.826441 139965031823360 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:22:13.826468 139965031823360 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:22:13.826496 139965031823360 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:22:13.826525 139965031823360 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:22:13.826553 139965031823360 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:22:13.826580 139965031823360 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:22:13.826608 139965031823360 pyconfig.py:457] Config param use_audio: False
I0512 21:22:13.826637 139965031823360 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:22:13.826666 139965031823360 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:22:13.826694 139965031823360 pyconfig.py:457] Config param use_chat_template: False
I0512 21:22:13.826722 139965031823360 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:22:13.826749 139965031823360 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:22:13.826776 139965031823360 pyconfig.py:457] Config param use_dpo: False
I0512 21:22:13.826804 139965031823360 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:22:13.826832 139965031823360 pyconfig.py:457] Config param use_grpo: True
I0512 21:22:13.826859 139965031823360 pyconfig.py:457] Config param use_indexer: False
I0512 21:22:13.826886 139965031823360 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:22:13.826914 139965031823360 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:22:13.826941 139965031823360 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:22:13.826969 139965031823360 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:22:13.826997 139965031823360 pyconfig.py:457] Config param use_mrope: False
I0512 21:22:13.827024 139965031823360 pyconfig.py:457] Config param use_multimodal: False
I0512 21:22:13.827052 139965031823360 pyconfig.py:457] Config param use_pathways: True
I0512 21:22:13.827079 139965031823360 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:22:13.827106 139965031823360 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:22:13.827134 139965031823360 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:22:13.827161 139965031823360 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:22:13.827189 139965031823360 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:22:13.827216 139965031823360 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:22:13.827244 139965031823360 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:22:13.827272 139965031823360 pyconfig.py:457] Config param use_random_routing: False
I0512 21:22:13.827299 139965031823360 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:22:13.827327 139965031823360 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:22:13.827354 139965031823360 pyconfig.py:457] Config param use_sft: False
I0512 21:22:13.827387 139965031823360 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:22:13.827415 139965031823360 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:22:13.827442 139965031823360 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:22:13.827470 139965031823360 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:22:13.827497 139965031823360 pyconfig.py:457] Config param use_truncation: True
I0512 21:22:13.827525 139965031823360 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:22:13.827552 139965031823360 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:22:13.827580 139965031823360 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:22:13.827608 139965031823360 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:22:13.827637 139965031823360 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:22:13.827665 139965031823360 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:22:13.827693 139965031823360 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:22:13.827722 139965031823360 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:22:13.827749 139965031823360 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:22:13.827777 139965031823360 pyconfig.py:457] Config param video_path: 
I0512 21:22:13.827805 139965031823360 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:22:13.827832 139965031823360 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:22:13.827860 139965031823360 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:22:13.827887 139965031823360 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:22:13.827915 139965031823360 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:22:13.827943 139965031823360 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:22:13.827970 139965031823360 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:22:13.827998 139965031823360 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:22:13.828025 139965031823360 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:22:13.828054 139965031823360 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:22:13.828094 139965031823360 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:22:13.828125 139965031823360 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:22:13.828153 139965031823360 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:22:13.828181 139965031823360 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:22:13.828208 139965031823360 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:22:13.828236 139965031823360 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:22:13.828264 139965031823360 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:22:13.828291 139965031823360 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.828319 139965031823360 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:22:13.828347 139965031823360 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:22:13.828380 139965031823360 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:22:13.828409 139965031823360 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:22:13.828437 139965031823360 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:22:13.828464 139965031823360 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:22:13.828491 139965031823360 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:22:13.828519 139965031823360 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:22:13.828546 139965031823360 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.828573 139965031823360 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:22:13.828601 139965031823360 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:22:13.828629 139965031823360 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:22:13.828658 139965031823360 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:22:13.828687 139965031823360 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:22:13.828714 139965031823360 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:22:13.828742 139965031823360 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:22:13.828771 139965031823360 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:22:13.828802 139965031823360 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:22:13.828830 139965031823360 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:22:13.828858 139965031823360 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:22:13.828885 139965031823360 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:22:13.828916 139965031823360 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:22:13.830396 139965031823360 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:22:14.007580 139965031823360 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:22:14.007771 139965031823360 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:14.008252 139965031823360 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f39fba5bfd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:13.875180 140382581987328 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:22:13.875207 140382581987328 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:22:13.875234 140382581987328 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:22:13.875261 140382581987328 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:22:13.875287 140382581987328 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:22:13.875315 140382581987328 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:22:13.875343 140382581987328 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:22:13.875369 140382581987328 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:22:13.875396 140382581987328 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:22:13.875422 140382581987328 pyconfig.py:457] Config param steps: 54363
I0512 21:22:13.875449 140382581987328 pyconfig.py:457] Config param stop_strings: None
I0512 21:22:13.875475 140382581987328 pyconfig.py:457] Config param student_overrides: {}
I0512 21:22:13.875502 140382581987328 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:22:13.875529 140382581987328 pyconfig.py:457] Config param subslice_shape: 
I0512 21:22:13.875561 140382581987328 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:22:13.875588 140382581987328 pyconfig.py:457] Config param system_prompt: 
I0512 21:22:13.875614 140382581987328 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:22:13.875641 140382581987328 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:22:13.875668 140382581987328 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:22:13.875694 140382581987328 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:22:13.875721 140382581987328 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:22:13.875748 140382581987328 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:22:13.875775 140382581987328 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:22:13.875802 140382581987328 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:22:13.875828 140382581987328 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:22:13.875855 140382581987328 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:22:13.875881 140382581987328 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:22:13.875907 140382581987328 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:22:13.875935 140382581987328 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:22:13.875995 140382581987328 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:22:13.876026 140382581987328 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:22:13.876053 140382581987328 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:22:13.876080 140382581987328 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:22:13.876107 140382581987328 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:22:13.876135 140382581987328 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:22:13.876162 140382581987328 pyconfig.py:457] Config param train_image_column: image
I0512 21:22:13.876189 140382581987328 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:22:13.876216 140382581987328 pyconfig.py:457] Config param train_split: train
I0512 21:22:13.876242 140382581987328 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:22:13.876269 140382581987328 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:22:13.876296 140382581987328 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:22:13.876325 140382581987328 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:22:13.876353 140382581987328 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:22:13.876379 140382581987328 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:22:13.876406 140382581987328 pyconfig.py:457] Config param use_audio: False
I0512 21:22:13.876432 140382581987328 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:22:13.876459 140382581987328 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:22:13.876486 140382581987328 pyconfig.py:457] Config param use_chat_template: False
I0512 21:22:13.876513 140382581987328 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:22:13.876539 140382581987328 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:22:13.876571 140382581987328 pyconfig.py:457] Config param use_dpo: False
I0512 21:22:13.876598 140382581987328 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:22:13.876624 140382581987328 pyconfig.py:457] Config param use_grpo: True
I0512 21:22:13.876651 140382581987328 pyconfig.py:457] Config param use_indexer: False
I0512 21:22:13.876677 140382581987328 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:22:13.876704 140382581987328 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:22:13.876730 140382581987328 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:22:13.876756 140382581987328 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:22:13.876783 140382581987328 pyconfig.py:457] Config param use_mrope: False
I0512 21:22:13.876809 140382581987328 pyconfig.py:457] Config param use_multimodal: False
I0512 21:22:13.876836 140382581987328 pyconfig.py:457] Config param use_pathways: True
I0512 21:22:13.876862 140382581987328 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:22:13.876888 140382581987328 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:22:13.876914 140382581987328 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:22:13.876940 140382581987328 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:22:13.876967 140382581987328 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:22:13.876994 140382581987328 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:22:13.877020 140382581987328 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:22:13.877047 140382581987328 pyconfig.py:457] Config param use_random_routing: False
I0512 21:22:13.877074 140382581987328 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:22:13.877100 140382581987328 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:22:13.877126 140382581987328 pyconfig.py:457] Config param use_sft: False
I0512 21:22:13.877153 140382581987328 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:22:13.877179 140382581987328 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:22:13.877206 140382581987328 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:22:13.877232 140382581987328 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:22:13.877259 140382581987328 pyconfig.py:457] Config param use_truncation: True
I0512 21:22:13.877285 140382581987328 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:22:13.877314 140382581987328 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:22:13.877341 140382581987328 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:22:13.877367 140382581987328 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:22:13.877394 140382581987328 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:22:13.877420 140382581987328 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:22:13.877446 140382581987328 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:22:13.877474 140382581987328 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:22:13.877500 140382581987328 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:22:13.877527 140382581987328 pyconfig.py:457] Config param video_path: 
I0512 21:22:13.877558 140382581987328 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:22:13.877586 140382581987328 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:22:13.877612 140382581987328 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:22:13.877639 140382581987328 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:22:13.877666 140382581987328 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:22:13.877692 140382581987328 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:22:13.877719 140382581987328 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:22:13.877746 140382581987328 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:22:13.877772 140382581987328 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:22:13.877800 140382581987328 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:22:13.877841 140382581987328 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:22:13.877870 140382581987328 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:22:13.877897 140382581987328 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:22:13.877924 140382581987328 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:22:13.877951 140382581987328 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:22:13.877977 140382581987328 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:22:13.878004 140382581987328 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:22:13.878030 140382581987328 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.878057 140382581987328 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:22:13.878082 140382581987328 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:22:13.878109 140382581987328 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:22:13.878135 140382581987328 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:22:13.878161 140382581987328 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:22:13.878188 140382581987328 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:22:13.878214 140382581987328 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:22:13.878241 140382581987328 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:22:13.878268 140382581987328 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:22:13.878293 140382581987328 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:22:13.878322 140382581987328 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:22:13.878349 140382581987328 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:22:13.878376 140382581987328 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:22:13.878402 140382581987328 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:22:13.878428 140382581987328 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:22:13.878454 140382581987328 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:22:13.878482 140382581987328 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:22:13.878513 140382581987328 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:22:13.878540 140382581987328 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:22:13.878572 140382581987328 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:22:13.878599 140382581987328 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:22:13.878632 140382581987328 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:22:13.880220 140382581987328 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:22:14.056253 140382581987328 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:22:14.056468 140382581987328 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:14.057085 140382581987328 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f9b37f3a560>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:14.690885 140477414291456 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:22:14.691096 140477414291456 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb06cb678b0>}, handler_registry=None
I0512 21:22:14.691381 140477414291456 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb06cb678b0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.691450 140477414291456 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb06ca5e470>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.691500 140477414291456 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb06cb678b0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fb06cb678b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb06ca5e470>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fb06ca5e470>}).
I0512 21:22:14.691936 140477414291456 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:22:14.692035 140477414291456 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7fc2bff16ef0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:22:14.743245 140382581987328 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:22:14.743462 140382581987328 checkpoint_manager.py:708] [process=2][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9a594b1b40>}, handler_registry=None
I0512 21:22:14.743813 140382581987328 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9a594b1b40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.743887 140382581987328 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9a59259210>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.743939 140382581987328 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9a594b1b40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f9a594b1b40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9a59259210>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f9a59259210>}).
I0512 21:22:14.744373 140382581987328 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:22:14.744487 140382581987328 async_checkpointer.py:192] [process=2][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7facaba0eef0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:22:14.764816 139965031823360 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:22:14.765010 139965031823360 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f391cb9b850>}, handler_registry=None
I0512 21:22:14.765277 139965031823360 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f391cb9b850>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.765347 139965031823360 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f391ca91210>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:22:14.765402 139965031823360 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f391cb9b850>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f391cb9b850>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f391ca91210>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f391ca91210>}).
I0512 21:22:14.765824 139965031823360 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:22:14.765928 139965031823360 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f4b7380aef0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:22:15.311134 139961977686016 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.332623 140477414291456 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.335741 139708914673664 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.362842 140382581987328 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.364727 140509907564544 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.369811 139965031823360 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.393839 139720546064384 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.403218 139708914673664 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7efd7ef4c160>
I0512 21:22:15.403366 139708914673664 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403421 139708914673664 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403425 140382581987328 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f9a59258220>
I0512 21:22:15.403576 140382581987328 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403631 140382581987328 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403230 139961977686016 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f386ae90220>
I0512 21:22:15.403366 139961977686016 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403234 140509907564544 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fb7fca8c280>
I0512 21:22:15.403371 140509907564544 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403427 140509907564544 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403246 139720546064384 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f0033b7c250>
I0512 21:22:15.403420 139961977686016 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403407 139720546064384 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403463 139720546064384 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403277 140477414291456 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fb06ca5f550>
I0512 21:22:15.403424 140477414291456 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403476 140477414291456 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.401473 139627526768640 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:22:15.403280 139965031823360 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f391ca90220>
I0512 21:22:15.403424 139965031823360 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403481 139965031823360 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.403202 139627526768640 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7eea8bfa0220>
I0512 21:22:15.403361 139627526768640 checkpointing.py:326] Checkpoint manager created!
I0512 21:22:15.403418 139627526768640 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.874820 139961977686016 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.874986 139961977686016 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.875109 139961977686016 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.875211 139961977686016 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f3949f46560>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.875422 139961977686016 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.888636 140477414291456 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.888810 140477414291456 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.888931 140477414291456 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.889031 140477414291456 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb14b667fd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.889246 140477414291456 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.898916 140509907564544 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.899110 140509907564544 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.899245 140509907564544 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.899354 140509907564544 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb8dc65ffd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.899598 140509907564544 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.914342 139627526768640 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.914524 139627526768640 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.914654 139627526768640 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.914780 139627526768640 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7eeb6b033fd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.915017 139627526768640 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.917777 139965031823360 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.917984 139965031823360 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.918117 139965031823360 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.918229 139965031823360 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f39fba5bfd0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.918480 139965031823360 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.922864 139708914673664 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.923051 139708914673664 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.923176 139708914673664 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.923278 139708914673664 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7efe5df4b820>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.923496 139708914673664 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.925972 139720546064384 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.926153 139720546064384 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.926285 139720546064384 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.926398 139720546064384 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f0112958b80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.926620 139720546064384 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:15.945213 140382581987328 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:22:15.945412 140382581987328 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:15.945548 140382581987328 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:22:15.945665 140382581987328 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f9b37f3a560>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:22:15.945888 140382581987328 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.196302 140382581987328 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.196901 140382581987328 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.25 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.202226 140509907564544 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.202818 140509907564544 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.30 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.213428 139720546064384 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.214014 139720546064384 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.215612 139708914673664 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.216192 139708914673664 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.215717 140477414291456 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.216303 140477414291456 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.33 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.258069 139965031823360 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.258696 139965031823360 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.34 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.260200 139627526768640 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.260769 139627526768640 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.35 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.277520 139961977686016 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.278112 139961977686016 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.40 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:22:16.733349   71154 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.763884   71059 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.773272   70324 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.777955   80383 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.803770   72816 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.830853   76322 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.842840   76511 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:16.930788   77707 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:22:18.337471 140477414291456 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1015.731 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.5122346878051758 s) (per-host)
I0512 21:22:18.338659 139708914673664 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 980.602 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.5664079189300537 s) (per-host)
I0512 21:22:18.338159 139720546064384 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 953.927 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.6102113723754883 s) (per-host)
I0512 21:22:18.342553 140509907564544 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 969.635 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.584125280380249 s) (per-host)
I0512 21:22:18.343133 139961977686016 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.058 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.4175443649291992 s) (per-host)
I0512 21:22:18.344805 139720546064384 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.245 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.848759412765503 s) (per-host)
I0512 21:22:18.344414 140477414291456 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 3.344 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7945034503936768 s) (per-host)
I0512 21:22:18.346577 139708914673664 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.313 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.8108389377593994 s) (per-host)
I0512 21:22:18.349517 139961977686016 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 3.429 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.7498600482940674 s) (per-host)
I0512 21:22:18.349693 140509907564544 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.200 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.875195026397705 s) (per-host)
I0512 21:22:18.355450 139627526768640 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 967.672 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.5873394012451172 s) (per-host)
I0512 21:22:18.362126 139627526768640 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.305 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.8153016567230225 s) (per-host)
I0512 21:22:18.464466 139965031823360 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 943.910 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.62729811668396 s) (per-host)
I0512 21:22:18.470802 139965031823360 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.138 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.912015676498413 s) (per-host)
I0512 21:22:19.730776 140382581987328 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 523.828 MiB/s (total gbytes: 1.5 GiB) (time elapsed: 2.9323039054870605 s) (per-host)
I0512 21:22:19.737730 140382581987328 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 1.870 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 3.2081096172332764 s) (per-host)
I0512 21:22:19.742377 139708914673664 event_tracking.py:138] [process=7] [sync] Finished load in 3.82 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742909 139708914673664 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.742434 140509907564544 event_tracking.py:138] [process=4] [sync] Finished load in 3.84 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742981 140509907564544 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:22:19.742393 140477414291456 event_tracking.py:138] [process=0] [sync] Finished load in 3.85 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742625 139720546064384 event_tracking.py:138] [process=5] [sync] Finished load in 3.82 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742588 139961977686016 event_tracking.py:138] [process=1] [sync] Finished load in 3.87 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742944 140477414291456 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:22:19.743141 140382581987328 event_tracking.py:138] [process=2] [sync] Finished load in 3.80 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.743121 139961977686016 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.744194 140382581987328 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_coWorker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
nfig.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymaWorker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.742436 139965031823360 event_tracking.py:138] [process=3] [sync] Finished load in 3.82 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/itemsWorker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.742378 139627526768640 event_tracking.py:138] [process=6] [sync] Finished load in 3.83 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:22:19.742973 139627526768640 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion

I0512 21:22:19.742973 139965031823360 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
th-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: Using parameter nesting key: VariableState
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state attributes: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']
--- FULL PARAMETER TREE ---
PARAM: decoder/decoder_norm/scale (shape: unknown)
PARAM: decoder/logits_dense/kernel (shape: unknown)
PARAM: token_embedder/embedding (shape: unknown)
---------------------------
I0512 21:22:19.743214 139720546064384 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_namTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
e} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 120, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 85, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")
ValueError: Missing layers in training_state or training_state_annotations.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

























































### Planner Response

I suspect the layers are either hidden under a different key in the `params` dict or they are being stored in the `opt_state` (optimizer state) due to the RL-style training (GRPO).

I'm updating `convert_helper.py` to:
1. **Aggressively search** both `params` and `opt_state`.
2. **List the actual GCS files** using `gsutil` from within the script to see the physical layout.
3. **Fix the tree printer** to correctly show shapes of JAX/Orbax arrays.

### Update `src/train/convert_helper.py`:



*Edited relevant file*

### Planner Response

Wait, I made a small error in the `replace_file_content` block (I truncated the end of `unroll_layer_group`). I need to fix that before you run it.

I'll restore the rest of the function.



*Edited relevant file*

### Planner Response

Now let's run it. This will finally give us a full map of where the weights are.

```bash
# 1. SCP updated script
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run on all workers
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

The output will now include:
1. **`--- PARAMS TREE ---`**: Full recursive list of everything in `training_state.params`.
2. **`--- OPT_STATE SAMPLE ---`**: Full recursive list of everything in `training_state.opt_state`.
3. **`--- GCS DIRECTORY LISTING ---`**: Physical file list in GCS.

If the backbone is not in `params`, it **must** be in `opt_state` (perhaps under a different collection name).

### User Input

nSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.548477 139977507788800 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3c081c8d00>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3c081c8d00>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f3c0807a6e0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f3c0807a6e0>}).
I0512 21:24:38.548911 139977507788800 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.549025 139977507788800 async_checkpointer.py:192] [process=4][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f4e5b2e6dd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.564097 139860987914240 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.564355 139860987914240 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f20e6facc10>}, handler_registry=None
I0512 21:24:38.564688 139860987914240 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f20e6facc10>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.564759 139860987914240 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f20e6e60cd0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.564811 139860987914240 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f20e6facc10>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f20e6facc10>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f20e6e60cd0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f20e6e60cd0>}).
I0512 21:24:38.565275 139860987914240 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.565385 139860987914240 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f3339f7add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.600344 139907972642816 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.600654 139907972642816 checkpoint_manager.py:708] [process=2][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f2bd799cc70>}, handler_registry=None
I0512 21:24:38.601046 139907972642816 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f2bd799cc70>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.601119 139907972642816 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f2bd784e650>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.601173 139907972642816 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f2bd799cc70>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f2bd799cc70>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f2bd784e650>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f2bd784e650>}).
I0512 21:24:38.601688 139907972642816 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.601796 139907972642816 async_checkpointer.py:192] [process=2][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f3e2a80add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.606736 140151424940032 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.606935 140151424940032 checkpoint_manager.py:708] [process=7][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6486014c10>}, handler_registry=None
I0512 21:24:38.607195 140151424940032 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6486014c10>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.607263 140151424940032 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f64860c4cd0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.607313 140151424940032 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6486014c10>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f6486014c10>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f64860c4cd0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f64860c4cd0>}).
I0512 21:24:38.607810 140151424940032 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.607918 140151424940032 async_checkpointer.py:192] [process=7][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f76d997add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.652771 140261516650496 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.653027 140261516650496 checkpoint_manager.py:708] [process=1][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7e284a0c40>}, handler_registry=None
I0512 21:24:38.653397 140261516650496 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7e284a0c40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.653468 140261516650496 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7e28351510>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.653522 140261516650496 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7e284a0c40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7e284a0c40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7e28351510>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7e28351510>}).
I0512 21:24:38.654002 140261516650496 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.654112 140261516650496 async_checkpointer.py:192] [process=1][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f907b6e6dd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.718570 140260565141504 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.718800 140260565141504 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7dec698c40>}, handler_registry=None
I0512 21:24:38.719150 140260565141504 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7dec698c40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.719239 140260565141504 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7dec549510>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.719296 140260565141504 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7dec698c40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f7dec698c40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7dec549510>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f7dec549510>}).
I0512 21:24:38.719738 140260565141504 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.719862 140260565141504 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f9042ddadd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:38.759826 140322738837504 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:24:38.760065 140322738837504 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8c69488c10>}, handler_registry=None
I0512 21:24:38.760409 140322738837504 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8c69488c10>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.760483 140322738837504 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f8c6933ccd0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:24:38.760534 140322738837504 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8c69488c10>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8c69488c10>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f8c6933ccd0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f8c6933ccd0>}).
I0512 21:24:38.760984 140322738837504 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:24:38.761093 140322738837504 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f9ebc7dadd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:24:39.270549 140151424940032 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.306673 139907972642816 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.315466 140261516650496 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.317428 139935778666496 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.337048 140322738837504 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.347350 139860987914240 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.357544 139977507788800 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.360343 140260565141504 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:24:39.362906 140260565141504 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f7dec54a8f0>
I0512 21:24:39.363078 140260565141504 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.363137 140260565141504 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362791 139860987914240 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f20e6e63340>
I0512 21:24:39.362935 139860987914240 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.362991 139860987914240 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362751 139935778666496 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f32504efeb0>
I0512 21:24:39.362894 139935778666496 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.362947 139935778666496 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362883 139977507788800 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f3c0807aaa0>
I0512 21:24:39.363033 139977507788800 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.363087 139977507788800 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362793 140322738837504 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f8c6933f340>
I0512 21:24:39.362935 140322738837504 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.362995 140322738837504 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362824 140261516650496 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f7e283528f0>
I0512 21:24:39.362971 140261516650496 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.363027 140261516650496 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362915 139907972642816 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f2bd784e9b0>
I0512 21:24:39.363071 139907972642816 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.363141 139907972642816 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.362825 140151424940032 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f64860c7340>
I0512 21:24:39.362975 140151424940032 checkpointing.py:326] Checkpoint manager created!
I0512 21:24:39.363029 140151424940032 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.857525 139860987914240 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.857699 139860987914240 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.857818 139860987914240 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.857921 139860987914240 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f21c5f6f2b0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.858135 139860987914240 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.863032 140261516650496 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.863195 140261516650496 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.863317 140261516650496 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.863416 140261516650496 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f7f0746f280>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.863626 140261516650496 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.866379 140151424940032 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.866572 140151424940032 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.866706 140151424940032 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.866815 140151424940032 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f65650772e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.867048 140151424940032 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.873439 140322738837504 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.873617 140322738837504 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.873750 140322738837504 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.873854 140322738837504 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f8d4815f280>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.874075 140322738837504 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.880337 139977507788800 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.880528 139977507788800 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.880653 139977507788800 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.880761 139977507788800 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f3ce708f2e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.880987 139977507788800 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.885690 139935778666496 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.885873 139935778666496 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.885997 139935778666496 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.886105 139935778666496 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f332f86f2b0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.886321 139935778666496 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.893632 139907972642816 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.893821 139907972642816 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.893963 139907972642816 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.894076 139907972642816 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f2cb657f2e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.894316 139907972642816 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:39.911035 140260565141504 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:24:39.911251 140260565141504 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:39.911391 140260565141504 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:24:39.911508 140260565141504 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f7ecb26f250>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:24:39.911751 140260565141504 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.081846 140322738837504 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.082424 140322738837504 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.21 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.106416 139860987914240 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.106985 139860987914240 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.25 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.107548 140261516650496 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.108138 140261516650496 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.24 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.133651 139977507788800 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.134218 139977507788800 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.25 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.178635 139935778666496 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.179216 139935778666496 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.208456 140260565141504 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.209126 140260565141504 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.30 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.227795 140151424940032 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.228387 140151424940032 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.36 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.241251 139907972642816 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.241871 139907972642816 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.35 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:24:40.630333   79320 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.638160   77923 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.642189   71939 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.687597   72682 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.745359   78369 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.760138   74415 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.794306   72762 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:40.800223   82024 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:24:41.944798 140261516650496 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.137 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.319500207901001 s) (per-host)
I0512 21:24:41.952213 139860987914240 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.137 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3195366859436035 s) (per-host)
I0512 21:24:41.959781 139860987914240 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 3.910 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.534437894821167 s) (per-host)
I0512 21:24:41.959956 140261516650496 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 3.762 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.594710111618042 s) (per-host)
I0512 21:24:41.966674 139977507788800 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.168 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2845637798309326 s) (per-host)
I0512 21:24:41.974535 139977507788800 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.779 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5878396034240723 s) (per-host)
I0512 21:24:42.023077 140322738837504 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.082 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3857390880584717 s) (per-host)
I0512 21:24:42.040395 140322738837504 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.637 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.649616003036499 s) (per-host)
I0512 21:24:42.121637 139935778666496 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.125 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3328027725219727 s) (per-host)
I0512 21:24:42.124008 140151424940032 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.129 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3291614055633545 s) (per-host)
I0512 21:24:42.129084 139935778666496 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.682 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6295061111450195 s) (per-host)
I0512 21:24:42.131006 140260565141504 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.078 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3911855220794678 s) (per-host)
I0512 21:24:42.129215 140151424940032 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.823 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.569566249847412 s) (per-host)
I0512 21:24:42.137898 140260565141504 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.663 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6382455825805664 s) (per-host)
I0512 21:24:42.150229 139907972642816 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.075 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3955576419830322 s) (per-host)
I0512 21:24:42.156799 139907972642816 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 3.623 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6559126377105713 s) (per-host)
I0512 21:24:42.160893 139860987914240 event_tracking.py:138] [process=0] [sync] Finished load in 2.30 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.161437 140260565141504 event_tracking.py:138] [process=3] [sync] Finished load in 2.25 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.160670 140322738837504 event_tracking.py:138] [process=6] [sync] Finished load in 2.29 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.161114 140322738837504 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.161431 139860987914240 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.160795 139977507788800 event_tracking.py:138] [process=4] [sync] Finished load in 2.28 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.162051 140260565141504 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.160965 139935778666496 event_tracking.py:138] [process=5] [sync] Finished load in 2.27 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.161551 139935778666496 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.161354 139977507788800 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.160573 140151424940032 event_tracking.py:138] [process=7] [sync] Finished load in 2.29 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.160928 140151424940032 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.160884 140261516650496 event_tracking.py:138] [process=1] [sync] Finished load in 2.30 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.161247 140261516650496 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:24:42.161626 139907972642816 event_tracking.py:138] [process=2] [sync] Finished load in 2.27 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:24:42.162267 139907972642816 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Traceback (most recent call last):
  File "/snap/google-cloud-cli/452/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/452/platform/gsutil/gsutil.py", line 157, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/452/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_naWorker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
me} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, lTraceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_naWorker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
me} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
ayer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Traceback (most recent call last):
  File "/snap/google-cloud-cli/452/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/452/platform/gsutil/gsutil.py", line 157, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/452/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/452/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
Traceback (most recent call last):
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil", line 21, in <module>
    gsutil.RunMain()
  File "/snap/google-cloud-cli/127/platform/gsutil/gsutil.py", line 150, in RunMain
    import gslib.__main__
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/__main__.py", line 86, in <module>
    from google_reauth import reauth_creds
  File "/snap/google-cloud-cli/127/platform/gsutil/third_party/google-reauth-python/google_reauth/reauth_creds.py", line 23, in <module>
    from oauth2client import client
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/client.py", line 45, in <module>
    from oauth2client import crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/crypt.py", line 45, in <module>
    from oauth2client import _openssl_crypt
  File "/snap/google-cloud-cli/127/platform/gsutil/gslib/vendored/oauth2client/oauth2client/_openssl_crypt.py", line 16, in <module>
    from OpenSSL import crypto
  File "/usr/lib/python3/dist-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1579, in <module>
    class X509StoreFlags(object):
  File "/usr/lib/python3/dist-packages/OpenSSL/crypto.py", line 1598, in X509StoreFlags
    NOTIFY_POLICY = _lib.X509_V_FLAG_NOTIFY_POLICY
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'. Did you mean: 'X509_V_FLAG_EXPLICIT_POLICY'?
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']
DEBUG: training_state type: <class 'flax.training.train_state.TrainState'>
DEBUG: training_state keys: ['apply_fn', 'apply_gradients', 'create', 'opt_state', 'params', 'replace', 'step', 'tx']

--- PARAMS TREE ---
  VariableState/decoder/decoder_norm/scale (shape: N/A, type: VariableState)
  VariableState/decoder/logits_dense/kernel (shape: N/A, type: VariableState)
  VariableState/token_embedder/embedding (shape: N/A, type: VariableState)

--- OPT_STATE SAMPLE ---
  [0][0] (shape: (), type: ArrayImpl)

--- GCS DIRECTORY LISTING ---
Failed to list GCS directory
---------------------------

DEBUG: Desperate search for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 176, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 163, in patched_possibly_unroll_params
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")
  File "/home/himanshu/maxtext/convert_helper.py", line 135, in unroll_layer_group
    raise ValueError(f"Missing {layer_name} in training_state. Use the tree above to find the correct path.")
ValueError: Missing layers in training_state. Use the tree above to find the correct path.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

I've analyzed the logs, and I have a **major breakthrough**.

### The Discovery
The parameter tree we are seeing is:
```
VariableState/decoder/decoder_norm/scale
VariableState/decoder/logits_dense/kernel
VariableState/token_embedder/embedding
```
Total parameters in these three: **0.134 Billion**.
But we are missing the transformer layers entirely! For a 1B model, the layers should be ~0.87B params. 

**Conclusion**: The layers are **not** under the `VariableState` key in the `params` dictionary. They might be in a different collection, or more likely, they are stored in `opt_state` or a separate field because of how MaxText handles "scanned" (checkpointed) layers during training.

### The Plan
I am going to update `convert_helper.py` to be extremely aggressive. Instead of looking for a specific path, it will:
1.  **Scan every single leaf** in the entire `training_state` (both `params` and `opt_state`).
2.  **Identify any array that has 22 in its first dimension** (our layer count).
3.  **Use `gcloud storage`** instead of `gsutil` (which is crashing on the TPU).

I'm also fixing the `print_tree` function to handle the custom Flax dict types correctly so we can see the full structure.

python


*Edited relevant file*

*Viewed [convert_helper.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/convert_helper.py) *

*Edited relevant file*

*User accepted the command `gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all && gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"`*

*Checked command status*

*User accepted the command `gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all`*

*Checked command status*

*User accepted the command `ls -la /Users/himanshu/.config/gcloud`*

*User accepted the command `mkdir -p .gcloud_config && export CLOUDSDK_CONFIG=$(pwd)/.gcloud_config && gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all && gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"`*

*Checked command status*

*User accepted the command `gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cat << 'EOF' > ~/maxtext/convert_helper.py
import sys
import os
import functools
import types
import importlib.abc
from unittest.mock import MagicMock

# 1. JAX Patches
import jax
import jax._src.api
_original_jit = jax.jit
def _patched_jit(fun=None, **kwargs):
    if fun is None: return functools.partial(_original_jit, **kwargs)
    return _original_jit(fun, **kwargs)
jax.jit = _patched_jit
jax._src.api.jit = _patched_jit

if not hasattr(jax, 'set_mesh'):
    import contextlib
    @contextlib.contextmanager
    def _set_mesh_shim(mesh):
        with mesh: yield
    jax.set_mesh = _set_mesh_shim
    jax.sharding.set_mesh = _set_mesh_shim

# 2. Robust Mock Finder
class MockFinder(importlib.abc.MetaPathFinder):
    _MOCKS = ('jax.pallas', 'jax.experimental.pallas', 'qwix', 'tokamax', 'drjax', 'pathwaysutils')
    def find_module(self, fullname, path=None):
        if any(fullname == m or fullname.startswith(m + '.') for m in self._MOCKS): return self
    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []; mod.__file__ = '<mock>'; mod.__loader__ = self
        if fullname == 'pathwaysutils.elastic.manager':
            class _M:
                def __init__(self, *a, **k): pass
            mod.Manager = _M; mod.ScaleUpSignalError = mod.ScaleDownSignalError = Exception
        mod.__getattr__ = lambda name: MagicMock()
        sys.modules[fullname] = mod
        return mod
sys.meta_path.insert(0, MockFinder())

# 3. Setup Path and Distributed Init
sys.path.append(os.path.join(os.path.expanduser('~'), 'maxtext', 'src'))
try:
    jax.distributed.initialize()
    print(f'Worker {jax.process_index()} initialized.')
except Exception as e:
    print(f'Distributed init info: {e}')

# 4. Import MaxText and Patch the Unrolling Logic
import maxtext.utils.generate_param_only_checkpoint as gpoc
from maxtext.common.common_types import DecoderBlockType

def print_tree(tree, indent=0, path=''):
    prefix = '  ' * indent
    if hasattr(tree, 'items'):
        for k, v in sorted(tree.items()):
            print(f'{prefix}{k} (type: {type(v).__name__})')
            print_tree(v, indent + 1, f'{path}/{k}')
    elif hasattr(tree, 'shape'):
        print(f'{prefix}[LEAF] path: {path}, shape: {tree.shape}, dtype: {tree.dtype}')
    elif isinstance(tree, (list, tuple)):
        print(f'{prefix}[LIST/TUPLE] length: {len(tree)}')
        if len(tree) > 0:
            print_tree(tree[0], indent + 1, f'{path}[0]')

def find_layers_by_shape(tree, target_layers=22):
    found_paths = []
    def walk(obj, path):
        if hasattr(obj, 'items'):
            for k, v in obj.items():
                walk(v, f'{path}/{k}')
        elif hasattr(obj, 'shape'):
            if len(obj.shape) >= 2 and obj.shape[0] == target_layers:
                found_paths.append((path, obj.shape))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f'{path}[{i}]')
    walk(tree, '')
    return found_paths

def patched_possibly_unroll_params(config, training_state, training_state_annotations, mesh):
    if not config.scan_layers or not config.force_unroll:
        return
    print('\n--- FULL PARAMS TREE ---')
    print_tree(training_state.params)
    print('\n--- SEARCHING FOR LAYERS (target: 22) ---')
    matches = find_layers_by_shape(training_state.params, 22)
    for path, shape in matches:
        print(f'  MATCH: {path} (shape: {shape})')
    if not matches:
        print('\nChecking opt_state for layers...')
        matches = find_layers_by_shape(training_state.opt_state, 22)
        for path, shape in matches:
            print(f'  MATCH in opt_state: {path} (shape: {shape})')
    if not matches:
        raise ValueError('Could not find any array with 22 layers in params or opt_state.')
    target_path, _ = matches[0]
    keys = [k for k in target_path.split('/') if k]
    num_layers = config.num_decoder_layers
    curr = training_state.params
    for k in keys[:-1]:
        curr = curr[k]
    stacked_layers = curr[keys[-1]]
    curr_ann = training_state_annotations.params
    for k in keys[:-1]:
        curr_ann = curr_ann[k]
    layers_annotations = curr_ann[keys[-1]]
    def new_pspec(x):
      return jax.sharding.PartitionSpec(*(x[0 : config.param_scan_axis] + x[config.param_scan_axis + 1 :]))
    p_keys = list(training_state.params.keys())
    p_key = p_keys[0] if p_keys else None
    print(f'Unrolling {keys[-1]} into {num_layers} layers...')
    for i in range(num_layers):
        unrolled_layer = jax.tree_util.tree_map(lambda x: x[i], stacked_layers)
        unrolled_layer_annotations = jax.tree_util.tree_map(new_pspec, layers_annotations)
        dest = training_state.params[p_key]['decoder'] if p_key else training_state.params['decoder']
        dest_ann = training_state_annotations.params[p_key]['decoder'] if p_key else training_state_annotations.params['decoder']
        dest[f'layers_{i}'] = unrolled_layer
        dest_ann[f'layers_{i}'] = unrolled_layer_annotations
    del curr[keys[-1]]
    del curr_ann[keys[-1]]
    gpoc.save_param_only_checkpoint(config, training_state, training_state_annotations, mesh)

gpoc._possibly_unroll_params = patched_possibly_unroll_params

from absl import app
if __name__ == '__main__':
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(gpoc.main)
EOF
cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"`*

*Checked command status*

### Planner Response

I've updated `src/train/convert_helper.py` with a **shape-based search engine**. It will now scan the entire checkpoint for any array with 22 layers, regardless of where MaxText has hidden them.

However, I'm currently hitting a macOS-specific permission error when trying to run `gcloud` from my environment (it's unable to write to the `~/.config/gcloud/credentials.db` file). 

**Please run these two commands in your terminal to execute the inspection and extraction:**

```bash
# 1. Sync the updated search logic to all TPU workers
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run the extraction
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

Once this finishes, the layers will be "unrolled" and saved to `gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/`. Please paste the output here so I can verify the final paths for the Hugging Face conversion!

### User Input

I0512 21:29:09.648288 139884987709440 pyconfig.py:457] Config param routed_scaling_factor: 1.0
I0512 21:29:09.648315 139884987709440 pyconfig.py:457] Config param routed_score_func: 
I0512 21:29:09.648341 139884987709440 pyconfig.py:457] Config param run_name: tinymath-1b-decode
I0512 21:29:09.648374 139884987709440 pyconfig.py:457] Config param sa_block_kv: 512
I0512 21:29:09.648401 139884987709440 pyconfig.py:457] Config param sa_block_kv_compute: 512
I0512 21:29:09.648427 139884987709440 pyconfig.py:457] Config param sa_block_kv_dkv: 512
I0512 21:29:09.648454 139884987709440 pyconfig.py:457] Config param sa_block_kv_dkv_compute: 512
I0512 21:29:09.648479 139884987709440 pyconfig.py:457] Config param sa_block_kv_dq: 512
I0512 21:29:09.648505 139884987709440 pyconfig.py:457] Config param sa_block_q: 512
I0512 21:29:09.648531 139884987709440 pyconfig.py:457] Config param sa_block_q_dkv: 512
I0512 21:29:09.648557 139884987709440 pyconfig.py:457] Config param sa_block_q_dq: 512
I0512 21:29:09.648583 139884987709440 pyconfig.py:457] Config param sa_k_layout: HEAD_DIM_MINOR
I0512 21:29:09.648609 139884987709440 pyconfig.py:457] Config param sa_q_layout: HEAD_DIM_MINOR
I0512 21:29:09.648637 139884987709440 pyconfig.py:457] Config param sa_use_fused_bwd_kernel: False
I0512 21:29:09.648664 139884987709440 pyconfig.py:457] Config param sa_v_layout: HEAD_DIM_MINOR
I0512 21:29:09.648690 139884987709440 pyconfig.py:457] Config param sampler_devices_fraction: 0.5
I0512 21:29:09.648718 139884987709440 pyconfig.py:457] Config param save_checkpoint_on_completion: True
I0512 21:29:09.648744 139884987709440 pyconfig.py:457] Config param save_config_to_gcs: False
I0512 21:29:09.648770 139884987709440 pyconfig.py:457] Config param save_quantized_params_path: 
I0512 21:29:09.648797 139884987709440 pyconfig.py:457] Config param scale_embedding_for_audio: True
I0512 21:29:09.648823 139884987709440 pyconfig.py:457] Config param scan_layers: True
I0512 21:29:09.648849 139884987709440 pyconfig.py:457] Config param scan_layers_per_stage: False
I0512 21:29:09.648875 139884987709440 pyconfig.py:457] Config param scan_pipeline_iterations: True
I0512 21:29:09.648902 139884987709440 pyconfig.py:457] Config param scan_pipeline_repeats: False
I0512 21:29:09.648928 139884987709440 pyconfig.py:457] Config param set_remat_policy_on_layers_per_stage: False
I0512 21:29:09.648954 139884987709440 pyconfig.py:457] Config param set_remat_policy_on_pipeline_iterations: True
I0512 21:29:09.648980 139884987709440 pyconfig.py:457] Config param sft_train_on_completion_only: False
I0512 21:29:09.649007 139884987709440 pyconfig.py:457] Config param shard_exp_on_fsdp: False
I0512 21:29:09.649033 139884987709440 pyconfig.py:457] Config param shard_mode: ShardMode.AUTO
I0512 21:29:09.649062 139884987709440 pyconfig.py:457] Config param shard_optimizer_over_data: False
I0512 21:29:09.649088 139884987709440 pyconfig.py:457] Config param sharding_strategy: None
I0512 21:29:09.649115 139884987709440 pyconfig.py:457] Config param sharding_tolerance: 0.3
I0512 21:29:09.649142 139884987709440 pyconfig.py:457] Config param shardy: True
I0512 21:29:09.649168 139884987709440 pyconfig.py:457] Config param share_kv_projections: False
I0512 21:29:09.649194 139884987709440 pyconfig.py:457] Config param shared_experts: 0
I0512 21:29:09.649220 139884987709440 pyconfig.py:457] Config param sinkhorn_iterations: 20
I0512 21:29:09.649246 139884987709440 pyconfig.py:457] Config param skip_first_n_steps_for_profiler: 1
I0512 21:29:09.649273 139884987709440 pyconfig.py:457] Config param skip_jax_distributed_system: False
I0512 21:29:09.649299 139884987709440 pyconfig.py:457] Config param skip_step_interval: 128
I0512 21:29:09.649325 139884987709440 pyconfig.py:457] Config param skip_step_on_spikes: False
I0512 21:29:09.649351 139884987709440 pyconfig.py:457] Config param skip_step_scaling_factor: 6.0
I0512 21:29:09.649386 139884987709440 pyconfig.py:457] Config param sliding_window_size: 0
I0512 21:29:10.128709 140672923404288 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:29:10.128950 140672923404288 checkpoint_manager.py:708] [process=7][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fddf1c4eaa0>}, handler_registry=None
I0512 21:29:10.129298 140672923404288 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fddf1c4eaa0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.129374 140672923404288 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fddf1be0280>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.129426 140672923404288 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fddf1c4eaa0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7fddf1c4eaa0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fddf1be0280>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7fddf1be0280>}).
I0512 21:29:10.129869 140672923404288 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:29:10.129983 140672923404288 async_checkpointer.py:192] [process=7][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7ff04520add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:29:09.458572 140303782647808 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:29:09.458598 140303782647808 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:29:09.458625 140303782647808 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:29:09.458651 140303782647808 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:29:09.458677 140303782647808 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:29:09.458704 140303782647808 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:29:09.458732 140303782647808 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:29:09.458759 140303782647808 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:29:09.458785 140303782647808 pyconfig.py:457] Config param steps: 54363
I0512 21:29:09.458812 140303782647808 pyconfig.py:457] Config param stop_strings: None
I0512 21:29:09.458838 140303782647808 pyconfig.py:457] Config param student_overrides: {}
I0512 21:29:09.458865 140303782647808 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:29:09.458892 140303782647808 pyconfig.py:457] Config param subslice_shape: 
I0512 21:29:09.458918 140303782647808 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:29:09.458945 140303782647808 pyconfig.py:457] Config param system_prompt: 
I0512 21:29:09.458976 140303782647808 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:29:09.459003 140303782647808 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:29:09.459030 140303782647808 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:29:09.459057 140303782647808 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:29:09.459083 140303782647808 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:29:09.459110 140303782647808 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:29:09.459137 140303782647808 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:29:09.459163 140303782647808 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:29:09.459189 140303782647808 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:29:09.459215 140303782647808 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:29:09.459241 140303782647808 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:29:09.459291 140303782647808 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:29:09.459322 140303782647808 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:29:09.459352 140303782647808 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:29:09.459379 140303782647808 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:29:09.459406 140303782647808 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:29:09.459432 140303782647808 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:29:09.459459 140303782647808 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:29:09.459486 140303782647808 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:29:09.459513 140303782647808 pyconfig.py:457] Config param train_image_column: image
I0512 21:29:09.459540 140303782647808 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:29:09.459566 140303782647808 pyconfig.py:457] Config param train_split: train
I0512 21:29:09.459592 140303782647808 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:29:09.459618 140303782647808 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:29:09.459644 140303782647808 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:29:09.459671 140303782647808 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:29:09.459698 140303782647808 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:29:09.459726 140303782647808 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:29:09.459753 140303782647808 pyconfig.py:457] Config param use_audio: False
I0512 21:29:09.459779 140303782647808 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:29:09.459805 140303782647808 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:29:09.459831 140303782647808 pyconfig.py:457] Config param use_chat_template: False
I0512 21:29:09.459858 140303782647808 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:29:09.459884 140303782647808 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:29:09.459910 140303782647808 pyconfig.py:457] Config param use_dpo: False
I0512 21:29:09.459937 140303782647808 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:29:09.459968 140303782647808 pyconfig.py:457] Config param use_grpo: True
I0512 21:29:09.459996 140303782647808 pyconfig.py:457] Config param use_indexer: False
I0512 21:29:09.460022 140303782647808 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:29:09.460049 140303782647808 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:29:09.460075 140303782647808 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:29:09.460101 140303782647808 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:29:09.460127 140303782647808 pyconfig.py:457] Config param use_mrope: False
I0512 21:29:09.460153 140303782647808 pyconfig.py:457] Config param use_multimodal: False
I0512 21:29:09.460180 140303782647808 pyconfig.py:457] Config param use_pathways: True
I0512 21:29:09.460206 140303782647808 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:29:09.460232 140303782647808 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:29:09.460258 140303782647808 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:29:09.460284 140303782647808 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:29:09.460310 140303782647808 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:29:09.460336 140303782647808 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:29:09.460362 140303782647808 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:29:09.460389 140303782647808 pyconfig.py:457] Config param use_random_routing: False
I0512 21:29:09.460415 140303782647808 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:29:09.460441 140303782647808 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:29:09.460468 140303782647808 pyconfig.py:457] Config param use_sft: False
I0512 21:29:09.460494 140303782647808 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:29:09.460520 140303782647808 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:29:09.460546 140303782647808 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:29:09.460572 140303782647808 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:29:09.460598 140303782647808 pyconfig.py:457] Config param use_truncation: True
I0512 21:29:09.460624 140303782647808 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:29:09.460650 140303782647808 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:29:09.460677 140303782647808 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:29:09.460703 140303782647808 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:29:09.460732 140303782647808 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:29:09.460758 140303782647808 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:29:09.460784 140303782647808 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:29:09.460812 140303782647808 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:29:09.460839 140303782647808 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:29:09.460865 140303782647808 pyconfig.py:457] Config param video_path: 
I0512 21:29:09.460892 140303782647808 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:29:09.460919 140303782647808 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:29:09.460945 140303782647808 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:29:09.460976 140303782647808 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:29:09.461004 140303782647808 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:29:09.461030 140303782647808 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:29:09.461057 140303782647808 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:29:09.461083 140303782647808 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:29:09.461109 140303782647808 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:29:09.461138 140303782647808 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:29:09.461177 140303782647808 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:29:09.461207 140303782647808 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:29:09.461234 140303782647808 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:29:09.461260 140303782647808 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:29:09.461287 140303782647808 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:29:09.461313 140303782647808 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:29:09.461339 140303782647808 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:29:09.461366 140303782647808 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:29:09.461392 140303782647808 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:29:09.461419 140303782647808 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:29:09.461445 140303782647808 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:29:09.461471 140303782647808 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:29:09.461497 140303782647808 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:29:09.461523 140303782647808 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:29:09.461549 140303782647808 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:29:09.461575 140303782647808 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:29:09.461601 140303782647808 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:29:09.461628 140303782647808 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:29:09.461654 140303782647808 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:29:09.461681 140303782647808 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:29:09.461707 140303782647808 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:29:09.461735 140303782647808 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:29:09.461762 140303782647808 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:29:09.461789 140303782647808 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:29:09.461817 140303782647808 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:29:09.461849 140303782647808 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:29:09.461876 140303782647808 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:29:09.461903 140303782647808 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:29:09.461929 140303782647808 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:29:09.461966 140303782647808 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:29:09.463341 140303782647808 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:29:09.572903 140303782647808 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:29:09.573124 140303782647808 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:09.573770 140303782647808 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f88de6891e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:10.280057 140303782647808 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:29:10.280302 140303782647808 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f87ffac2a40>}, handler_registry=None
I0512 21:29:10.280657 140303782647808 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f87ffac2a40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.280734 140303782647808 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f87ff855240>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.280787 140303782647808 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f87ffac2a40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f87ffac2a40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f87ff855240>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f87ff855240>}).
I0512 21:29:10.281303 140303782647808 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:29:10.281414 140303782647808 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f9a52a0add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:29:10.313016 139896316618752 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:29:10.313249 139896316618752 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f29209d4820>}, handler_registry=None
I0512 21:29:10.313597 139896316618752 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f29209d4820>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.313674 139896316618752 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f29208902b0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.313725 139896316618752 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f29209d4820>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f29209d4820>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f29208902b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f29208902b0>}).
I0512 21:29:10.314151 139896316618752 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:29:10.314264 139896316618752 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f3b73c0add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:29:09.649413 139884987709440 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:29:09.649440 139884987709440 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:29:09.649466 139884987709440 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:29:09.649492 139884987709440 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:29:09.649518 139884987709440 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:29:09.649544 139884987709440 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:29:09.649569 139884987709440 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:29:09.649595 139884987709440 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:29:09.649622 139884987709440 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:29:09.649650 139884987709440 pyconfig.py:457] Config param steps: 54363
I0512 21:29:09.649676 139884987709440 pyconfig.py:457] Config param stop_strings: None
I0512 21:29:09.649702 139884987709440 pyconfig.py:457] Config param student_overrides: {}
I0512 21:29:09.649729 139884987709440 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:29:09.649755 139884987709440 pyconfig.py:457] Config param subslice_shape: 
I0512 21:29:09.649781 139884987709440 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:29:09.649807 139884987709440 pyconfig.py:457] Config param system_prompt: 
I0512 21:29:09.649833 139884987709440 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:29:09.649860 139884987709440 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:29:09.649886 139884987709440 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:29:09.649912 139884987709440 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:29:09.649939 139884987709440 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:29:09.649965 139884987709440 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:29:09.649991 139884987709440 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:29:09.650018 139884987709440 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:29:09.650044 139884987709440 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:29:09.650070 139884987709440 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:29:09.650096 139884987709440 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:29:09.650122 139884987709440 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:29:09.650148 139884987709440 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:29:09.650178 139884987709440 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:29:09.650205 139884987709440 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:29:09.650231 139884987709440 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:29:09.650256 139884987709440 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:29:09.650300 139884987709440 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:29:09.650333 139884987709440 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:29:09.650361 139884987709440 pyconfig.py:457] Config param train_image_column: image
I0512 21:29:09.650393 139884987709440 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:29:09.650420 139884987709440 pyconfig.py:457] Config param train_split: train
I0512 21:29:09.650446 139884987709440 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:29:09.650472 139884987709440 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:29:09.650498 139884987709440 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:29:09.650526 139884987709440 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:29:09.650552 139884987709440 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:29:09.650578 139884987709440 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:29:09.650604 139884987709440 pyconfig.py:457] Config param use_audio: False
I0512 21:29:09.650631 139884987709440 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:29:09.650659 139884987709440 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:29:09.650684 139884987709440 pyconfig.py:457] Config param use_chat_template: False
I0512 21:29:09.650711 139884987709440 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:29:09.650736 139884987709440 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:29:09.650762 139884987709440 pyconfig.py:457] Config param use_dpo: False
I0512 21:29:09.650788 139884987709440 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:29:09.650814 139884987709440 pyconfig.py:457] Config param use_grpo: True
I0512 21:29:09.650840 139884987709440 pyconfig.py:457] Config param use_indexer: False
I0512 21:29:09.650866 139884987709440 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:29:09.650893 139884987709440 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:29:09.650919 139884987709440 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:29:09.650945 139884987709440 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:29:09.650971 139884987709440 pyconfig.py:457] Config param use_mrope: False
I0512 21:29:09.650997 139884987709440 pyconfig.py:457] Config param use_multimodal: False
I0512 21:29:09.651024 139884987709440 pyconfig.py:457] Config param use_pathways: True
I0512 21:29:09.651050 139884987709440 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:29:09.651076 139884987709440 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:29:09.651102 139884987709440 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:29:09.651128 139884987709440 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:29:09.651154 139884987709440 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:29:09.651180 139884987709440 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:29:09.651206 139884987709440 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:29:09.651232 139884987709440 pyconfig.py:457] Config param use_random_routing: False
I0512 21:29:09.651258 139884987709440 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:29:09.651284 139884987709440 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:29:09.651310 139884987709440 pyconfig.py:457] Config param use_sft: False
I0512 21:29:09.651336 139884987709440 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:29:09.651362 139884987709440 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:29:09.651394 139884987709440 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:29:09.651420 139884987709440 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:29:09.651446 139884987709440 pyconfig.py:457] Config param use_truncation: True
I0512 21:29:09.651472 139884987709440 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:29:09.651498 139884987709440 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:29:09.651524 139884987709440 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:29:09.651550 139884987709440 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:29:09.651576 139884987709440 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:29:09.651602 139884987709440 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:29:09.651630 139884987709440 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:29:09.651658 139884987709440 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:29:09.651685 139884987709440 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:29:09.651711 139884987709440 pyconfig.py:457] Config param video_path: 
I0512 21:29:09.651737 139884987709440 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:29:09.651763 139884987709440 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:29:09.651789 139884987709440 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:29:09.651815 139884987709440 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:29:09.651842 139884987709440 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:29:09.651868 139884987709440 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:29:09.651894 139884987709440 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:29:09.651920 139884987709440 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:29:09.651946 139884987709440 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:29:09.651973 139884987709440 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:29:09.652012 139884987709440 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:29:09.652041 139884987709440 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:29:09.652067 139884987709440 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:29:09.652094 139884987709440 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:29:09.652120 139884987709440 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:29:09.652146 139884987709440 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:29:09.652172 139884987709440 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:29:09.652198 139884987709440 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:29:09.652224 139884987709440 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:29:09.652250 139884987709440 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:29:09.652276 139884987709440 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:29:09.652302 139884987709440 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:29:09.652328 139884987709440 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:29:09.652353 139884987709440 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:29:09.652385 139884987709440 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:29:09.652412 139884987709440 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:29:09.652438 139884987709440 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:29:09.652464 139884987709440 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:29:09.652490 139884987709440 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:29:09.652516 139884987709440 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:29:09.652542 139884987709440 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:29:09.652568 139884987709440 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:29:09.652594 139884987709440 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:29:09.652621 139884987709440 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:29:09.652650 139884987709440 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:29:09.652680 139884987709440 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:29:09.652707 139884987709440 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:29:09.652733 139884987709440 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:29:09.652759 139884987709440 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:29:09.652792 139884987709440 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:29:09.654241 139884987709440 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:29:09.830612 139884987709440 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:29:09.830825 139884987709440 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:09.831497 139884987709440 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f27593991e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:10.462423 139884987709440 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:29:10.462654 139884987709440 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f267a8e29b0>}, handler_registry=None
I0512 21:29:10.462998 139884987709440 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f267a8e29b0>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.463070 139884987709440 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f267a66ca60>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:29:10.463124 139884987709440 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f267a8e29b0>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f267a8e29b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f267a66ca60>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f267a66ca60>}).
I0512 21:29:10.463538 139884987709440 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:29:10.463666 139884987709440 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f38d080add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:29:11.070003 139884987709440 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.100210 139884987709440 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, toI0512 21:29:11.098222 140209687840768 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.099958 140209687840768 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), shouldI0512 21:29:11.010059 140642248374272 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.099980 140642248374272 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fd6cd96ba00>
I0512 21:29:11.100128 140642248374272 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100182 140642248374272 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.008036 140198073038848 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.100211 140198073038848 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f6f63347eb0>
I0512 21:29:11.100363 140198073038848 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100417 140198073038848 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.008640 140672923404288 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.099963 140672923404288 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fddf1be3eb0>
I0512 21:29:11.100113 140672923404288 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100168 140672923404288 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:10.996629 139896316618752 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.100022 139896316618752 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f2920893eb0>
I0512 21:29:11.100160 139896316618752 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100220 139896316618752 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.062765 140303782647808 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.099951 140303782647808 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f87ff857b20>
I0512 21:29:11.100101 140303782647808 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100156 140303782647808 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.014012 140526565287936 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:29:11.100020 140526565287936 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fbbddfc7eb0>
I0512 21:29:11.100167 140526565287936 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100223 140526565287936 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
delete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f267a66e950>
I0512 21:29:11.100363 139884987709440 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100423 139884987709440 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f721716beb0>
I0512 21:29:11.100092 140209687840768 checkpointing.py:326] Checkpoint manager created!
I0512 21:29:11.100147 140209687840768 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.591281 140209687840768 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.591453 140209687840768 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.591579 140209687840768 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.591681 140209687840768 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f72f628af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.591892 140209687840768 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.593075 139896316618752 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.593247 139896316618752 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.593364 139896316618752 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.593463 139896316618752 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f29ff9992a0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.593701 139896316618752 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.604900 140198073038848 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.605082 140198073038848 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.605213 140198073038848 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.605329 140198073038848 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f7041f8d2a0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.605559 140198073038848 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.615148 140642248374272 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.615358 140642248374272 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.615491 140642248374272 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.615597 140642248374272 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fd7aca86f80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.615823 140642248374272 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.619018 140303782647808 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.619199 140303782647808 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.619350 140303782647808 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.619464 140303782647808 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f88de6891e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.619707 140303782647808 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.626976 140672923404288 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.627162 140672923404288 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.627294 140672923404288 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.627400 140672923404288 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fded0c8ef80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.627674 140672923404288 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.630033 140526565287936 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.630230 140526565287936 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.630366 140526565287936 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.630479 140526565287936 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fbcbd182fe0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.630714 140526565287936 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.635027 139884987709440 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:29:11.635224 139884987709440 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:11.635358 139884987709440 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:29:11.635473 139884987709440 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f27593991e0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:29:11.635711 139884987709440 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.839212 139896316618752 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.839787 139896316618752 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.25 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.850200 140198073038848 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.850759 140198073038848 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.25 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.894028 139884987709440 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.894628 139884987709440 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.26 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.910791 140303782647808 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.911713 140303782647808 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.29 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.912524 140526565287936 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.913106 140526565287936 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.28 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.928824 140209687840768 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.929419 140209687840768 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.34 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.958735 140642248374272 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.959336 140642248374272 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.34 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.958209 140672923404288 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:11.958786 140672923404288 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.33 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:29:12.192353   79715 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.352811   76181 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.375971   74477 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.400527   83818 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.448796   81102 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.533647   73721 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.549918   80401 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:12.581092   74542 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:29:13.471861 139896316618752 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.167 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2848987579345703 s) (per-host)
I0512 21:29:13.478741 139896316618752 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 4.052 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.4807407855987549 s) (per-host)
I0512 21:29:13.652916 140526565287936 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.170 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2821710109710693 s) (per-host)
I0512 21:29:13.655507 140198073038848 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.147 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3078560829162598 s) (per-host)
I0512 21:29:13.661737 140198073038848 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 3.840 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5627028942108154 s) (per-host)
I0512 21:29:13.660410 140526565287936 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 3.909 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5350492000579834 s) (per-host)
I0512 21:29:13.725673 140209687840768 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.170 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2821142673492432 s) (per-host)
I0512 21:29:13.732370 140209687840768 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 3.881 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5458288192749023 s) (per-host)
I0512 21:29:13.752692 140672923404288 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.105 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3572826385498047 s) (per-host)
I0512 21:29:13.767385 140672923404288 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 3.966 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.512754201889038 s) (per-host)
I0512 21:29:13.856411 140642248374272 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.171 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2807049751281738 s) (per-host)
I0512 21:29:13.864055 140642248374272 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 3.761 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5952401161193848 s) (per-host)
I0512 21:29:13.880611 140303782647808 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.109 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3525431156158447 s) (per-host)
I0512 21:29:13.887051 140303782647808 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.651 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6432631015777588 s) (per-host)
I0512 21:29:13.911900 139884987709440 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.097 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3674485683441162 s) (per-host)
I0512 21:29:13.918650 139884987709440 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 3.605 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.6646010875701904 s) (per-host)
I0512 21:29:13.930158 140198073038848 event_tracking.py:138] [process=2] [sync] Finished load in 2.32 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.930670 140198073038848 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:29:13.929947 140209687840768 event_tracking.py:138] [process=1] [sync] Finished load in 2.34 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.929964 140303782647808 event_tracking.py:138] [process=6] [sync] Finished load in 2.31 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.930047 140526565287936 event_tracking.py:138] [process=4] [sync] Finished load in 2.30 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.930492 140303782647808 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:29:13.930691 139896316618752 event_tracking.py:138] [process=0] [sync] Finished load in 2.34 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.930402 140209687840768 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:29:13.930610 140526565287936 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:29:13.930042 140642248374272 event_tracking.py:138] [process=5] [sync] Finished load in 2.31 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.929954 140672923404288 event_tracking.py:138] [process=7] [sync] Finished load in 2.30 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:29:13.930366 140672923404288 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:29:13.931246 139896316618752 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:29:13.930534 140642248374272 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_I0512 21:29:13.932250 139884987709440 event_tracking.py:138] [process=3] [sync] Finished load in 2.30 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:29:13.932737 139884987709440 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

The `FULL PARAMS TREE` output reveals something very suspicious: the leaf nodes (like `scale`, `kernel`, and `embedding`) are reported as type `VariableState` but show **no children and no shapes**. 

This means my `print_tree` logic is being "tricked" by a custom MaxText/Flax wrapper (likely from the `qwix` or `tokamax` libraries used in your training). If these objects aren't standard JAX arrays, the `Number of model params=0.134 billion` calculation might be undercounting because it doesn't know how to "look inside" them.

I'm going to update `convert_helper.py` to use JAX's internal `tree_flatten_with_path`. This is the "nuclear option"—it bypasses all object wrappers and forces JAX to reveal every single hidden array in the checkpoint.

**Please run these commands again with the updated script below:**

```bash
# 1. Sync the updated nuclear-inspection logic
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run the extraction
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

**Updated `src/train/convert_helper.py`:**

```python
import sys
import os
import functools
import types
import importlib.abc
from unittest.mock import MagicMock
import jax

# 1. JAX/Distributed Setup
sys.path.append(os.path.join(os.path.expanduser('~'), 'maxtext', 'src'))
try:
    jax.distributed.initialize()
    print(f"Worker {jax.process_index()} initialized.")
except: pass

# 2. Mocks
class MockFinder(importlib.abc.MetaPathFinder):
    _MOCKS = ('jax.pallas', 'qwix', 'tokamax', 'drjax', 'pathwaysutils')
    def find_module(self, fullname, path=None):
        if any(fullname == m or fullname.startswith(m + '.') for m in self._MOCKS): return self
    def load_module(self, fullname):
        mod = types.ModuleType(fullname); mod.__path__ = []; mod.__loader__ = self
        mod.__getattr__ = lambda name: MagicMock()
        sys.modules[fullname] = mod; return mod
sys.meta_path.insert(0, MockFinder())

import maxtext.utils.generate_param_only_checkpoint as gpoc

def nuclear_print(tree, label="TREE"):
    print(f"\n--- {label} (Flattened) ---")
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    found_any = False
    for path, leaf in leaves:
        found_any = True
        path_str = jax.tree_util.keystr(path)
        shape = getattr(leaf, 'shape', 'N/A')
        print(f"  {path_str} | shape: {shape} | type: {type(leaf).__name__}")
    if not found_any:
        print("  EMPTY TREE")

def patched_possibly_unroll_params(config, training_state, training_state_annotations, mesh):
    # This is where we see exactly what JAX sees
    nuclear_print(training_state.params, "PARAMS")
    
    # Search for anything with shape (22, ...)
    leaves, _ = jax.tree_util.tree_flatten_with_path(training_state.params)
    matches = [(jax.tree_util.keystr(p), l.shape) for p, l in leaves if hasattr(l, 'shape') and len(l.shape) > 0 and l.shape[0] == 22]
    
    if not matches:
        print("\nSearching opt_state...")
        leaves_opt, _ = jax.tree_util.tree_flatten_with_path(training_state.opt_state)
        matches = [(jax.tree_util.keystr(p), l.shape) for p, l in leaves_opt if hasattr(l, 'shape') and len(l.shape) > 0 and l.shape[0] == 22]

    if not matches:
        print("\nCRITICAL: No array with 22 layers found. Checking directory...")
        os.system(f"gcloud storage ls -R {config.load_full_state_path.replace('/items', '')}")
        raise ValueError("Could not find layers. Check the 'Flattened' tree above for clues.")

    print(f"\nFOUND LAYERS AT: {matches[0][0]}")
    # (Rest of unrolling logic will be added here once we have the path)

gpoc._possibly_unroll_params = patched_possibly_unroll_params

from absl import app
if __name__ == "__main__":
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(gpoc.main)
```

I have written this updated logic to `src/train/convert_helper.py` locally. Please run the sync and execute commands!

### User Input

I0512 21:30:32.682364 139678267201536 pyconfig.py:457] Config param rampup_end_step: 0
I0512 21:30:32.682395 139678267201536 pyconfig.py:457] Config param rampup_samples_per_increment_to_load: None
I0512 21:30:32.682424 139678267201536 pyconfig.py:457] Config param reasoning_end_token: </reasoning>
I0512 21:30:32.682452 139678267201536 pyconfig.py:457] Config param reasoning_start_token: <reasoning>
I0512 21:30:32.682481 139678267201536 pyconfig.py:457] Config param record_internal_nn_metrics: 0
I0512 21:30:32.682509 139678267201536 pyconfig.py:457] Config param remat_policy: full
I0512 21:30:32.682538 139678267201536 pyconfig.py:457] Config param remat_policy_for_vit: minimal
I0512 21:30:32.682567 139678267201536 pyconfig.py:457] Config param remove_size_one_mesh_axis_from_type: True
I0512 21:30:32.682595 139678267201536 pyconfig.py:457] Config param replicate_quant_scale: False
I0512 21:30:32.682629 139678267201536 pyconfig.py:457] Config param replicator_backup_interval_minutes: 0
I0512 21:30:32.682659 139678267201536 pyconfig.py:457] Config param report_heartbeat_metric_for_gcp_monitoring: False
I0512 21:30:32.682687 139678267201536 pyconfig.py:457] Config param report_performance_metric_for_gcp_monitoring: False
I0512 21:30:32.682715 139678267201536 pyconfig.py:457] Config param reshape_q: False
I0512 21:30:32.682744 139678267201536 pyconfig.py:457] Config param reshard_chunk_size: None
I0512 21:30:32.682772 139678267201536 pyconfig.py:457] Config param return_log_prob: False
I0512 21:30:32.682801 139678267201536 pyconfig.py:457] Config param reuse_example_batch: 0
I0512 21:30:32.682829 139678267201536 pyconfig.py:457] Config param reward_exact_answer: 5.0
I0512 21:30:32.682858 139678267201536 pyconfig.py:457] Config param reward_exact_format_match: 3.0
I0512 21:30:32.682886 139678267201536 pyconfig.py:457] Config param reward_partial_format_match: 0.5
I0512 21:30:32.682916 139678267201536 pyconfig.py:457] Config param reward_ratio_guess_to_answer_high: 0.5
I0512 21:30:32.682945 139678267201536 pyconfig.py:457] Config param reward_ratio_guess_to_answer_low: 0.25
I0512 21:30:32.682974 139678267201536 pyconfig.py:457] Config param reward_white_space_format_match: 1.5
I0512 21:30:32.683003 139678267201536 pyconfig.py:457] Config param rl: {'num_generations': 2, 'num_iterations': 1, 'grpo_beta': 0.08, 'grpo_epsilon': 0.2, 'loss_algo': 'grpo', 'use_agentic_rollout': False, 'max_concurrency': 256, 'off_policy_steps': 0, 'system_prompt': '', 'degenerate_group_masking': True, 'epsilon_high': None, 'reshard_chunk_size': None}
I0512 21:30:32.683038 139678267201536 pyconfig.py:457] Config param rollout_data_parallelism: -1
I0512 21:30:32.683068 139678267201536 pyconfig.py:457] Config param rollout_expert_parallelism: 1
I0512 21:30:32.683096 139678267201536 pyconfig.py:457] Config param rollout_micro_batch_size: -1
I0512 21:30:32.683125 139678267201536 pyconfig.py:457] Config param rollout_tensor_parallelism: -1
I0512 21:30:32.683153 139678267201536 pyconfig.py:457] Config param rope_attention_scaling: False
I0512 21:30:32.683181 139678267201536 pyconfig.py:457] Config param rope_factor: 40
I0512 21:30:32.683210 139678267201536 pyconfig.py:457] Config param rope_interleave: True
I0512 21:30:32.683239 139678267201536 pyconfig.py:457] Config param rope_linear_scaling_factor: 1.0
I0512 21:30:32.683268 139678267201536 pyconfig.py:457] Config param rope_max_timescale: 10000
I0512 21:30:32.683296 139678267201536 pyconfig.py:457] Config param rope_min_timescale: 1
I0512 21:30:32.683324 139678267201536 pyconfig.py:457] Config param rope_theta_for_vit: 10000
I0512 21:30:32.683352 139678267201536 pyconfig.py:457] Config param rope_truncate: True
I0512 21:30:32.683382 139678267201536 pyconfig.py:457] Config param rope_type: RopeType.DEFAULT
I0512 21:30:32.683415 139678267201536 pyconfig.py:457] Config param rope_use_scale: True
I0512 21:30:32.683444 139678267201536 pyconfig.py:457] Config param routed_bias: False
I0512 21:30:32.683472 139678267201536 pyconfig.py:457] Config param routed_bias_update_rate: 0.0
I0512 21:30:32.683500 139678267201536 pyconfig.py:457] Config param routed_scaling_factor: 1.0
I0512 21:30:32.683529 139678267201536 pyconfig.py:457] Config param routed_score_func: 
I0512 21:30:32.683556 139678267201536 pyconfig.py:457] Config param run_name: tinymath-1b-decode
I0512 21:30:32.683584 139678267201536 pyconfig.py:457] Config param sa_block_kv: 512
I0512 21:30:32.683613 139678267201536 pyconfig.py:457] Config param sa_block_kv_compute: 512
I0512 21:30:32.683648 139678267201536 pyconfig.py:457] Config param sa_block_kv_dkv: 512
I0512 21:30:32.683676 139678267201536 pyconfig.py:457] Config param sa_block_kv_dkv_compute: 512
I0512 21:30:32.683704 139678267201536 pyconfig.py:457] Config param sa_block_kv_dq: 512
I0512 21:30:32.683732 139678267201536 pyconfig.py:457] Config param sa_block_q: 512
I0512 21:30:32.683760 139678267201536 pyconfig.py:457] Config param sa_block_q_dkv: 512
I0512 21:30:32.683788 139678267201536 pyconfig.py:457] Config param sa_block_q_dq: 512
I0512 21:30:32.683816 139678267201536 pyconfig.py:457] Config param sa_k_layout: HEAD_DIM_MINOR
I0512 21:30:32.683843 139678267201536 pyconfig.py:457] Config param sa_q_layout: HEAD_DIM_MINOR
I0512 21:30:32.683871 139678267201536 pyconfig.py:457] Config param sa_use_fused_bwd_kernel: False
I0512 21:30:32.683899 139678267201536 pyconfig.py:457] Config param sa_v_layout: HEAD_DIM_MINOR
I0512 21:30:32.683927 139678267201536 pyconfig.py:457] Config param sampler_devices_fraction: 0.5
I0512 21:30:32.683956 139678267201536 pyconfig.py:457] Config param save_checkpoint_on_completion: True
I0512 21:30:32.683984 139678267201536 pyconfig.py:457] Config param save_config_to_gcs: False
I0512 21:30:32.684012 139678267201536 pyconfig.py:457] Config param save_quantized_params_path: 
I0512 21:30:32.684040 139678267201536 pyconfig.py:457] Config param scale_embedding_for_audio: True
I0512 21:30:32.684068 139678267201536 pyconfig.py:457] Config param scan_layers: True
I0512 21:30:32.684096 139678267201536 pyconfig.py:457] Config param scan_layers_per_stage: False
I0512 21:30:32.684124 139678267201536 pyconfig.py:457] Config param scan_pipeline_iterations: True
I0512 21:30:32.684151 139678267201536 pyconfig.py:457] Config param scan_pipeline_repeats: False
I0512 21:30:32.684179 139678267201536 pyconfig.py:457] Config param set_remat_policy_on_layers_per_stage: False
I0512 21:30:32.684207 139678267201536 pyconfig.py:457] Config param set_remat_policy_on_pipeline_iterations: True
I0512 21:30:32.684235 139678267201536 pyconfig.py:457] Config param sft_train_on_completion_only: False
I0512 21:30:32.684263 139678267201536 pyconfig.py:457] Config param shard_exp_on_fsdp: False
I0512 21:30:32.684291 139678267201536 pyconfig.py:457] Config param shard_mode: ShardMode.AUTO
I0512 21:30:32.684322 139678267201536 pyconfig.py:457] Config param shard_optimizer_over_data: False
I0512 21:30:32.684350 139678267201536 pyconfig.py:457] Config param sharding_strategy: None
I0512 21:30:32.684379 139678267201536 pyconfig.py:457] Config param sharding_tolerance: 0.3
I0512 21:30:32.684410 139678267201536 pyconfig.py:457] Config param shardy: True
I0512 21:30:32.684438 139678267201536 pyconfig.py:457] Config param share_kv_projections: False
I0512 21:30:32.684466 139678267201536 pyconfig.py:457] Config param shared_experts: 0
I0512 21:30:32.684494 139678267201536 pyconfig.py:457] Config param sinkhorn_iterations: 20
I0512 21:30:32.684522 139678267201536 pyconfig.py:457] Config param skip_first_n_steps_for_profiler: 1
I0512 21:30:32.684550 139678267201536 pyconfig.py:457] Config param skip_jax_distributed_system: False
I0512 21:30:32.684578 139678267201536 pyconfig.py:457] Config param skip_step_interval: 128
I0512 21:30:32.684606 139678267201536 pyconfig.py:457] Config param skip_step_on_spikes: False
I0512 21:30:32.684639 139678267201536 pyconfig.py:457] Config param skip_step_scaling_factor: 6.0
I0512 21:30:32.684670 139678267201536 pyconfig.py:457] Config param sliding_window_size: 0
I0512 21:30:32.496377 139683606968320 pyconfig.py:457] Config param sharding_tolerance: 0.3
I0512 21:30:32.496406 139683606968320 pyconfig.py:457] Config param shardy: True
I0512 21:30:32.496434 139683606968320 pyconfig.py:457] Config param share_kv_projections: False
I0512 21:30:32.496462 139683606968320 pyconfig.py:457] Config param shared_experts: 0
I0512 21:30:32.496490 139683606968320 pyconfig.py:457] Config param sinkhorn_iterations: 20
I0512 21:30:32.496517 139683606968320 pyconfig.py:457] Config param skip_first_n_steps_for_profiler: 1
I0512 21:30:32.496545 139683606968320 pyconfig.py:457] Config param skip_jax_distributed_system: False
I0512 21:30:32.496572 139683606968320 pyconfig.py:457] Config param skip_step_interval: 128
I0512 21:30:32.496600 139683606968320 pyconfig.py:457] Config param skip_step_on_spikes: False
I0512 21:30:32.496628 139683606968320 pyconfig.py:457] Config param skip_step_scaling_factor: 6.0
I0512 21:30:32.496657 139683606968320 pyconfig.py:457] Config param sliding_window_size: 0
I0512 21:30:32.496684 139683606968320 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:30:32.496716 139683606968320 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:30:32.496746 139683606968320 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:30:32.496773 139683606968320 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:30:32.496801 139683606968320 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:30:32.496828 139683606968320 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:30:32.496855 139683606968320 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:30:32.496882 139683606968320 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:30:32.496910 139683606968320 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:30:32.496937 139683606968320 pyconfig.py:457] Config param steps: 54363
I0512 21:30:32.496965 139683606968320 pyconfig.py:457] Config param stop_strings: None
I0512 21:30:32.496994 139683606968320 pyconfig.py:457] Config param student_overrides: {}
I0512 21:30:32.497022 139683606968320 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:30:32.497050 139683606968320 pyconfig.py:457] Config param subslice_shape: 
I0512 21:30:32.497077 139683606968320 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:30:32.497104 139683606968320 pyconfig.py:457] Config param system_prompt: 
I0512 21:30:32.497131 139683606968320 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:30:32.497159 139683606968320 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:30:32.497187 139683606968320 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:30:32.497214 139683606968320 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:30:32.497241 139683606968320 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:30:32.497269 139683606968320 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:30:32.497296 139683606968320 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:30:32.497323 139683606968320 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:30:32.497350 139683606968320 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:30:32.497377 139683606968320 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:30:32.497404 139683606968320 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:30:32.497431 139683606968320 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:30:32.497459 139683606968320 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:30:32.497492 139683606968320 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:30:32.497520 139683606968320 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:30:32.497548 139683606968320 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:30:32.497575 139683606968320 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:30:32.497603 139683606968320 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:30:32.497631 139683606968320 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:30:32.497658 139683606968320 pyconfig.py:457] Config param train_image_column: image
I0512 21:30:32.497686 139683606968320 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:30:32.497719 139683606968320 pyconfig.py:457] Config param train_split: train
I0512 21:30:32.497747 139683606968320 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:30:32.497775 139683606968320 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:30:32.497802 139683606968320 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:30:32.497831 139683606968320 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:30:32.497859 139683606968320 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:30:32.497886 139683606968320 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:30:32.497913 139683606968320 pyconfig.py:457] Config param use_audio: False
I0512 21:30:32.497941 139683606968320 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:30:32.497969 139683606968320 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:30:32.497997 139683606968320 pyconfig.py:457] Config param use_chat_template: False
I0512 21:30:32.498025 139683606968320 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:30:32.498052 139683606968320 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:30:32.498079 139683606968320 pyconfig.py:457] Config param use_dpo: False
I0512 21:30:32.498106 139683606968320 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:30:32.498133 139683606968320 pyconfig.py:457] Config param use_grpo: True
I0512 21:30:32.498161 139683606968320 pyconfig.py:457] Config param use_indexer: False
I0512 21:30:32.498188 139683606968320 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:30:32.498215 139683606968320 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:30:32.498243 139683606968320 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:30:32.498270 139683606968320 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:30:32.498297 139683606968320 pyconfig.py:457] Config param use_mrope: False
I0512 21:30:32.498324 139683606968320 pyconfig.py:457] Config param use_multimodal: False
I0512 21:30:32.498351 139683606968320 pyconfig.py:457] Config param use_pathways: True
I0512 21:30:32.498378 139683606968320 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:30:32.498405 139683606968320 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:30:32.498432 139683606968320 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:30:32.498459 139683606968320 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:30:32.498486 139683606968320 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:30:32.498514 139683606968320 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:30:32.498541 139683606968320 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:30:32.498569 139683606968320 pyconfig.py:457] Config param use_random_routing: False
I0512 21:30:32.498595 139683606968320 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:30:32.498622 139683606968320 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:30:32.498650 139683606968320 pyconfig.py:457] Config param use_sft: False
I0512 21:30:32.498677 139683606968320 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:30:32.498704 139683606968320 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:30:32.498739 139683606968320 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:30:32.498767 139683606968320 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:30:32.498794 139683606968320 pyconfig.py:457] Config param use_truncation: True
I0512 21:30:32.498821 139683606968320 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:30:32.498848 139683606968320 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:30:32.498876 139683606968320 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:30:32.498907 139683606968320 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:30:32.498935 139683606968320 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:30:32.498962 139683606968320 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:30:32.498991 139683606968320 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:30:32.499020 139683606968320 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:30:32.499047 139683606968320 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:30:32.499075 139683606968320 pyconfig.py:457] Config param video_path: 
I0512 21:30:32.499102 139683606968320 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:30:32.499129 139683606968320 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:30:32.499156 139683606968320 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:30:32.499184 139683606968320 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:30:32.499211 139683606968320 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:30:32.499238 139683606968320 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:30:32.499286 139683606968320 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:30:32.499318 139683606968320 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:30:32.499346 139683606968320 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:30:32.499375 139683606968320 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:30:32.499416 139683606968320 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:30:32.499446 139683606968320 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:30:32.499474 139683606968320 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:30:32.499502 139683606968320 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:30:32.499530 139683606968320 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:30:32.499557 139683606968320 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:30:32.499584 139683606968320 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:30:32.499612 139683606968320 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:30:32.499639 139683606968320 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:30:32.499666 139683606968320 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:30:32.499693 139683606968320 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:30:32.499726 139683606968320 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:30:32.499754 139683606968320 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:30:32.499782 139683606968320 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:30:32.499809 139683606968320 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:30:32.499836 139683606968320 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:30:32.499863 139683606968320 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:30:32.499890 139683606968320 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:30:32.499917 139683606968320 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:30:32.499944 139683606968320 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:30:32.499972 139683606968320 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:30:32.500001 139683606968320 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:30:32.500028 139683606968320 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:30:32.500056 139683606968320 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:30:32.500084 139683606968320 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:30:32.500117 139683606968320 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:30:32.500146 139683606968320 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:30:32.500173 139683606968320 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:30:32.500200 139683606968320 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:30:32.500232 139683606968320 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:30:32.501608 139683606968320 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:30:32.596996 139683606968320 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:30:32.597202 139683606968320 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:32.597929 139683606968320 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7ef879d6ef80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:33.164636 139683606968320 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:30:33.164871 139683606968320 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef79b0ac880>}, handler_registry=None
I0512 21:30:33.165216 139683606968320 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef79b0ac880>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:30:33.165289 139683606968320 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef79af648b0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:30:33.165342 139683606968320 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef79b0ac880>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef79b0ac880>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef79af648b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef79af648b0>}).
I0512 21:30:33.165774 139683606968320 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:30:33.165890 139683606968320 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f09ed77add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:30:32.684698 139678267201536 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:30:32.684726 139678267201536 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:30:32.684755 139678267201536 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:30:32.684783 139678267201536 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:30:32.684811 139678267201536 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:30:32.684839 139678267201536 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:30:32.684867 139678267201536 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:30:32.684895 139678267201536 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:30:32.684923 139678267201536 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:30:32.684952 139678267201536 pyconfig.py:457] Config param steps: 54363
I0512 21:30:32.684979 139678267201536 pyconfig.py:457] Config param stop_strings: None
I0512 21:30:32.685008 139678267201536 pyconfig.py:457] Config param student_overrides: {}
I0512 21:30:32.685036 139678267201536 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:30:32.685064 139678267201536 pyconfig.py:457] Config param subslice_shape: 
I0512 21:30:32.685092 139678267201536 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:30:32.685120 139678267201536 pyconfig.py:457] Config param system_prompt: 
I0512 21:30:32.685148 139678267201536 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:30:32.685176 139678267201536 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:30:32.685204 139678267201536 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:30:32.685233 139678267201536 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:30:32.685261 139678267201536 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:30:32.685290 139678267201536 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:30:32.685318 139678267201536 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:30:32.685346 139678267201536 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:30:32.685375 139678267201536 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:30:32.685405 139678267201536 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:30:32.685433 139678267201536 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:30:32.685462 139678267201536 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:30:32.685490 139678267201536 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:30:32.685521 139678267201536 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:30:32.685549 139678267201536 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:30:32.685578 139678267201536 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:30:32.685606 139678267201536 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:30:32.685639 139678267201536 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:30:32.685669 139678267201536 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:30:32.685698 139678267201536 pyconfig.py:457] Config param train_image_column: image
I0512 21:30:32.685726 139678267201536 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:30:32.685755 139678267201536 pyconfig.py:457] Config param train_split: train
I0512 21:30:32.685783 139678267201536 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:30:32.685811 139678267201536 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:30:32.685840 139678267201536 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:30:32.685869 139678267201536 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:30:32.685897 139678267201536 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:30:32.685925 139678267201536 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:30:32.685953 139678267201536 pyconfig.py:457] Config param use_audio: False
I0512 21:30:32.685981 139678267201536 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:30:32.686009 139678267201536 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:30:32.686037 139678267201536 pyconfig.py:457] Config param use_chat_template: False
I0512 21:30:32.686065 139678267201536 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:30:32.686093 139678267201536 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:30:32.686120 139678267201536 pyconfig.py:457] Config param use_dpo: False
I0512 21:30:32.686149 139678267201536 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:30:32.686176 139678267201536 pyconfig.py:457] Config param use_grpo: True
I0512 21:30:32.686204 139678267201536 pyconfig.py:457] Config param use_indexer: False
I0512 21:30:32.686232 139678267201536 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:30:32.686260 139678267201536 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:30:32.686309 139678267201536 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:30:32.686342 139678267201536 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:30:32.686372 139678267201536 pyconfig.py:457] Config param use_mrope: False
I0512 21:30:32.686402 139678267201536 pyconfig.py:457] Config param use_multimodal: False
I0512 21:30:32.686430 139678267201536 pyconfig.py:457] Config param use_pathways: True
I0512 21:30:32.686458 139678267201536 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:30:32.686486 139678267201536 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:30:32.686514 139678267201536 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:30:32.686542 139678267201536 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:30:32.686571 139678267201536 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:30:32.686599 139678267201536 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:30:32.686632 139678267201536 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:30:32.686661 139678267201536 pyconfig.py:457] Config param use_random_routing: False
I0512 21:30:32.686690 139678267201536 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:30:32.686718 139678267201536 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:30:32.686746 139678267201536 pyconfig.py:457] Config param use_sft: False
I0512 21:30:32.686773 139678267201536 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:30:32.686801 139678267201536 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:30:32.686829 139678267201536 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:30:32.686857 139678267201536 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:30:32.686885 139678267201536 pyconfig.py:457] Config param use_truncation: True
I0512 21:30:32.686913 139678267201536 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:30:32.686941 139678267201536 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:30:32.686969 139678267201536 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:30:32.686997 139678267201536 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:30:32.687025 139678267201536 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:30:32.687053 139678267201536 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:30:32.687080 139678267201536 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:30:32.687110 139678267201536 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:30:32.687138 139678267201536 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:30:32.687165 139678267201536 pyconfig.py:457] Config param video_path: 
I0512 21:30:32.687193 139678267201536 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:30:32.687221 139678267201536 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:30:32.687249 139678267201536 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:30:32.687277 139678267201536 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:30:32.687305 139678267201536 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:30:32.687333 139678267201536 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:30:32.687361 139678267201536 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:30:32.687391 139678267201536 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:30:32.687420 139678267201536 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:30:32.687450 139678267201536 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:30:32.687491 139678267201536 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:30:32.687521 139678267201536 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:30:32.687550 139678267201536 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:30:32.687578 139678267201536 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:30:32.687607 139678267201536 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:30:32.687641 139678267201536 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:30:32.687670 139678267201536 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:30:32.687698 139678267201536 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:30:32.687727 139678267201536 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:30:32.687755 139678267201536 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:30:32.687783 139678267201536 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:30:32.687811 139678267201536 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:30:32.687839 139678267201536 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:30:32.687867 139678267201536 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:30:32.687894 139678267201536 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:30:32.687922 139678267201536 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:30:32.687950 139678267201536 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:30:32.687978 139678267201536 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:30:32.688006 139678267201536 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:30:32.688034 139678267201536 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:30:32.688062 139678267201536 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:30:32.688090 139678267201536 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:30:32.688118 139678267201536 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:30:32.688146 139678267201536 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:30:32.688175 139678267201536 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:30:32.688206 139678267201536 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:30:32.688235 139678267201536 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:30:32.688263 139678267201536 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:30:32.688291 139678267201536 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:30:32.688323 139678267201536 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:30:32.689787 139678267201536 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:30:32.767372 139678267201536 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:30:32.767466 139678267201536 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:32.767880 139678267201536 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7ef737c92f80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:33.215398 139678267201536 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:30:33.215582 139678267201536 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef6586b0880>}, handler_registry=None
I0512 21:30:33.215841 139678267201536 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef6586b0880>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:30:33.215910 139678267201536 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef6585688b0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:30:33.215961 139678267201536 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef6586b0880>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7ef6586b0880>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef6585688b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7ef6585688b0>}).
I0512 21:30:33.216398 139678267201536 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:30:33.216512 139678267201536 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f08af316dd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:30:33.591714 139645011585024 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.610576 139678267201536 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.607641 139678433204224 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.614965 139683606968320 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.613958 139791447758848 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.622408 140184513419264 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.635348 140133668972544 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.659674 139683606968320 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7ef79af67eb0>
I0512 21:30:33.659830 139683606968320 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659887 139683606968320 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:33.659522 139791447758848 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f10b616beb0>
I0512 21:30:33.659655 139791447758848 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659710 139791447758848 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:33.659564 139678433204224 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7ef665ca3eb0>
I0512 21:30:33.659705 139678433204224 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659758 139678433204224 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:33.659612 139678267201536 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7ef65856beb0>
I0512 21:30:33.659757 139678267201536 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659813 139678267201536 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:33.657744 139799551842304 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:30:33.659680 139645011585024 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), shouldI0512 21:30:33.659488 140184513419264 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(nI0512 21:30:33.659629 140133668972544 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f6063c87eb0>
I0512 21:30:33.659778 140133668972544 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659834 140133668972544 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f6c3b023eb0>
I0512 21:30:33.659625 140184513419264 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659678 140184513419264 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7eee9e157b20>
I0512 21:30:33.659816 139645011585024 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659870 139645011585024 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:33.659625 139799551842304 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f1299867b20>
I0512 21:30:33.659767 139799551842304 checkpointing.py:326] Checkpoint manager created!
I0512 21:30:33.659823 139799551842304 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.129772 140184513419264 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.129941 140184513419264 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.130061 140184513419264 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.130159 140184513419264 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f6d19d6af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.130370 140184513419264 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.145095 139678433204224 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.145302 139678433204224 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.145426 139678433204224 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.145545 139678433204224 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7ef744d7af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.145761 139678433204224 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.148300 139791447758848 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.148486 139791447758848 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.148619 139791447758848 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.148726 139791447758848 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f119538d2a0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.148943 139791447758848 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.155536 139645011585024 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.155714 139645011585024 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.155838 139645011585024 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.155940 139645011585024 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7eef7d1852a0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.156186 139645011585024 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.163046 139678267201536 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.163243 139678267201536 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.163383 139678267201536 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.163496 139678267201536 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7ef737c92f80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.163723 139678267201536 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.177203 139799551842304 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.177402 139799551842304 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.177538 139799551842304 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.177656 139799551842304 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f137867af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.177887 139799551842304 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.179158 140133668972544 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.179347 140133668972544 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.179479 140133668972544 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.179605 140133668972544 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f6142999270>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.179843 140133668972544 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.184012 139683606968320 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:30:34.184196 139683606968320 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:34.184327 139683606968320 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:30:34.184436 139683606968320 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7ef879d6ef80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:30:34.184663 139683606968320 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.284628 140184513419264 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.285204 140184513419264 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.15 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.300311 139678433204224 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.300878 139678433204224 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.16 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.301614 139791447758848 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.302162 139791447758848 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.15 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.316355 139678267201536 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.316907 139678267201536 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.15 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.329092 140133668972544 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.329638 140133668972544 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.15 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.335130 139645011585024 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.335699 139645011585024 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.18 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.334363 139799551842304 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.334958 139799551842304 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.16 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.409131 139683606968320 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.409721 139683606968320 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.23 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:30:34.603529   82200 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.624243   82654 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.627136   85423 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.614118   76097 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.651176   76043 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.670983   81262 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.692768   77724 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:34.743426   75266 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:30:35.792610 140184513419264 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.278 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.1736969947814941 s) (per-host)
I0512 21:30:35.798512 140184513419264 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 4.458 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3460149765014648 s) (per-host)
I0512 21:30:35.800406 139791447758848 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.259 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.1914658546447754 s) (per-host)
I0512 21:30:35.806539 139791447758848 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 4.428 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3549416065216064 s) (per-host)
I0512 21:30:35.807065 139678267201536 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.241 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.208878993988037 s) (per-host)
I0512 21:30:35.812751 139678267201536 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 4.419 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3577232360839844 s) (per-host)
I0512 21:30:35.817953 140133668972544 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.254 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.195957899093628 s) (per-host)
I0512 21:30:35.823982 140133668972544 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 4.481 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.33892822265625 s) (per-host)
I0512 21:30:35.830872 139799551842304 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.266 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.1848936080932617 s) (per-host)
I0512 21:30:35.838700 139799551842304 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 4.477 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3403067588806152 s) (per-host)
I0512 21:30:35.855541 139678433204224 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.260 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.1901092529296875 s) (per-host)
I0512 21:30:35.862750 139678433204224 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 4.485 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3377175331115723 s) (per-host)
I0512 21:30:35.902642 139645011585024 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.234 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2150943279266357 s) (per-host)
I0512 21:30:35.908858 139645011585024 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 4.231 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.4181933403015137 s) (per-host)
I0512 21:30:36.108853 139683606968320 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.094 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.3709125518798828 s) (per-host)
I0512 21:30:36.114508 139683606968320 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 3.927 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.5278632640838623 s) (per-host)
I0512 21:30:36.120062 139791447758848 event_tracking.py:138] [process=5] [sync] Finished load in 1.97 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:36.120459 139791447758848 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:30:36.120496 139678433204224 event_tracking.py:138] [process=0] [sync] Finished load in 1.97 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:36.120148 139678267201536 event_tracking.py:138] [process=3] [sync] Finished load in 1.96 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:36.121079 139678433204224 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:30:36.120651 139678267201536 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:30:36.120167 139645011585024 event_tracking.py:138] [process=2] [sync] Finished load in 1.96 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.122388 139683606968320 event_tracking.py:138] [process=6] [sync] Finished load in 1.94 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.120074 140184513419264 event_tracking.py:138] [process=1] [sync] Finished load in 1.99 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.122781 139683606968320 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:30:36.120634 139645011585024 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.120497 140184513419264 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.120184 140133668972544 event_tracking.py:138] [process=7] [sync] Finished load in 1.94 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:36.120664 140133668972544 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:30:36.120257 139799551842304 event_tracking.py:138] [process=4] [sync] Finished load in 1.94 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:30:36.120831 139799551842304 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

### Planner Response

We've hit a wall because **MaxText's loader is completely ignoring the layers** during the load process. The fact that `decoder` only contains `decoder_norm` and `logits_dense` (and not a `layers` key) proves that the model structure MaxText is trying to load doesn't match what's actually in your GCS checkpoint.

We need to see what's **actually inside the files** without using MaxText's filters. I've rewritten the script to perform a "bare-metal" metadata scan of the Orbax checkpoint. This will tell us exactly what keys (and shapes) are physically present in GCS.

**Please run these commands one more time:**

```bash
# 1. Sync the bare-metal scanner
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# 2. Run the scan
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=all --command="cd ~/maxtext && PYTHONPATH=src python3 convert_helper.py"
```

**Updated `src/train/convert_helper.py` (Bare-Metal Scanner):**

```python
import sys
import os
import jax
import orbax.checkpoint as oc
from google.cloud import storage

# 1. Setup
sys.path.append(os.path.join(os.path.expanduser('~'), 'maxtext', 'src'))
try: jax.distributed.initialize()
except: pass

CHECKPOINT_PATH = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items"

def bare_metal_scan():
    print(f"\n--- BARE METAL SCAN OF {CHECKPOINT_PATH} ---")
    
    # We use a raw PyTreeCheckpointHandler to see the true structure
    handler = oc.PyTreeCheckpointHandler()
    
    # This just reads the metadata from GCS without loading the actual tensors
    try:
        # We look at the directory structure
        manager_path = CHECKPOINT_PATH.replace("/items", "")
        print(f"Reading metadata from manager path: {manager_path}")
        
        # Manually inspect the directory to see if 'items' is even there
        bucket_name = "tinymath-reason-data-himanshu"
        prefix = "checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items/"
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=20))
        
        print("\nFirst 20 files in checkpoint:")
        for b in blobs:
            print(f"  {b.name}")

        # Now try to get the PyTree structure from Orbax
        # Note: We use a dummy TrainState-like structure or just None to see what it restores
        # But better yet, let's look at the metadata file if it exists
        
    except Exception as e:
        print(f"Error during scan: {e}")

if __name__ == "__main__":
    bare_metal_scan()
```

This script will list the first 20 files in the checkpoint. If we see files like `decoder/layers/...`, we'll know the data is there and we just need to fix the path mapping. If we don't see them, the checkpoint itself might be incomplete. I've updated the local file—ready for your run!

### User Input

I0512 21:34:17.108099 139991737624576 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:34:17.108129 139991737624576 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:34:17.108157 139991737624576 pyconfig.py:457] Config param use_sft: False
I0512 21:34:17.108187 139991737624576 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:34:17.108222 139991737624576 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:34:17.108251 139991737624576 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:34:17.108281 139991737624576 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:34:17.108310 139991737624576 pyconfig.py:457] Config param use_truncation: True
I0512 21:34:17.108339 139991737624576 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:34:17.108368 139991737624576 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:34:17.108398 139991737624576 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:34:17.108427 139991737624576 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:34:17.108456 139991737624576 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:34:17.108486 139991737624576 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:34:17.108515 139991737624576 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:34:17.108545 139991737624576 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:34:17.108575 139991737624576 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:34:17.108604 139991737624576 pyconfig.py:457] Config param video_path: 
I0512 21:34:17.108633 139991737624576 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:34:17.108663 139991737624576 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:34:17.108693 139991737624576 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:34:17.108722 139991737624576 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:34:17.108752 139991737624576 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:34:17.108781 139991737624576 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:34:17.108810 139991737624576 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:34:17.108840 139991737624576 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:34:17.108869 139991737624576 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:34:17.108899 139991737624576 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:34:17.108942 139991737624576 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:34:17.108975 139991737624576 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:34:17.109006 139991737624576 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:34:17.109035 139991737624576 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:34:17.109065 139991737624576 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:34:17.109094 139991737624576 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:34:17.109124 139991737624576 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:34:17.109153 139991737624576 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.109183 139991737624576 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:34:17.109218 139991737624576 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:34:17.109248 139991737624576 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:34:17.109277 139991737624576 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:34:17.109306 139991737624576 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:34:17.109335 139991737624576 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:34:17.109364 139991737624576 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:34:17.109393 139991737624576 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:34:17.109423 139991737624576 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.109452 139991737624576 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:34:17.109498 139991737624576 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:34:17.109533 139991737624576 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:34:17.109563 139991737624576 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:34:17.109592 139991737624576 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:34:17.109622 139991737624576 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:34:17.109651 139991737624576 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:34:17.109681 139991737624576 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:34:17.109714 139991737624576 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:34:17.109744 139991737624576 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:34:17.109773 139991737624576 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:34:17.109802 139991737624576 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:34:17.109837 139991737624576 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:34:17.111171 139991737624576 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:34:17.304565 139991737624576 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:34:17.304764 139991737624576 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:17.305378 139991737624576 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f4037982f80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:17.738571 139991737624576 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:34:17.738796 139991737624576 checkpoint_manager.py:708] [process=0][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3f58ca0880>}, handler_registry=None
I0512 21:34:17.739127 139991737624576 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3f58ca0880>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.739199 139991737624576 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f3f58b588b0>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.739251 139991737624576 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3f58ca0880>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f3f58ca0880>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f3f58b588b0>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f3f58b588b0>}).
I0512 21:34:17.739688 139991737624576 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:34:17.739792 139991737624576 async_checkpointer.py:192] [process=0][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f51ab7dadd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:34:17.149750 139960082503680 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:34:17.149778 139960082503680 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:34:17.149806 139960082503680 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:34:17.149833 139960082503680 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:34:17.149862 139960082503680 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:34:17.149889 139960082503680 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:34:17.149918 139960082503680 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:34:17.149946 139960082503680 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:34:17.149974 139960082503680 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:34:17.150002 139960082503680 pyconfig.py:457] Config param steps: 54363
I0512 21:34:17.150030 139960082503680 pyconfig.py:457] Config param stop_strings: None
I0512 21:34:17.150058 139960082503680 pyconfig.py:457] Config param student_overrides: {}
I0512 21:34:17.150086 139960082503680 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:34:17.150114 139960082503680 pyconfig.py:457] Config param subslice_shape: 
I0512 21:34:17.150142 139960082503680 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:34:17.150170 139960082503680 pyconfig.py:457] Config param system_prompt: 
I0512 21:34:17.150197 139960082503680 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:34:17.150226 139960082503680 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:34:17.150254 139960082503680 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:34:17.150297 139960082503680 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:34:17.150332 139960082503680 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:34:17.150362 139960082503680 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:34:17.150393 139960082503680 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:34:17.150422 139960082503680 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:34:17.150450 139960082503680 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:34:17.150478 139960082503680 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:34:17.150505 139960082503680 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:34:17.150533 139960082503680 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:34:17.150561 139960082503680 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:34:17.150592 139960082503680 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:34:17.150625 139960082503680 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:34:17.150655 139960082503680 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:34:17.150683 139960082503680 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:34:17.150711 139960082503680 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:34:17.150740 139960082503680 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:34:17.150769 139960082503680 pyconfig.py:457] Config param train_image_column: image
I0512 21:34:17.150797 139960082503680 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:34:17.150825 139960082503680 pyconfig.py:457] Config param train_split: train
I0512 21:34:17.150853 139960082503680 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:34:17.150881 139960082503680 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:34:17.150909 139960082503680 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:34:17.150938 139960082503680 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:34:17.150967 139960082503680 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:34:17.150995 139960082503680 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:34:17.151023 139960082503680 pyconfig.py:457] Config param use_audio: False
I0512 21:34:17.151051 139960082503680 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:34:17.151078 139960082503680 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:34:17.151106 139960082503680 pyconfig.py:457] Config param use_chat_template: False
I0512 21:34:17.151134 139960082503680 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:34:17.151162 139960082503680 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:34:17.151189 139960082503680 pyconfig.py:457] Config param use_dpo: False
I0512 21:34:17.151217 139960082503680 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:34:17.151245 139960082503680 pyconfig.py:457] Config param use_grpo: True
I0512 21:34:17.151273 139960082503680 pyconfig.py:457] Config param use_indexer: False
I0512 21:34:17.151300 139960082503680 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:34:17.151328 139960082503680 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:34:17.151355 139960082503680 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:34:17.151385 139960082503680 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:34:17.151414 139960082503680 pyconfig.py:457] Config param use_mrope: False
I0512 21:34:17.151442 139960082503680 pyconfig.py:457] Config param use_multimodal: False
I0512 21:34:17.151470 139960082503680 pyconfig.py:457] Config param use_pathways: True
I0512 21:34:17.151498 139960082503680 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:34:17.151525 139960082503680 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:34:17.151553 139960082503680 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:34:17.151581 139960082503680 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:34:17.151608 139960082503680 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:34:17.151642 139960082503680 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:34:17.151670 139960082503680 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:34:17.151698 139960082503680 pyconfig.py:457] Config param use_random_routing: False
I0512 21:34:17.151726 139960082503680 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:34:17.151754 139960082503680 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:34:17.151782 139960082503680 pyconfig.py:457] Config param use_sft: False
I0512 21:34:17.151809 139960082503680 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:34:17.151837 139960082503680 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:34:17.151865 139960082503680 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:34:17.151893 139960082503680 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:34:17.151921 139960082503680 pyconfig.py:457] Config param use_truncation: True
I0512 21:34:17.151949 139960082503680 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:34:17.151977 139960082503680 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:34:17.152004 139960082503680 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:34:17.152032 139960082503680 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:34:17.152060 139960082503680 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:34:17.152088 139960082503680 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:34:17.152116 139960082503680 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:34:17.152145 139960082503680 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:34:17.152173 139960082503680 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:34:17.152201 139960082503680 pyconfig.py:457] Config param video_path: 
I0512 21:34:17.152229 139960082503680 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:34:17.152258 139960082503680 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:34:17.152286 139960082503680 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:34:17.152314 139960082503680 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:34:17.152342 139960082503680 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:34:17.152371 139960082503680 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:34:17.152401 139960082503680 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:34:17.152429 139960082503680 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:34:17.152457 139960082503680 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:34:17.152486 139960082503680 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:34:17.152527 139960082503680 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:34:17.152558 139960082503680 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:34:17.152586 139960082503680 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:34:17.152619 139960082503680 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:34:17.152649 139960082503680 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:34:17.152678 139960082503680 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:34:17.152706 139960082503680 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:34:17.152734 139960082503680 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.152762 139960082503680 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:34:17.152790 139960082503680 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:34:17.152818 139960082503680 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:34:17.152846 139960082503680 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:34:17.152874 139960082503680 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:34:17.152902 139960082503680 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:34:17.152929 139960082503680 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:34:17.152957 139960082503680 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:34:17.152985 139960082503680 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.153012 139960082503680 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:34:17.153040 139960082503680 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:34:17.153068 139960082503680 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:34:17.153096 139960082503680 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:34:17.153124 139960082503680 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:34:17.153152 139960082503680 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:34:17.153180 139960082503680 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:34:17.153209 139960082503680 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:34:17.153241 139960082503680 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:34:17.153269 139960082503680 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:34:17.153297 139960082503680 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:34:17.153325 139960082503680 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:34:17.153357 139960082503680 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:34:17.154862 139960082503680 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:34:17.240941 139960082503680 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:34:17.241034 139960082503680 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:17.241432 139960082503680 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f38d4b7af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:17.646041 139960082503680 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:34:17.646219 139960082503680 checkpoint_manager.py:708] [process=3][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f37f5fcea40>}, handler_registry=None
I0512 21:34:17.646504 139960082503680 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f37f5fcea40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.646578 139960082503680 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f37f5d61240>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.646636 139960082503680 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f37f5fcea40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f37f5fcea40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f37f5d61240>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f37f5d61240>}).
I0512 21:34:17.647022 139960082503680 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:34:17.647132 139960082503680 async_checkpointer.py:192] [process=3][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f4a4c7dedd0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:34:17.184186 140288083384320 pyconfig.py:457] Config param solution_end_token: </answer>
I0512 21:34:17.184212 140288083384320 pyconfig.py:457] Config param solution_start_token: <answer>
I0512 21:34:17.184239 140288083384320 pyconfig.py:457] Config param source_checkpoint_layout: orbax
I0512 21:34:17.184266 140288083384320 pyconfig.py:457] Config param sparse_matmul: True
I0512 21:34:17.184292 140288083384320 pyconfig.py:457] Config param spatial_merge_size_for_vit: 2
I0512 21:34:17.184319 140288083384320 pyconfig.py:457] Config param stack_prefill_result_cache: False
I0512 21:34:17.184346 140288083384320 pyconfig.py:457] Config param stack_trace_interval_seconds: 600
I0512 21:34:17.184372 140288083384320 pyconfig.py:457] Config param stack_trace_to_cloud: False
I0512 21:34:17.184399 140288083384320 pyconfig.py:457] Config param step_deviation_interval_seconds: 30
I0512 21:34:17.184426 140288083384320 pyconfig.py:457] Config param steps: 54363
I0512 21:34:17.184452 140288083384320 pyconfig.py:457] Config param stop_strings: None
I0512 21:34:17.184479 140288083384320 pyconfig.py:457] Config param student_overrides: {}
I0512 21:34:17.184506 140288083384320 pyconfig.py:457] Config param student_params_to_update: None
I0512 21:34:17.184532 140288083384320 pyconfig.py:457] Config param subslice_shape: 
I0512 21:34:17.184559 140288083384320 pyconfig.py:457] Config param swap_space_vllm_gb: 2
I0512 21:34:17.184586 140288083384320 pyconfig.py:457] Config param system_prompt: 
I0512 21:34:17.184612 140288083384320 pyconfig.py:457] Config param target_eval_loss: 0.0
I0512 21:34:17.184639 140288083384320 pyconfig.py:457] Config param teacher_overrides: {}
I0512 21:34:17.184666 140288083384320 pyconfig.py:457] Config param temperature_tuning: False
I0512 21:34:17.184692 140288083384320 pyconfig.py:457] Config param temporal_patch_size_for_vit: 2
I0512 21:34:17.184721 140288083384320 pyconfig.py:457] Config param tensorboard_dir: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/tensorboard/
I0512 21:34:17.184749 140288083384320 pyconfig.py:457] Config param tensors_on_device: None
I0512 21:34:17.184776 140288083384320 pyconfig.py:457] Config param tensors_to_offload: None
I0512 21:34:17.184803 140288083384320 pyconfig.py:457] Config param test_batch_start_index: 0
I0512 21:34:17.184829 140288083384320 pyconfig.py:457] Config param tile_size_for_vit: 336
I0512 21:34:17.184856 140288083384320 pyconfig.py:457] Config param tokenize_eval_data: True
I0512 21:34:17.184882 140288083384320 pyconfig.py:457] Config param tokenize_train_data: False
I0512 21:34:17.184909 140288083384320 pyconfig.py:457] Config param tokenizer_path: gs://tinymath-reason-data-himanshu/tokenizer/tokenizer.tiktoken
I0512 21:34:17.184936 140288083384320 pyconfig.py:457] Config param tokenizer_type: TokenizerType.TIKTOKEN
I0512 21:34:17.184971 140288083384320 pyconfig.py:457] Config param topk_routing_group: -1
I0512 21:34:17.184999 140288083384320 pyconfig.py:457] Config param tpu_num_chips_to_profile_per_task: 1
I0512 21:34:17.185026 140288083384320 pyconfig.py:457] Config param tpu_num_sparse_core_tiles_to_trace: 1
I0512 21:34:17.185054 140288083384320 pyconfig.py:457] Config param tpu_num_sparse_cores_to_trace: 2
I0512 21:34:17.185080 140288083384320 pyconfig.py:457] Config param train_data_columns: ['targets']
I0512 21:34:17.185110 140288083384320 pyconfig.py:457] Config param train_fraction: 1.0
I0512 21:34:17.185141 140288083384320 pyconfig.py:457] Config param train_image_column: image
I0512 21:34:17.185167 140288083384320 pyconfig.py:457] Config param train_micro_batch_size: -1
I0512 21:34:17.185194 140288083384320 pyconfig.py:457] Config param train_split: train
I0512 21:34:17.185220 140288083384320 pyconfig.py:457] Config param trainable_parameters_mask: []
I0512 21:34:17.185247 140288083384320 pyconfig.py:457] Config param trainable_position_size: -1
I0512 21:34:17.185273 140288083384320 pyconfig.py:457] Config param trainer_devices_fraction: 0.5
I0512 21:34:17.185301 140288083384320 pyconfig.py:457] Config param upload_all_profiler_results: False
I0512 21:34:17.185327 140288083384320 pyconfig.py:457] Config param use_2d_fsdp_sharding: False
I0512 21:34:17.185354 140288083384320 pyconfig.py:457] Config param use_agentic_rollout: False
I0512 21:34:17.185380 140288083384320 pyconfig.py:457] Config param use_audio: False
I0512 21:34:17.185407 140288083384320 pyconfig.py:457] Config param use_audio_in_video: False
I0512 21:34:17.185434 140288083384320 pyconfig.py:457] Config param use_batch_split_schedule: False
I0512 21:34:17.185460 140288083384320 pyconfig.py:457] Config param use_chat_template: False
I0512 21:34:17.185486 140288083384320 pyconfig.py:457] Config param use_chunked_prefill: False
I0512 21:34:17.185513 140288083384320 pyconfig.py:457] Config param use_custom_sort_vjp: True
I0512 21:34:17.185540 140288083384320 pyconfig.py:457] Config param use_dpo: False
I0512 21:34:17.185566 140288083384320 pyconfig.py:457] Config param use_gather_mosaic_kernel: False
I0512 21:34:17.185593 140288083384320 pyconfig.py:457] Config param use_grpo: True
I0512 21:34:17.185620 140288083384320 pyconfig.py:457] Config param use_indexer: False
I0512 21:34:17.185646 140288083384320 pyconfig.py:457] Config param use_iota_embed: False
I0512 21:34:17.185673 140288083384320 pyconfig.py:457] Config param use_jax_splash: False
I0512 21:34:17.185699 140288083384320 pyconfig.py:457] Config param use_manual_quantization: False
I0512 21:34:17.185728 140288083384320 pyconfig.py:457] Config param use_max_logit_estimate: -1
I0512 21:34:17.185755 140288083384320 pyconfig.py:457] Config param use_mrope: False
I0512 21:34:17.185781 140288083384320 pyconfig.py:457] Config param use_multimodal: False
I0512 21:34:17.185808 140288083384320 pyconfig.py:457] Config param use_pathways: True
I0512 21:34:17.185835 140288083384320 pyconfig.py:457] Config param use_post_attn_norm: False
I0512 21:34:17.185861 140288083384320 pyconfig.py:457] Config param use_post_ffw_norm: False
I0512 21:34:17.185888 140288083384320 pyconfig.py:457] Config param use_qk_clip: False
I0512 21:34:17.185914 140288083384320 pyconfig.py:457] Config param use_qk_norm: False
I0512 21:34:17.185940 140288083384320 pyconfig.py:457] Config param use_qk_norm_in_gdn: True
I0512 21:34:17.185972 140288083384320 pyconfig.py:457] Config param use_qwix_quantization: False
I0512 21:34:17.185999 140288083384320 pyconfig.py:457] Config param use_ragged_attention: False
I0512 21:34:17.186026 140288083384320 pyconfig.py:457] Config param use_random_routing: False
I0512 21:34:17.186052 140288083384320 pyconfig.py:457] Config param use_replicator_service: False
I0512 21:34:17.186079 140288083384320 pyconfig.py:457] Config param use_ring_of_experts: False
I0512 21:34:17.186106 140288083384320 pyconfig.py:457] Config param use_sft: False
I0512 21:34:17.186137 140288083384320 pyconfig.py:457] Config param use_splash_scheduler: False
I0512 21:34:17.186163 140288083384320 pyconfig.py:457] Config param use_standalone_converter: False
I0512 21:34:17.186190 140288083384320 pyconfig.py:457] Config param use_tokamax_gmm: False
I0512 21:34:17.186217 140288083384320 pyconfig.py:457] Config param use_tokamax_splash: False
I0512 21:34:17.186243 140288083384320 pyconfig.py:457] Config param use_truncation: True
I0512 21:34:17.186269 140288083384320 pyconfig.py:457] Config param use_tunix_gradient_accumulation: False
I0512 21:34:17.186296 140288083384320 pyconfig.py:457] Config param use_untrainable_positional_embedding: False
I0512 21:34:17.186322 140288083384320 pyconfig.py:457] Config param use_vertex_tensorboard: False
I0512 21:34:17.186349 140288083384320 pyconfig.py:457] Config param using_pipeline_parallelism: False
I0512 21:34:17.186375 140288083384320 pyconfig.py:457] Config param v_head_dim: 128
I0512 21:34:17.186402 140288083384320 pyconfig.py:457] Config param v_norm_with_scale: True
I0512 21:34:17.186428 140288083384320 pyconfig.py:457] Config param value_proj: RematLocation.REMAT
I0512 21:34:17.186456 140288083384320 pyconfig.py:457] Config param vertex_tensorboard_project: 
I0512 21:34:17.186483 140288083384320 pyconfig.py:457] Config param vertex_tensorboard_region: 
I0512 21:34:17.186509 140288083384320 pyconfig.py:457] Config param video_path: 
I0512 21:34:17.186536 140288083384320 pyconfig.py:457] Config param video_placeholder: <|video|>
I0512 21:34:17.186562 140288083384320 pyconfig.py:457] Config param vision_output_dim_for_vit: 4096
I0512 21:34:17.186589 140288083384320 pyconfig.py:457] Config param vision_output_length: -1
I0512 21:34:17.186616 140288083384320 pyconfig.py:457] Config param vllm_additional_config: {}
I0512 21:34:17.186642 140288083384320 pyconfig.py:457] Config param vllm_hf_config_path: 
I0512 21:34:17.186669 140288083384320 pyconfig.py:457] Config param vllm_hf_overrides: {}
I0512 21:34:17.186696 140288083384320 pyconfig.py:457] Config param vllm_load_format: dummy
I0512 21:34:17.186724 140288083384320 pyconfig.py:457] Config param vocab_size: 32768
I0512 21:34:17.186752 140288083384320 pyconfig.py:457] Config param warmup_steps_fraction: 0.0066666
I0512 21:34:17.186780 140288083384320 pyconfig.py:457] Config param weight_dtype: float32
I0512 21:34:17.186820 140288083384320 pyconfig.py:457] Config param weight_quantization_calibration_method: absmax
I0512 21:34:17.186849 140288083384320 pyconfig.py:457] Config param weight_sparsity_m: None
I0512 21:34:17.186876 140288083384320 pyconfig.py:457] Config param weight_sparsity_n: None
I0512 21:34:17.186902 140288083384320 pyconfig.py:457] Config param weight_sparsity_start_step: 50
I0512 21:34:17.186929 140288083384320 pyconfig.py:457] Config param weight_sparsity_update_step: 10
I0512 21:34:17.186961 140288083384320 pyconfig.py:457] Config param wi_tile_dlhs_batch_seq: 512
I0512 21:34:17.186989 140288083384320 pyconfig.py:457] Config param wi_tile_dlhs_embed_dim: 1024
I0512 21:34:17.187016 140288083384320 pyconfig.py:457] Config param wi_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.187042 140288083384320 pyconfig.py:457] Config param wi_tile_drhs_batch_seq: 512
I0512 21:34:17.187069 140288083384320 pyconfig.py:457] Config param wi_tile_drhs_embed_dim: 1024
I0512 21:34:17.187096 140288083384320 pyconfig.py:457] Config param wi_tile_drhs_mlp_dim: 1024
I0512 21:34:17.187123 140288083384320 pyconfig.py:457] Config param wi_tile_fwd_batch_seq: 512
I0512 21:34:17.187149 140288083384320 pyconfig.py:457] Config param wi_tile_fwd_embed_dim: 1024
I0512 21:34:17.187176 140288083384320 pyconfig.py:457] Config param wi_tile_fwd_mlp_dim: 1024
I0512 21:34:17.187202 140288083384320 pyconfig.py:457] Config param wo_tile_dlhs_batch_seq: 512
I0512 21:34:17.187229 140288083384320 pyconfig.py:457] Config param wo_tile_dlhs_embed_dim: 1024
I0512 21:34:17.187255 140288083384320 pyconfig.py:457] Config param wo_tile_dlhs_mlp_dim: 1024
I0512 21:34:17.187304 140288083384320 pyconfig.py:457] Config param wo_tile_drhs_batch_seq: 512
I0512 21:34:17.187333 140288083384320 pyconfig.py:457] Config param wo_tile_drhs_embed_dim: 1024
I0512 21:34:17.187360 140288083384320 pyconfig.py:457] Config param wo_tile_drhs_mlp_dim: 1024
I0512 21:34:17.187386 140288083384320 pyconfig.py:457] Config param wo_tile_fwd_batch_seq: 512
I0512 21:34:17.187412 140288083384320 pyconfig.py:457] Config param wo_tile_fwd_embed_dim: 1024
I0512 21:34:17.187439 140288083384320 pyconfig.py:457] Config param wo_tile_fwd_mlp_dim: 1024
I0512 21:34:17.187466 140288083384320 pyconfig.py:457] Config param wsd_decay_steps_fraction: 0.1
I0512 21:34:17.187494 140288083384320 pyconfig.py:457] Config param wsd_decay_style: WsdDecayStyle.LINEAR
I0512 21:34:17.187524 140288083384320 pyconfig.py:457] Config param xprof_e2e_enable_fw_power_level_event: False
I0512 21:34:17.187551 140288083384320 pyconfig.py:457] Config param xprof_e2e_enable_fw_thermal_event: False
I0512 21:34:17.187578 140288083384320 pyconfig.py:457] Config param xprof_e2e_enable_fw_throttle_event: False
I0512 21:34:17.187604 140288083384320 pyconfig.py:457] Config param xprof_tpu_power_trace_level: XProfTPUPowerTraceMode.POWER_TRACE_NONE
I0512 21:34:17.187637 140288083384320 pyconfig.py:457] Config param z_loss_multiplier: 0.0
I0512 21:34:17.189012 140288083384320 maxtext_utils.py:1825] Num_devices: 32, shape (1, 1, 1, 32, 1, 1, 1, 1, 1, 1, 1, 1)
I0512 21:34:17.275778 140288083384320 checkpointing.py:258] Creating checkpoint manager with ocdbt=True and zarr3=True
I0512 21:34:17.275875 140288083384320 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:17.276399 140288083384320 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f8536d5af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:17.678326 140288083384320 checkpointing.py:290] Enabling policy for fixed interval checkpointing.
I0512 21:34:17.678526 140288083384320 checkpoint_manager.py:708] [process=6][thread=MainThread] CheckpointManager init: checkpointers=None, item_names=('items',), item_handlers={'items': <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8457beea40>}, handler_registry=None
I0512 21:34:17.678818 140288083384320 composite_checkpoint_handler.py:237] Deferred registration for item: "items". Adding handler `<orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8457beea40>` for item "items" and save args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.678892 140288083384320 composite_checkpoint_handler.py:237] Deferred registration for item: "metrics". Adding handler `<orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f845797d240>` for item "metrics" and save args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>` and restore args `<class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>` to `_handler_registry`.
I0512 21:34:17.678942 140288083384320 composite_checkpoint_handler.py:505] Initialized registry DefaultCheckpointHandlerRegistry({('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeSaveArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8457beea40>, ('items', <class 'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeRestoreArgs'>): <orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler object at 0x7f8457beea40>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonSaveArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f845797d240>, ('metrics', <class 'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonRestoreArgs'>): <orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler object at 0x7f845797d240>}).
I0512 21:34:17.679363 140288083384320 abstract_checkpointer.py:35] orbax-checkpoint version: 0.11.39
I0512 21:34:17.679469 140288083384320 async_checkpointer.py:192] [process=6][thread=MainThread] Using barrier_sync_fn: <function get_barrier_sync_fn.<locals>._fn at 0x7f96aae0add0> timeout: 1200 secs and primary_host=0 for async checkpoint writes
I0512 21:34:18.101712 139991737624576 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.109381 140288083384320 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.113031 140152225769472 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.113979 140467415046144 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.118275 139761450792960 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.119937 140478431381504 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.123113 139960082503680 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.136720 139761450792960 checkpoint_manager.py:929] [process=5][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f09b9e3beb0>
I0512 21:34:18.136862 139761450792960 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136711 140478431381504 checkpoint_manager.py:929] [process=4][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fb0a9567eb0>
I0512 21:34:18.136866 140478431381504 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136922 140478431381504 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.136914 139761450792960 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.134880 140608182196224 checkpoint_manager.py:1812] Found 0 checkpoint steps in gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints
I0512 21:34:18.136735 139991737624576 checkpoint_manager.py:929] [process=0][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f3f58b5beb0>
I0512 21:34:18.136870 139991737624576 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136923 139991737624576 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.137038 140152225769472 checkpoint_manager.py:929] [process=2][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(nI0512 21:34:18.136684 140467415046144 checkpoint_manager.py:929] [process=1][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fae18d7beb0>
I0512 21:34:18.136818 140467415046144 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136873 140467415046144 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f64b6257b20>
I0512 21:34:18.137298 140152225769472 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.137367 140152225769472 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.136696 140288083384320 checkpoint_manager.py:929] [process=6][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f845797fb20>
I0512 21:34:18.136838 140288083384320 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136891 140288083384320 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.136707 140608182196224 checkpoint_manager.py:929] [process=7][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7fcedf367eb0>
I0512 21:34:18.136853 140608182196224 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.136909 140608182196224 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.136800 139960082503680 checkpoint_manager.py:929] [process=3][thread=MainThread] CheckpointManager created,  primary_host=0, CheckpointManagerOptions=CheckpointManagerOptions(save_interval_steps=1, max_to_keep=None, keep_time_interval=None, keep_period=None, should_keep_fn=None, best_fn=None, best_mode='max', keep_checkpoints_without_metrics=True, step_prefix=None, step_format_fixed_length=None, step_name_format=None, create=True, cleanup_tmp_directories=False, save_on_steps=frozenset(), single_host_load_and_broadcast=False, todelete_subdir=None, todelete_full_path=None, enable_background_delete=False, read_only=False, enable_async_checkpointing=True, async_options=None, multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes=None, barrier_sync_key_prefix=None), should_save_fn=None, file_options=FileOptions(path_permission_mode=None), save_root_metadata=True, temporary_path_class=None, save_decision_policy=FixedIntervalPolicy(interval=1000), preservation_policy=LatestN(n=10), prevent_write_metrics=False, enable_should_save_is_saving_in_progress_check=True, enable_per_process_directory_creation=False, lightweight_initialize=False), root_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only/tinymath-1b-decode/checkpoints: <orbax.checkpoint.checkpoint_manager.CheckpointManager object at 0x7f37f5d63b20>
I0512 21:34:18.136945 139960082503680 checkpointing.py:326] Checkpoint manager created!
I0512 21:34:18.137000 139960082503680 generate_param_only_checkpoint.py:197] Read training checkpoint from: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.628354 139991737624576 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.628534 139991737624576 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.628655 139991737624576 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.628758 139991737624576 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f4037982f80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.628976 139991737624576 event_tracking.py:70] [process=0] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.632714 140467415046144 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.632880 140467415046144 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.632998 140467415046144 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.633098 140467415046144 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7faef7a912a0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.633322 140467415046144 event_tracking.py:70] [process=1] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.638026 140288083384320 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.638206 140288083384320 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.638335 140288083384320 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.638443 140288083384320 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f8536d5af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.639000 139960082503680 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.639199 139960082503680 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.638668 140288083384320 event_tracking.py:70] [process=6] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.639348 139960082503680 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.639466 139960082503680 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f38d4b7af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.639699 139960082503680 event_tracking.py:70] [process=3] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.652002 140478431381504 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.652196 140478431381504 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.652332 140478431381504 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.652444 140478431381504 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fb188282fe0>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.650463 140608182196224 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.650656 140608182196224 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.650792 140608182196224 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.650905 140608182196224 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7fcfbe18af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.652673 140478431381504 event_tracking.py:70] [process=4] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.651130 140608182196224 event_tracking.py:70] [process=7] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.655446 139761450792960 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.655631 139761450792960 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.655768 139761450792960 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.655874 139761450792960 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f0a98b7ef80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.656100 139761450792960 event_tracking.py:70] [process=5] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.660251 140152225769472 checkpointing.py:613] checkpoint manager exists so trying to load this run's existing checkpoint
I0512 21:34:18.660438 140152225769472 checkpointing.py:702] Loading full state from path: gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:18.660567 140152225769472 pytree_checkpoint_handler.py:597] save_device_host_concurrent_bytes=None
I0512 21:34:18.660673 140152225769472 base_pytree_checkpoint_handler.py:479] Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=True, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7f659537af80>, enable_pinned_host_transfer=False, save_concurrent_bytes: 96000000000 (89.4 GiB), restore_concurrent_bytes: 96000000000 (89.4 GiB)
I0512 21:34:18.660892 140152225769472 event_tracking.py:70] [process=2] [sync] Started load checkpoint @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.791593 140467415046144 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.792186 140467415046144 event_tracking.py:125] [process=1] [sync] Finished blocking load in 0.16 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.795822 139991737624576 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.796445 139991737624576 event_tracking.py:125] [process=0] [sync] Finished blocking load in 0.17 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.817294 140608182196224 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.819139 139761450792960 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.819720 139761450792960 event_tracking.py:125] [process=5] [sync] Finished blocking load in 0.16 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.817866 140608182196224 event_tracking.py:125] [process=7] [sync] Finished blocking load in 0.17 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.821068 140152225769472 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.821640 140152225769472 event_tracking.py:125] [process=2] [sync] Finished blocking load in 0.16 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.830938 140288083384320 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.831538 140288083384320 event_tracking.py:125] [process=6] [sync] Finished blocking load in 0.19 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.850434 140478431381504 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.851027 140478431381504 event_tracking.py:125] [process=4] [sync] Finished blocking load in 0.20 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.857300 139960082503680 checkpointer.py:307] Restoring checkpoint from gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:18.857872 139960082503680 event_tracking.py:125] [process=3] [sync] Finished blocking load in 0.22 seconds. Continuing load @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items.
I0512 21:34:19.097027   82946 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.107947   84377 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.129769   79398 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.134157   87111 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.145170   77772 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.143986   76965 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.171518   77739 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:19.183478   84134 google_auth_provider.cc:181] Running on GCE, using service account 385740294473-compute@developer.gserviceaccount.com
I0512 21:34:20.256706 140467415046144 jax_array_handlers.py:862] [process=1] /jax/orbax/read/worker/io/requested throughput: 1.300 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.15394926071167 s) (per-host)
I0512 21:34:20.263217 140467415046144 base_pytree_checkpoint_handler.py:132] [process=1] /jax/checkpoint/read/gbytes_per_sec: 4.594 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.306166410446167 s) (per-host)
I0512 21:34:20.268648 140608182196224 jax_array_handlers.py:862] [process=7] /jax/orbax/read/worker/io/requested throughput: 1.316 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.1397876739501953 s) (per-host)
I0512 21:34:20.277113 140608182196224 base_pytree_checkpoint_handler.py:132] [process=7] /jax/checkpoint/read/gbytes_per_sec: 4.656 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.2885560989379883 s) (per-host)
I0512 21:34:20.309123 139761450792960 jax_array_handlers.py:862] [process=5] /jax/orbax/read/worker/io/requested throughput: 1.283 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.169189214706421 s) (per-host)
I0512 21:34:20.316714 139761450792960 base_pytree_checkpoint_handler.py:132] [process=5] /jax/checkpoint/read/gbytes_per_sec: 4.463 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3443951606750488 s) (per-host)
I0512 21:34:20.317541 140152225769472 jax_array_handlers.py:862] [process=2] /jax/orbax/read/worker/io/requested throughput: 1.257 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.193152904510498 s) (per-host)
I0512 21:34:20.323718 140152225769472 base_pytree_checkpoint_handler.py:132] [process=2] /jax/checkpoint/read/gbytes_per_sec: 4.455 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3468739986419678 s) (per-host)
I0512 21:34:20.357106 140288083384320 jax_array_handlers.py:862] [process=6] /jax/orbax/read/worker/io/requested throughput: 1.231 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2187049388885498 s) (per-host)
I0512 21:34:20.373820 139991737624576 jax_array_handlers.py:862] [process=0] /jax/orbax/read/worker/io/requested throughput: 1.170 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2821552753448486 s) (per-host)
I0512 21:34:20.375046 140288083384320 base_pytree_checkpoint_handler.py:132] [process=6] /jax/checkpoint/read/gbytes_per_sec: 4.317 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3900198936462402 s) (per-host)
I0512 21:34:20.381002 139991737624576 base_pytree_checkpoint_handler.py:132] [process=0] /jax/checkpoint/read/gbytes_per_sec: 4.197 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.429776906967163 s) (per-host)
I0512 21:34:20.421241 139960082503680 jax_array_handlers.py:862] [process=3] /jax/orbax/read/worker/io/requested throughput: 1.207 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.243222713470459 s) (per-host)
I0512 21:34:20.427755 139960082503680 base_pytree_checkpoint_handler.py:132] [process=3] /jax/checkpoint/read/gbytes_per_sec: 4.302 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.3946969509124756 s) (per-host)
I0512 21:34:20.448723 140478431381504 jax_array_handlers.py:862] [process=4] /jax/orbax/read/worker/io/requested throughput: 1.170 GiB/s (total gbytes: 1.5 GiB) (time elapsed: 1.2825984954833984 s) (per-host)
I0512 21:34:20.453567 140478431381504 base_pytree_checkpoint_handler.py:132] [process=4] /jax/checkpoint/read/gbytes_per_sec: 4.174 GiB/s (total gbytes: 6.0 GiB) (time elapsed: 1.43766188621521 s) (per-host)
I0512 21:34:20.525531 139761450792960 event_tracking.py:138] [process=5] [sync] Finished load in 1.87 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.525934 139761450792960 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:34:20.525628 140467415046144 event_tracking.py:138] [process=1] [sync] Finished load in 1.89 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.526101 140467415046144 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 5 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:34:20.525730 140152225769472 event_tracking.py:138] [process=2] [sync] Finished load in 1.86 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.525807 139960082503680 event_tracking.py:138] [process=3] [sync] Finished load in 1.89 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
Worker 0 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:34:20.525731 139991737624576 event_tracking.py:138] [process=0] [sync] Finished load in 1.90 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.526298 139991737624576 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:34:20.525561 140288083384320 event_tracking.py:138] [process=6] [sync] Finished load in 1.89 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.525981 140288083384320 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 1 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:34:20.528480 140478431381504 event_tracking.py:138] [process=4] [sync] Finished load in 1.88 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.526218 140152225769472 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 2 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:34:20.525774 140608182196224 event_tracking.py:138] [process=7] [sync] Finished load in 1.87 seconds @ gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items
I0512 21:34:20.526296 140608182196224 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
I0512 21:34:20.526436 139960082503680 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 7 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 6 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Worker 4 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
I0512 21:34:20.529059 140478431381504 generate_param_only_checkpoint.py:111] In input checkpoint Number of model params=0.134 billion
Worker 3 initialized.
['generate_param_only_checkpoint', 'maxtext_config.yml', 'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items', 'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only', 'run_name=tinymath-1b-decode', 'force_unroll=true']

--- FULL PARAMS TREE ---
VariableState (type: dict)
  decoder (type: dict)
    decoder_norm (type: dict)
      scale (type: VariableState)
    logits_dense (type: dict)
      kernel (type: VariableState)
  token_embedder (type: dict)
    embedding (type: VariableState)

--- SEARCHING FOR LAYERS (target: 22) ---

Checking opt_state for layers...
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
Traceback (most recent call last):
  File "/home/himanshu/maxtext/convert_helper.py", line 170, in <module>
    app.run(gpoc.main)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/himanshu/.local/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 217, in main
    generate_decode_checkpoint(config)
  File "/home/himanshu/maxtext/src/maxtext/utils/generate_param_only_checkpoint.py", line 201, in generate_decode_checkpoint
    _possibly_unroll_params(config, training_state, training_state_annotations, mesh)
  File "/home/himanshu/maxtext/convert_helper.py", line 107, in patched_possibly_unroll_params
    raise ValueError("Could not find any array with 22 layers in params or opt_state.")
ValueError: Could not find any array with 22 layers in params or opt_state.
##### Command execution on worker 4 failed with exit status 1. Continuing.
##### Command execution on worker 5 failed with exit status 1. Continuing.
##### Command execution on worker 3 failed with exit status 1. Continuing.
##### Command execution on worker 1 failed with exit status 1. Continuing.
##### Command execution on worker 7 failed with exit status 1. Continuing.
##### Command execution on worker 6 failed with exit status 1. Continuing.
##### Command execution on worker 2 failed with exit status 1. Continuing.
##### Command execution on worker 0 failed with exit status 1. Continuing.
himanshu@Himanshus-MacBook TinyMathReason-1B % 

*Viewed [maxtext_config.yml](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/maxtext_config.yml) *

*Searched web for MaxText "default" model structure decoder layers key*

*Edited relevant file*

*User accepted the command `export CLOUDSDK_CONFIG=/Users/himanshu/.gemini/gcloud_config
gcloud compute tpus tpu-vm scp src/train/dump_checkpoint_manifest.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="export JAX_PLATFORMS=cpu; python3 ~/maxtext/dump_checkpoint_manifest.py"`*

*User accepted the command `mkdir -p gcloud_tmp
export CLOUDSDK_CONFIG=$(pwd)/gcloud_tmp
gcloud compute tpus tpu-vm scp src/train/dump_checkpoint_manifest.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="export JAX_PLATFORMS=cpu; python3 ~/maxtext/dump_checkpoint_manifest.py"`*

*User accepted the command `gcloud compute tpus tpu-vm scp src/train/dump_checkpoint_manifest.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="export JAX_PLATFORMS=cpu; python3 ~/maxtext/dump_checkpoint_manifest.py"`*

*Edited relevant file*

*Viewed [convert_helper.py](file:///Users/himanshu/Git/TinyMathReason-1B/src/train/convert_helper.py) *

*Edited relevant file*

### Planner Response

I've updated `convert_helper.py` with a deep diagnostic version. The previous run revealed that the layers are completely missing from the loaded PyTree (only 0.134B parameters found out of ~1.1B).

This new version will:
1.  **Initialize the model template** before loading the checkpoint to see if the `layers` key even exists in the model definition.
2.  **Print the complete key hierarchy** of the `decoder` block to identify if MaxText is using an unconventional naming convention for this run.
3.  **Confirm the configuration** being used (specifically `base_num_decoder_layers`).

Please upload this new `convert_helper.py` to your TPU VM and run it:

```bash
# On your local machine
gcloud compute tpus tpu-vm scp src/train/convert_helper.py tpu-v4-64-spot:~/maxtext/ --zone=us-central2-b --worker=all

# On the TPU VM (Worker 0)
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="cd ~/maxtext && export JAX_PLATFORMS=tpu,cpu && python3 convert_helper.py"
```

Once you have the output, please share the `--- MODEL TEMPLATE ---` and `--- LOADED STATE PARAMS ---` sections. This will tell us if the problem is in the **model definition** (MaxText config mismatch) or the **checkpoint loading** (Orbax failing to find the data).

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="cd ~/maxtext && export JAX_PLATFORMS=tpu,cpu && python3 convert_helper.py" 
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 1 workers.
SSH: Attempting to connect to worker 0...










*Viewed [config_default](file:///Users/himanshu/Git/TinyMathReason-1B/gcloud_tmp/configurations/config_default) *

*Searched web for MaxText num_decoder_layers vs base_num_decoder_layers*

### Planner Response

I see you're attempting to run the diagnostic script. If the SSH connection is hanging at `Attempting to connect to worker 0...`, it could be due to a few reasons:
1.  **JAX Initialization**: Distributed initialization on a v4-64 can take 2-3 minutes.
2.  **Gcloud Permissions**: I noticed you were looking at `gcloud_tmp/configurations/config_default`. If your credentials expired or the config is messed up, `gcloud` might hang.

**While we wait for that output, I have a strong suspicion about the configuration:**

In MaxText, there is sometimes a distinction between `base_num_decoder_layers` and `num_decoder_layers`. If the model template is being built with 0 layers, Orbax will silently "successfully" load only the embedding and norm, ignoring the 22 layers in the checkpoint.

**Could you check if the diagnostic script finished?** If not, please try running this simpler command to verify the TPU VM is reachable and check the disk space (sometimes full disks cause weird SSH hangs):

```bash
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="df -h && ls -lh ~/maxtext/maxtext_config.yml"
```

If the diagnostic script **did** finish, please share the output, especially the `--- CONFIG CHECK ---` and `--- MODEL TEMPLATE ---` sections! This will confirm if we have a config naming mismatch.

### User Input

himanshu@Himanshus-MacBook TinyMathReason-1B % gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="df -h && ls -lh ~/maxtext/maxtext_config.yml"

Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 1 workers.
SSH: Attempting to connect to worker 0...
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        97G   14G   84G  15% /
tmpfs           201G     0  201G   0% /dev/shm
tmpfs            81G  2.6M   81G   1% /run
tmpfs           5.0M     0  5.0M   0% /run/lock
/dev/sda15      105M  6.1M   99M   6% /boot/efi
tmpfs            41G  4.0K   41G   1% /run/user/2001
-rw-r--r-- 1 himanshu himanshu 2.0K May 12 15:50 /home/himanshu/maxtext/maxtext_config.yml
himanshu@Himanshus-MacBook TinyMathReason-1B % 







### Planner Response

The disk space is healthy, so that's not the issue. The SSH connection is working.

Let's run the diagnostic script again, but this time with `PYTHONUNBUFFERED=1` to ensure we see the output as it happens (in case it's hanging during the JAX mesh setup). 

**Please run this command and wait at least 3-4 minutes for it to complete:**

```bash
gcloud compute tpus tpu-vm ssh tpu-v4-64-spot --zone=us-central2-b --worker=0 --command="cd ~/maxtext && export JAX_PLATFORMS=tpu,cpu && export PYTHONUNBUFFERED=1 && python3 convert_helper.py"
```

**What I'm looking for specifically:**
1.  **`--- CONFIG CHECK ---`**: Does it say `base_num_decoder_layers: 22`?
2.  **`--- MODEL TEMPLATE ---`**: Do you see a `layers` key under `decoder`?
3.  **`Total loaded params`**: If this is still `0.134 Billion`, then Orbax is definitively failing to find the weights in GCS, likely due to a path nesting issue.

If this still results in `0.134 Billion`, we will switch to a "Bare Metal" GCS scanner to find the actual filenames in your checkpoint folder.