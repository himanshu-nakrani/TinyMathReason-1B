"""
Convert a MaxText/Orbax checkpoint to HuggingFace LlamaForCausalLM format.

TinyMathReason-1B: MaxText (JAX) → HuggingFace (PyTorch safetensors)

This handles the critical structural differences:
  1. MaxText uses scanned/stacked layers (all layers in one array dimension)
  2. MaxText bakes 1/sqrt(head_dim) scaling into query weights
  3. MaxText permutes Q/K weights for its RoPE implementation
  4. Vocab size is 32768 (padded from 32000 for FSDP alignment)

Usage:
    # Step 1: ALWAYS inspect the checkpoint first
    python src/train/inspect_checkpoint.py \\
        --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

    # Step 2: Run conversion
    python src/train/convert_checkpoint.py \\
        --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362 \\
        --hf_out_dir ./hf_model \\
        --tokenizer_path ./tokenizer

Based on the official MaxText conversion:
  https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/standalone_scripts/llama_mistral_mixtral_orbax_to_hf.py
"""
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Architecture constants (must match maxtext_config.yml) ──────────────────
VOCAB_SIZE = 32768          # Padded from 32000 for FSDP alignment
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128     # SwiGLU
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 4     # Query heads
NUM_KV_HEADS = 1            # GQA 4:1
HEAD_DIM = 16               # HIDDEN_SIZE // NUM_ATTENTION_HEADS
MAX_POSITION_EMBEDDINGS = 4096
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0


def reverse_query_scale(arr, head_dim):
    """
    MaxText bakes 1/sqrt(head_dim) scaling into query weights during training.
    We must reverse this when converting to HuggingFace format.

    In MaxText's attention: attn = dot(Q, K) (no explicit /sqrt(d) because it's in Q)
    In HuggingFace: attn = dot(Q, K) / sqrt(d) (explicit scaling)

    So: Q_hf = Q_maxtext * sqrt(head_dim)
    """
    return arr * np.sqrt(head_dim)


def unpermute_from_maxtext_rope(arr, num_heads, head_dim):
    """
    Reverse the permutation that MaxText applies to Q/K weights for RoPE.

    MaxText interleaves the real and imaginary parts differently than HuggingFace.
    MaxText stores weights as [d_model, heads, head_dim] where head_dim is ordered as:
      [re_0, re_1, ..., re_{d/2-1}, im_0, im_1, ..., im_{d/2-1}]

    HuggingFace expects the standard interleaved order:
      [re_0, im_0, re_1, im_1, ..., re_{d/2-1}, im_{d/2-1}]

    This function reverses that permutation.
    """
    # arr shape: [d_model, num_heads, head_dim] after extracting from stacked
    # Reshape to separate real/imaginary halves
    half = head_dim // 2

    # Reshape: [d_model, num_heads, 2, half] where dim2=0 is real, dim2=1 is imaginary
    reshaped = arr.reshape(arr.shape[0], num_heads, 2, half)

    # Interleave: [re_0, im_0, re_1, im_1, ...]
    # Transpose the last two dims and flatten
    interleaved = np.stack([reshaped[:, :, 0, :], reshaped[:, :, 1, :]], axis=-1)
    return interleaved.reshape(arr.shape[0], num_heads, head_dim)


def navigate_params(ckpt):
    """
    MaxText checkpoints have different nesting depending on how they are saved.
    This safely navigates to the core params dict.
    Usually:
      - ckpt['params']['params'][...]  (training checkpoint)
      - ckpt['params'][...]            (param-only checkpoint)
      - ckpt[...]                      (raw params)
    """
    if isinstance(ckpt, dict):
        if 'params' in ckpt:
            inner = ckpt['params']
            # Check for double-nesting: training checkpoints have params.params
            if isinstance(inner, dict) and 'params' in inner:
                logger.info("Checkpoint structure: ckpt['params']['params'][...]")
                inner = inner['params']
            else:
                logger.info("Checkpoint structure: ckpt['params'][...]")
                
        else:
            logger.info("Checkpoint structure: flat (using ckpt directly)")
            inner = ckpt
            
        if isinstance(inner, dict) and 'VariableState' in inner:
            logger.info("Unnesting VariableState...")
            return inner['VariableState']
        return inner
    return ckpt


def load_orbax_with_tensorstore(orbax_dir):
    """Bypasses Orbax and loads the checkpoint directly using TensorStore.
    This avoids Topology mismatch errors when loading TPU checkpoints on CPU.
    """
    import tensorstore as ts
    if not orbax_dir.endswith("items") and not orbax_dir.endswith("items/"):
        items_dir = orbax_dir.rstrip("/") + "/items"
    else:
        items_dir = orbax_dir

    if items_dir.startswith("gs://"):
        kvstore_spec = {'driver': 'ocdbt', 'base': items_dir}
    else:
        kvstore_spec = {'driver': 'ocdbt', 'base': f'file://{items_dir}'}

    logger.info(f"Opening TensorStore KvStore at: {kvstore_spec['base']}")
    kvs = ts.KvStore.open(kvstore_spec).result()
    keys = kvs.list().result()
    
    ckpt = {}
    array_count = 0
    for k in keys:
        k_str = k.decode()
        if k_str.endswith("zarr.json") or k_str.endswith(".zarray"):
            arr_path = k_str.replace("/zarr.json", "").replace("/.zarray", "")
            if arr_path in ("step",) or "opt_state" in arr_path:
                continue # Skip optimizer state
                
            driver_name = 'zarr3' if k_str.endswith("zarr.json") else 'zarr'
            dataset = ts.open({
                'driver': driver_name,
                'kvstore': kvstore_spec,
                'path': arr_path
            }).result()
            
            arr = dataset.read().result()
            parts = arr_path.split('.')
            current = ckpt
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = arr
            array_count += 1
            
    if array_count == 0:
        raise FileNotFoundError(f"No arrays found at {kvstore_spec['base']}. Please check if the checkpoint step number (e.g., 1 vs 2) and path are correct.")
        
    logger.info(f"Loaded {array_count} arrays via TensorStore.")
    return ckpt


def convert_checkpoint(orbax_dir: str, hf_out_dir: str, tokenizer_path: str,
                        skip_rope_unpermute: bool = False,
                        skip_query_scale: bool = False):
    """
    Convert a MaxText/Orbax checkpoint to HuggingFace LlamaForCausalLM format.

    The key insight is that MaxText stores layer parameters in a STACKED format:
      params['decoder']['layers']['self_attention']['query']['kernel']
    has shape [hidden, num_layers, heads, head_dim] - NOT per-layer keys.
    """
    import jax
    import torch
    from orbax import checkpoint as ocp
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    # ── Step 1: Load Orbax checkpoint ──────────────────────────────────────
    logger.info(f"Loading Orbax checkpoint from: {orbax_dir}")
    try:
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt = checkpointer.restore(orbax_dir)
    except Exception as e:
        logger.warning(f"PyTreeCheckpointer failed ({e}), trying StandardCheckpointer...")
        try:
            checkpointer = ocp.StandardCheckpointer()
            ckpt = checkpointer.restore(orbax_dir)
        except Exception as e2:
            logger.warning(f"StandardCheckpointer failed ({e2}). Using TensorStore directly to bypass topology mismatch...")
            ckpt = load_orbax_with_tensorstore(orbax_dir)

    params = navigate_params(ckpt)
    logger.info(f"Top-level param keys: {sorted(params.keys())}")

    # ── Step 2: Validate checkpoint structure ──────────────────────────────
    assert 'token_embedder' in params, \
        f"Expected 'token_embedder' in params, got: {sorted(params.keys())}"
    assert 'decoder' in params, \
        f"Expected 'decoder' in params, got: {sorted(params.keys())}"

    decoder = params['decoder']
    # Verify layer structure
    if 'layers' in decoder:
        layers = decoder['layers']
    else:
        # For scan_layers=False, layers_0, layers_1 etc are directly inside decoder
        layers = decoder
        
    logger.info(f"Layers/Decoder keys: {sorted(layers.keys())}")

    # Sample a weight to confirm the stacking dimension
    if 'self_attention' in layers:
        q_kernel = np.asarray(layers['self_attention']['query']['kernel'])
    else:
        q_kernel = np.asarray(layers['layers_0']['self_attention']['query']['kernel'])
        
    logger.info(f"Query kernel shape: {q_kernel.shape}")
    logger.info(f"Expected: [hidden={HIDDEN_SIZE}, num_layers={NUM_HIDDEN_LAYERS}, "
                f"heads={NUM_ATTENTION_HEADS}, head_dim={HEAD_DIM}]")

    # ── Step 3: Create HuggingFace model config ───────────────────────────
    logger.info("Creating HuggingFace LlamaConfig...")
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )

    # ── Step 4: Map weights ───────────────────────────────────────────────
    logger.info("Mapping MaxText weights to HuggingFace format...")
    hf_state_dict = {}

    def to_torch(arr, dtype=torch.bfloat16):
        """Convert numpy/jax array to PyTorch tensor."""
        return torch.tensor(np.asarray(arr), dtype=dtype)

    # --- Embedding ---
    embedding = np.asarray(params['token_embedder']['embedding'])
    logger.info(f"  Embedding: {embedding.shape} (expected [{VOCAB_SIZE}, {HIDDEN_SIZE}])")
    hf_state_dict["model.embed_tokens.weight"] = to_torch(embedding)

    # --- Final LayerNorm ---
    decoder_norm = np.asarray(decoder['decoder_norm']['scale'])
    logger.info(f"  Decoder norm: {decoder_norm.shape}")
    hf_state_dict["model.norm.weight"] = to_torch(decoder_norm.reshape(HIDDEN_SIZE))

    # --- LM Head ---
    logits_kernel = np.asarray(decoder['logits_dense']['kernel'])
    logger.info(f"  Logits dense: {logits_kernel.shape} (expected [{HIDDEN_SIZE}, {VOCAB_SIZE}])")
    hf_state_dict["lm_head.weight"] = to_torch(logits_kernel.T)

    # --- Layer-by-layer extraction ---
    is_scanned = 'self_attention' in layers
    logger.info(f"Detected scan_layers={is_scanned}")
    
    if is_scanned:
        sa_global = layers['self_attention']
        mlp_global = layers['mlp']
        pre_attn_norm_key = None
        post_attn_norm_key = None
        for candidate in ['pre_self_attention_layer_norm', 'pre_self_attention_norm']:
            if candidate in layers:
                pre_attn_norm_key = candidate
                break
        for candidate in ['post_self_attention_layer_norm', 'pre_ffw_norm', 'post_self_attention_norm']:
            if candidate in layers:
                post_attn_norm_key = candidate
                break

    for layer_idx in range(NUM_HIDDEN_LAYERS):
        prefix = f"model.layers.{layer_idx}"
        logger.info(f"  Converting layer {layer_idx}/{NUM_HIDDEN_LAYERS}...")
        
        if is_scanned:
            wq = np.asarray(sa_global['query']['kernel'])[:, layer_idx, :, :]
            wk = np.asarray(sa_global['key']['kernel'])[:, layer_idx, :, :]
            wv = np.asarray(sa_global['value']['kernel'])[:, layer_idx, :, :]
            wo = np.asarray(sa_global['out']['kernel'])[:, layer_idx, :, :]
            
            wi_0 = np.asarray(mlp_global['wi_0']['kernel'])[:, layer_idx, :]
            wi_1 = np.asarray(mlp_global['wi_1']['kernel'])[:, layer_idx, :]
            wo_mlp = np.asarray(mlp_global['wo']['kernel'])[:, layer_idx, :]
            
            pre_attn_norm = np.asarray(layers[pre_attn_norm_key]['scale'])[:, layer_idx]
            post_attn_norm = np.asarray(layers[post_attn_norm_key]['scale'])[:, layer_idx]
        else:
            layer_dict = layers[f'layers_{layer_idx}']
            sa_local = layer_dict['self_attention']
            mlp_local = layer_dict['mlp']
            
            wq = np.asarray(sa_local['query']['kernel'])
            wk = np.asarray(sa_local['key']['kernel'])
            wv = np.asarray(sa_local['value']['kernel'])
            wo = np.asarray(sa_local['out']['kernel'])
            
            wi_0 = np.asarray(mlp_local['wi_0']['kernel'])
            wi_1 = np.asarray(mlp_local['wi_1']['kernel'])
            wo_mlp = np.asarray(mlp_local['wo']['kernel'])
            
            pre_attn_norm_key = None
            post_attn_norm_key = None
            for candidate in ['pre_self_attention_layer_norm', 'pre_self_attention_norm']:
                if candidate in layer_dict:
                    pre_attn_norm_key = candidate
                    break
            for candidate in ['post_self_attention_layer_norm', 'pre_ffw_norm', 'post_self_attention_norm']:
                if candidate in layer_dict:
                    post_attn_norm_key = candidate
                    break
            pre_attn_norm = np.asarray(layer_dict[pre_attn_norm_key]['scale'])
            post_attn_norm = np.asarray(layer_dict[post_attn_norm_key]['scale'])

        # --- Attention Norms ---
        hf_state_dict[f"{prefix}.input_layernorm.weight"] = to_torch(
            pre_attn_norm.reshape(HIDDEN_SIZE)
        )
        hf_state_dict[f"{prefix}.post_attention_layernorm.weight"] = to_torch(
            post_attn_norm.reshape(HIDDEN_SIZE)
        )

        # --- Query projection ---
        if not skip_query_scale:
            wq = reverse_query_scale(wq, HEAD_DIM)

        if not skip_rope_unpermute:
            wq = unpermute_from_maxtext_rope(wq, NUM_ATTENTION_HEADS, HEAD_DIM)

        hf_state_dict[f"{prefix}.self_attn.q_proj.weight"] = to_torch(
            wq.reshape(HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM).T
        )

        # --- Key projection ---
        if not skip_rope_unpermute:
            wk = unpermute_from_maxtext_rope(wk, NUM_KV_HEADS, HEAD_DIM)

        hf_state_dict[f"{prefix}.self_attn.k_proj.weight"] = to_torch(
            wk.reshape(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM).T
        )

        # --- Value projection ---
        hf_state_dict[f"{prefix}.self_attn.v_proj.weight"] = to_torch(
            wv.reshape(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM).T
        )

        # --- Output projection ---
        hf_state_dict[f"{prefix}.self_attn.o_proj.weight"] = to_torch(
            wo.reshape(NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE).T
        )

        # --- MLP ---
        hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = to_torch(wi_0.T)
        hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = to_torch(wi_1.T)
        hf_state_dict[f"{prefix}.mlp.down_proj.weight"] = to_torch(wo_mlp.T)

    # ── Step 5: Create model and load weights ─────────────────────────────
    logger.info("Creating HuggingFace LlamaForCausalLM and loading weights...")
    with torch.device("meta"):
        hf_model = LlamaForCausalLM(config)
    hf_model = hf_model.to_empty(device="cpu")

    # Validate shapes before loading
    model_state = hf_model.state_dict()
    for key in model_state:
        if key not in hf_state_dict:
            logger.error(f"  MISSING: {key} (expected shape: {model_state[key].shape})")
        elif model_state[key].shape != hf_state_dict[key].shape:
            logger.error(f"  SHAPE MISMATCH: {key}: "
                        f"model={model_state[key].shape} vs converted={hf_state_dict[key].shape}")

    hf_model.load_state_dict(hf_state_dict, strict=True)
    logger.info("✅ All weights loaded successfully!")

    # ── Step 6: Verify with forward pass ──────────────────────────────────
    logger.info("Running verification forward pass...")
    test_input = torch.randint(0, VOCAB_SIZE, (1, 16))
    with torch.no_grad():
        output = hf_model(test_input)
    assert output.logits.shape == (1, 16, VOCAB_SIZE), \
        f"Forward pass shape mismatch: {output.logits.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(output.logits).any(), "Forward pass produced NaN!"
    assert not torch.isinf(output.logits).any(), "Forward pass produced Inf!"
    logger.info(f"✅ Forward pass OK: logits shape={output.logits.shape}, "
                f"logits range=[{output.logits.min():.4f}, {output.logits.max():.4f}]")

    # ── Step 7: Save model ────────────────────────────────────────────────
    out_path = Path(hf_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    import jax
    if jax.process_index() == 0:
        logger.info(f"Saving HF model to: {hf_out_dir}")
        hf_model.save_pretrained(hf_out_dir, safe_serialization=True)
    else:
        logger.info(f"Process {jax.process_index()} skipping HF model save.")

    # ── Step 8: Save tokenizer ────────────────────────────────────────────
    logger.info(f"Copying tokenizer from: {tokenizer_path}")
    tok_path = Path(tokenizer_path)

    if (tok_path / "tokenizer.json").exists():
        # Load tokenizer using the tokenizers library directly
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tok_path / "tokenizer.json"),
            bos_token="<|bos|>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            pad_token="<|pad|>",
            model_max_length=MAX_POSITION_EMBEDDINGS,
        )
        if jax.process_index() == 0:
            tokenizer.save_pretrained(hf_out_dir)
            logger.info(f"✅ Tokenizer saved (vocab_size={tokenizer.vocab_size})")
    else:
        logger.warning(f"No tokenizer.json found in {tokenizer_path}. "
                       "Skipping tokenizer save — you'll need to add it manually.")

    # ── Step 9: Save model card ───────────────────────────────────────────
    if jax.process_index() == 0:
        model_card = """---
language: en
license: apache-2.0
tags:
- math
- reasoning
- llama
- tinymath
library_name: transformers
pipeline_tag: text-generation
---

# TinyMathReason-1B (Base)

A 1.07B parameter language model trained from scratch for mathematical reasoning.

## Architecture
- **Type:** Llama-2 style decoder-only transformer
- **Parameters:** ~1.07B
- **Hidden dim:** 2048
- **Layers:** 22
- **Attention:** GQA (16 query heads, 4 KV heads)
- **MLP:** SwiGLU (intermediate dim 5632)
- **Context length:** 4096
- **Vocab size:** 32,768 (custom BPE tokenizer)

## Training
- **Framework:** MaxText (JAX) on TPU v4-64
- **Data:** ~57B tokens (FineWeb-Edu, MathPile, OpenWebMath, Stack-Edu)
- **Optimizer:** AdamW (lr=3e-4, cosine decay)
- **Precision:** bfloat16

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/hf_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/hf_model")

inputs = tokenizer("The sum of 2 and 3 is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```
"""
        (out_path / "README.md").write_text(model_card)

        logger.info("=" * 60)
        logger.info("✅ CONVERSION COMPLETE!")
        logger.info(f"   Output directory: {hf_out_dir}")
        logger.info(f"   Model config: {out_path / 'config.json'}")
        logger.info(f"   Model weights: {out_path / 'model.safetensors'}")
        logger.info(f"   Tokenizer: {out_path / 'tokenizer.json'}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MaxText Orbax checkpoint to HuggingFace format"
    )
    parser.add_argument("--orbax_dir", type=str, required=True,
                        help="Path to MaxText Orbax checkpoint (GCS or local)")
    parser.add_argument("--hf_out_dir", type=str, required=True,
                        help="Path to save HuggingFace model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to tokenizer directory (must contain tokenizer.json)")
    parser.add_argument("--skip_rope_unpermute", action="store_true",
                        help="Skip RoPE weight unpermutation (use if unsure about MaxText's RoPE layout)")
    parser.add_argument("--skip_query_scale", action="store_true",
                        help="Skip query weight scale reversal (use if MaxText config doesn't bake scaling)")
    args = parser.parse_args()

    convert_checkpoint(
        orbax_dir=args.orbax_dir,
        hf_out_dir=args.hf_out_dir,
        tokenizer_path=args.tokenizer_path,
        skip_rope_unpermute=args.skip_rope_unpermute,
        skip_query_scale=args.skip_query_scale,
    )
