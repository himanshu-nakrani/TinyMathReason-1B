import argparse
import logging
from pathlib import Path
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def convert_checkpoint(orbax_dir: str, hf_out_dir: str, tokenizer_path: str):
    """
    Converts a MaxText/Orbax checkpoint into a HuggingFace LlamaForCausalLM format.
    Our TinyMathReason-1B architecture is a standard Llama-style decoder.
    """
    logging.info(f"Loading Orbax checkpoint from {orbax_dir}")
    
    # Initialize Orbax Checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Read the tree structure and arrays
    # MaxText typically saves under a structure like: params -> {layers, embed, ...}
    ckpt_data = checkpointer.restore(orbax_dir)
    if 'params' in ckpt_data:
        params = ckpt_data['params']
    else:
        params = ckpt_data
        
    logging.info("Initializing HuggingFace Llama model...")
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    
    # Create empty HF model
    with torch.device("meta"):
        hf_model = LlamaForCausalLM(config)
    hf_model = hf_model.to_empty(device="cpu")
    
    hf_state_dict = hf_model.state_dict()
    
    logging.info("Mapping weights...")
    # Map MaxText JAX arrays to PyTorch tensors
    # NOTE: Exact mapping depends on MaxText's internal naming. 
    # This is the standard correspondence for Llama-style models in MaxText.
    
    def pt(jax_arr):
        # Convert JAX/Numpy array to PyTorch tensor
        return torch.from_numpy(jax_arr).to(torch.bfloat16)

    try:
        # Embeddings
        hf_state_dict["model.embed_tokens.weight"] = pt(params['token_embedder']['embedding'])
        
        # Final Norm
        hf_state_dict["model.norm.weight"] = pt(params['decoder']['decoder_norm']['scale'])
        
        # LM Head
        if not config.tie_word_embeddings:
            hf_state_dict["lm_head.weight"] = pt(params['decoder']['logits_dense']['kernel']).t()
            
        # Layers
        for i in range(config.num_hidden_layers):
            layer_prefix = f"decoder.layers_{i}"
            hf_prefix = f"model.layers.{i}"
            
            # Attention Norm
            hf_state_dict[f"{hf_prefix}.input_layernorm.weight"] = pt(params['decoder'][f'layers_{i}']['pre_self_attention_norm']['scale'])
            
            # Attention Projections
            # MaxText often stores Q/K/V separately or combined. Assume separated here.
            wq = params['decoder'][f'layers_{i}']['self_attention']['query']['kernel']
            wk = params['decoder'][f'layers_{i}']['self_attention']['key']['kernel']
            wv = params['decoder'][f'layers_{i}']['self_attention']['value']['kernel']
            wo = params['decoder'][f'layers_{i}']['self_attention']['out']['kernel']
            
            hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"] = pt(wq).t()
            hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"] = pt(wk).t()
            hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"] = pt(wv).t()
            hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"] = pt(wo).t()
            
            # MLP Norm
            hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"] = pt(params['decoder'][f'layers_{i}']['pre_ffw_norm']['scale'])
            
            # MLP Projections (SwiGLU)
            w1 = params['decoder'][f'layers_{i}']['mlp']['wi_0']['kernel'] # gate
            w3 = params['decoder'][f'layers_{i}']['mlp']['wi_1']['kernel'] # up
            w2 = params['decoder'][f'layers_{i}']['mlp']['wo']['kernel']   # down
            
            hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"] = pt(w1).t()
            hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"] = pt(w3).t()
            hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"] = pt(w2).t()
            
        # Load state dict
        hf_model.load_state_dict(hf_state_dict)
        logging.info("Weights mapped successfully.")
        
    except KeyError as e:
        logging.error(f"KeyError during mapping: {e}. Check the MaxText param structure.")
        logging.info(f"Available keys in top level: {params.keys()}")
        return

    # Verify with a quick forward pass
    logging.info("Verifying model forward pass...")
    test_input = torch.randint(0, 32000, (1, 10))
    with torch.no_grad():
        output = hf_model(test_input)
    assert output.logits.shape == (1, 10, 32000)
    logging.info("Forward pass successful.")
    
    # Save the model
    Path(hf_out_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving HF model to {hf_out_dir}")
    hf_model.save_pretrained(hf_out_dir, safe_serialization=True)
    
    # Also save tokenizer
    logging.info(f"Saving tokenizer from {tokenizer_path} to {hf_out_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(hf_out_dir)
    
    logging.info("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbax_dir", type=str, required=True, help="Path to MaxText orbax checkpoint")
    parser.add_argument("--hf_out_dir", type=str, required=True, help="Path to save HF format model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to original HF tokenizer directory")
    args = parser.parse_args()
    
    convert_checkpoint(args.orbax_dir, args.hf_out_dir, args.tokenizer_path)
