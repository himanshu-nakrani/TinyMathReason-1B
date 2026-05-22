import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

def test_model():
    model_path = "src/sft/sft_output/stage2/final"
    logging.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading model from {model_path} to {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(device)
    
    logging.info("=== Model Config ===")
    logging.info(f"  vocab_size: {model.config.vocab_size}")
    logging.info(f"  pad_token_id: {model.config.pad_token_id}")
    logging.info(f"  eos_token_id: {model.config.eos_token_id}")
    logging.info(f"  bos_token_id: {model.config.bos_token_id}")
    logging.info(f"  Embedding weight shape: {model.model.embed_tokens.weight.shape}")
    logging.info(f"  LM Head weight shape: {model.lm_head.weight.shape}")
    
    # Run simple generation A: ChatML template
    messages = [
        {"role": "system", "content": "You are a mathematical reasoning assistant. Solve problems step by step inside <think> tags, and then provide the final answer."},
        {"role": "user", "content": "If a train travels 60 miles per hour, how far does it go in 2.5 hours?"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logging.info(f"\n--- Standalone Generation Test (ChatML) ---")
    logging.info(f"Prompt text:\n{repr(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)  # Prevent transformers generation error
    logging.info(f"Tokenized prompt IDs: {inputs.input_ids[0].tolist()}")
    
    # Get stop tokens
    eos_token_ids = [tokenizer.eos_token_id or 1]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id > 0:
        eos_token_ids.append(im_end_id)
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id or 1
        )
        
    generated_ids = outputs[0][inputs.input_ids.shape[1]:].tolist()
    logging.info(f"Generated token IDs: {generated_ids[:20]}...")
    decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
    logging.info(f"Decoded generated output:\n{decoded}")
    logging.info("=" * 60)

if __name__ == "__main__":
    test_model()
