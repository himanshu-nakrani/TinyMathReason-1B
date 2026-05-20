import torch
import sys
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_hf_model():
    model_path = "./hf_1b_model"
    logging.info(f"Loading tokenizer and model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)
        
    # Force CPU for local testing to avoid MPS shader compilation latency
    device = "cpu"
    dtype = torch.bfloat16
        
    logging.info(f"Using device: {device} with dtype: {dtype}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype
        ).to(device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.info("Attempting fallback to CPU loading...")
        try:
            device = "cpu"
            dtype = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype
            ).to(device)
        except Exception as e_fallback:
            logging.error(f"Fallback CPU load failed: {e_fallback}")
            sys.exit(1)

    logging.info("Model loaded successfully. Preparing a sample prompt...")
    
    prompt = "Problem: Solve 12 + 15 =\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    logging.info("Running text generation...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id or 2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "="*50)
        print("PROMPT:")
        print(prompt)
        print("-"*50)
        print("GENERATED RESPONSE:")
        print(response)
        print("="*50 + "\n")
        logging.info("Verification complete: Base model successfully ran a forward pass and generated text!")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_hf_model()
