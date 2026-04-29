import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def generate_comparison(output_file: str):
    """
    Runs the same prompts on Base, SFT, DPO, TinyLlama, and Qwen2.5 
    for side-by-side comparison.
    """
    
    # Define models to compare
    # Note: In reality, paths would point to actual local checkpoints.
    # We use placeholder paths here.
    models = {
        "TinyMath-Base": "./hf_checkpoints/tinymath-1b-base",
        "TinyMath-SFT": "./sft_output/final",
        "TinyMath-DPO": "./dpo_output/final",
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    test_prompts = [
        "Solve: 5x - 7 = 2x + 8. Find x.",
        "A rectangular garden has a length that is twice its width. If the perimeter is 60 meters, what is the area?",
        "If f(x) = x^2 + 3x, what is f(f(2))?"
    ]
    
    results = {prompt: {} for prompt in test_prompts}
    
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step."
    
    for model_name, model_path in models.items():
        logging.info(f"Loading {model_name} from {model_path}...")
        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
        except Exception as e:
            logging.error(f"Failed to load {model_name}: {e}. Skipping.")
            continue
            
        for prompt in test_prompts:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            try:
                inputs_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback for models without chat template (e.g. base model)
                inputs_text = f"User: {prompt}\nAssistant:"
                
            inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            gen_len = outputs.shape[1] - inputs.input_ids.shape[1]
            response = tokenizer.decode(outputs[0][-gen_len:], skip_special_tokens=True)
            
            results[prompt][model_name] = response
            
        # Free memory before loading next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write("# Side-by-Side Model Comparison\n\n")
        
        for prompt in test_prompts:
            f.write(f"## Prompt: {prompt}\n\n")
            f.write("| Model | Response |\n")
            f.write("| :--- | :--- |\n")
            for model_name in models.keys():
                response = results[prompt].get(model_name, "N/A").replace('\n', '<br>')
                f.write(f"| **{model_name}** | {response} |\n")
            f.write("\n---\n\n")
            
    logging.info(f"Saved comparison to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="./eval_results/comparison.md")
    args = parser.parse_args()
    
    generate_comparison(args.output_file)
