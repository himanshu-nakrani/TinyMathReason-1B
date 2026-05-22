import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Hand-curated problems (3 of each for brevity here, actual script would have 30)
EVAL_PROBLEMS = [
    {"level": "Easy", "q": "What is 15% of 200?"},
    {"level": "Easy", "q": "Solve for x: 3x + 5 = 20"},
    {"level": "Easy", "q": "If a train travels 60 miles per hour, how far does it go in 2.5 hours?"},
    
    {"level": "Medium", "q": "Find the roots of the quadratic equation: x^2 - 5x + 6 = 0"},
    {"level": "Medium", "q": "What is the derivative of f(x) = 3x^4 - 2x^2 + x?"},
    {"level": "Medium", "q": "If A = {1, 2, 3} and B = {3, 4, 5}, what is the union of sets A and B?"},
    
    {"level": "Hard", "q": "Prove that the sum of the first n positive integers is n(n+1)/2."},
    {"level": "Hard", "q": "Evaluate the integral of x * e^x dx from 0 to 1."},
    {"level": "Hard", "q": "Find the eigenvalues of the matrix [[2, 1], [1, 2]]."}
]

def run_custom_eval(model_path: str, output_file: str):
    """
    Runs hand-curated math problems and formats outputs as markdown.
    """
    logging.info(f"Loading model {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype
    ).to(device)
    
    system_prompt = "You are a mathematical reasoning assistant. Solve problems step by step inside <think> tags, and then provide the final answer."
    
    results = []
    
    for prob in EVAL_PROBLEMS:
        logging.info(f"Evaluating ({prob['level']}): {prob['q']}")
        
        # Check if tokenizer has a chat template configured
        if getattr(tokenizer, "chat_template", None) is not None:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prob["q"]}
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback formatting for raw base models
            prompt = f"Below is a math problem. Solve it step-by-step.\nProblem: {prob['q']}\nSolution:"
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Cap at 256 for CPU local baseline speed
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id or 2
            )
            
        generated_len = outputs.shape[1] - inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][-generated_len:], skip_special_tokens=True)
        
        results.append({
            "level": prob["level"],
            "question": prob["q"],
            "response": response
        })
        
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(f"# Custom Evaluation: {model_path}\n\n")
        for res in results:
            f.write(f"### Level: {res['level']}\n")
            f.write(f"**Q: {res['question']}**\n\n")
            f.write(f"**A:**\n{res['response']}\n\n")
            f.write("---\n")
            
    logging.info(f"Saved custom evaluation to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./eval_results/custom_eval.md")
    args = parser.parse_args()
    
    run_custom_eval(args.model_path, args.output_file)
