import argparse
import logging
import json
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_benchmarks(model_path: str, output_dir: str):
    """
    Wraps lm-evaluation-harness to run standard benchmarks.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    tasks = {
        "gsm8k": {"shots": 8},
        "math_algebra": {"shots": 4}, # 'math' is a collection in lm-eval, we test algebra as representative
        "arc_easy": {"shots": 0},
        "arc_challenge": {"shots": 25},
        "hellaswag": {"shots": 10},
        "mmlu": {"shots": 5}
    }
    
    results_summary = {}
    
    for task, config in tasks.items():
        logging.info(f"Running benchmark: {task} ({config['shots']}-shot)")
        
        output_file = out_path / f"{task}_results.json"
        
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16",
            "--tasks", task,
            "--num_fewshot", str(config['shots']),
            "--batch_size", "auto",
            "--output_path", str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Read back results
            with open(output_file, 'r') as f:
                res = json.load(f)
                
            # lm-eval returns complex JSON, we extract the main metric
            metrics = res['results'][task]
            # typical metrics: acc, acc_norm, exact_match
            primary_metric = metrics.get('acc_norm,none', metrics.get('exact_match,none', metrics.get('acc,none', "N/A")))
            results_summary[task] = primary_metric
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {task}: {e}")
            results_summary[task] = "Error"
            
    # Output clean markdown table
    md_file = out_path / "benchmark_summary.md"
    with open(md_file, "w") as f:
        f.write(f"# Benchmark Results for {model_path}\n\n")
        f.write("| Benchmark | Shots | Score |\n")
        f.write("| :--- | :---: | :---: |\n")
        for task, config in tasks.items():
            score = results_summary.get(task, "N/A")
            if isinstance(score, float):
                score = f"{score*100:.1f}%"
            f.write(f"| {task} | {config['shots']} | {score} |\n")
            
    logging.info(f"Saved benchmark summary to {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    args = parser.parse_args()
    
    run_benchmarks(args.model_path, args.output_dir)
