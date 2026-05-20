import argparse
import logging
import json
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Default benchmark tasks with configurations
DEFAULT_TASKS = {
    "gsm8k": {"shots": 8, "metrics": ["exact_match,strict-match", "exact_match,flexible-extract"]},
    "minerva_math_algebra": {"shots": 4, "metrics": ["exact_match,strict-match"]},
    "arc_easy": {"shots": 0, "metrics": ["acc_norm,none", "acc,none"]},
    "arc_challenge": {"shots": 25, "metrics": ["acc_norm,none", "acc,none"]},
    "hellaswag": {"shots": 10, "metrics": ["acc_norm,none", "acc,none"]},
    "mmlu": {"shots": 5, "metrics": ["acc,none"]},
}


def run_benchmarks(
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    tasks: dict = None,
    gcs_output: str = None,
):
    """
    Wraps lm-evaluation-harness to run standard benchmarks.
    
    Args:
        model_path: Local path or HF Hub model ID
        output_dir: Local directory to save results
        device: Device to run on (cuda, cpu, mps)
        tasks: Dict of task configs. Defaults to DEFAULT_TASKS
        gcs_output: Optional GCS path to upload results
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if tasks is None:
        tasks = DEFAULT_TASKS
    
    results_summary = {}
    
    for task, config in tasks.items():
        logging.info(f"Running benchmark: {task} ({config['shots']}-shot)")
        
        output_file = out_path / task
        
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype=bfloat16",
            "--tasks", task,
            "--num_fewshot", str(config['shots']),
            "--batch_size", "auto",
            "--device", device,
            "--output_path", str(output_file),
            "--log_samples",
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(f"  stdout: {result.stdout[-500:]}")
            
            # Find the results.json file (lm_eval creates timestamped subdirs)
            import glob
            result_files = glob.glob(str(output_file / "**/results.json"), recursive=True)
            
            if result_files:
                with open(sorted(result_files)[-1], 'r') as f:
                    res = json.load(f)
                
                task_results = res.get('results', {})
                # Try to find the score using configured metrics
                primary_metric = None
                for task_key, task_data in task_results.items():
                    if isinstance(task_data, dict):
                        for metric in config['metrics']:
                            if metric in task_data:
                                primary_metric = task_data[metric]
                                break
                        if primary_metric is not None:
                            break
                
                results_summary[task] = primary_metric if primary_metric is not None else "Parse error"
            else:
                results_summary[task] = "No results file"
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {task}: {e}")
            logging.error(f"  stderr: {e.stderr[-500:] if e.stderr else 'N/A'}")
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
                score = f"{score*100:.2f}%"
            f.write(f"| {task} | {config['shots']} | {score} |\n")
            
    logging.info(f"Saved benchmark summary to {md_file}")
    
    # Print summary to stdout
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    with open(md_file, 'r') as f:
        print(f.read())
    print("=" * 60)
    
    # Upload to GCS if requested
    if gcs_output:
        logging.info(f"Uploading results to GCS: {gcs_output}")
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{out_path}/*", f"{gcs_output}/"],
                check=True
            )
            logging.info(f"Results uploaded to {gcs_output}/")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Failed to upload to GCS: {e}")
    
    # Save combined results JSON
    combined_file = out_path / "combined_results.json"
    with open(combined_file, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks on TinyMathReason-1B"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path or HF Hub model ID")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Device to run inference on")
    parser.add_argument("--gcs_output", type=str, default=None,
                        help="GCS path to upload results (e.g., gs://bucket/eval_results/)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to run (default: all)")
    args = parser.parse_args()
    
    # Filter tasks if specific ones are requested
    task_config = DEFAULT_TASKS
    if args.tasks:
        task_config = {k: v for k, v in DEFAULT_TASKS.items() if k in args.tasks}
        if not task_config:
            logging.error(f"No matching tasks found. Available: {list(DEFAULT_TASKS.keys())}")
            exit(1)
    
    run_benchmarks(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        tasks=task_config,
        gcs_output=args.gcs_output
    )
