import argparse
import logging
import wandb
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_wandb_history(run_path: str):
    """Downloads history data from a specific W&B run."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
        # Fetch history
        history = run.history(samples=1000)
        return pd.DataFrame(history)
    except Exception as e:
        logging.error(f"Failed to fetch W&B run {run_path}: {e}")
        return pd.DataFrame()

def plot_curves(project_name: str, run_id: str, output_dir: str):
    """
    Downloads W&B logs and plots them using seaborn/matplotlib.
    Requires W&B authentication (wandb login).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    run_path = f"{project_name}/{run_id}"
    logging.info(f"Fetching data from W&B: {run_path}")
    
    df = download_wandb_history(run_path)
    
    if df.empty:
        logging.warning("No data retrieved. Ensure W&B is logged in and the run exists.")
        return
        
    logging.info("Generating plots...")
    
    # 1. Plot Loss
    if 'train/loss' in df.columns or 'loss' in df.columns:
        loss_col = 'train/loss' if 'train/loss' in df.columns else 'loss'
        step_col = '_step'
        
        plt.figure(figsize=(10, 6))
        plt.plot(df[step_col], df[loss_col], label='Training Loss', color='blue', alpha=0.8)
        
        # Smoothed loss
        smoothed = df[loss_col].rolling(window=max(1, len(df)//20)).mean()
        plt.plot(df[step_col], smoothed, label='Smoothed', color='red', linewidth=2)
        
        plt.title(f"Training Loss - {run_id}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_path / f"{run_id}_loss.png", dpi=300)
        plt.close()
        
    # 2. Plot Learning Rate
    if 'train/learning_rate' in df.columns or 'learning_rate' in df.columns:
        lr_col = 'train/learning_rate' if 'train/learning_rate' in df.columns else 'learning_rate'
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['_step'], df[lr_col], label='Learning Rate', color='purple')
        plt.title(f"Learning Rate Schedule - {run_id}")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / f"{run_id}_lr.png", dpi=300)
        plt.close()

    logging.info(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="himanshu/tinymath-1b-sft")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results/plots")
    args = parser.parse_args()
    
    plot_curves(args.project, args.run_id, args.output_dir)
