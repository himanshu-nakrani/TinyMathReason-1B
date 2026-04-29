import argparse
import logging
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def parse_maxtext_logs(log_file: str):
    """
    Parses a MaxText stdout log file to extract steps, loss, and throughput.
    """
    data = []
    
    # Typical MaxText log format for metrics:
    # "Step: 100, Loss: 5.432, Tokens/sec: 110543, LR: 0.00015"
    
    step_pattern = re.compile(r'Step:\s*(\d+)')
    loss_pattern = re.compile(r'Loss:\s*([0-9.]+)')
    tps_pattern = re.compile(r'Tokens/sec:\s*([0-9.]+)')
    lr_pattern = re.compile(r'LR:\s*([0-9.e-]+)')
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check if this line contains Step metrics
            if 'Step:' in line and 'Loss:' in line:
                try:
                    step = int(step_pattern.search(line).group(1))
                    loss = float(loss_pattern.search(line).group(1))
                    
                    tps_match = tps_pattern.search(line)
                    tps = float(tps_match.group(1)) if tps_match else None
                    
                    lr_match = lr_pattern.search(line)
                    lr = float(lr_match.group(1)) if lr_match else None
                    
                    data.append({
                        "step": step,
                        "loss": loss,
                        "tps": tps,
                        "lr": lr
                    })
                except AttributeError:
                    continue
                    
    return pd.DataFrame(data)

def monitor_training(log_file: str, output_dir: str):
    """
    Parses logs and generates training curve plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Parsing logs from {log_file}...")
    df = parse_maxtext_logs(log_file)
    
    if df.empty:
        logging.error("No metrics found in log file.")
        return
        
    logging.info(f"Found {len(df)} metric steps. Plotting...")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], label='Training Loss', color='blue', alpha=0.7)
    # Optional: rolling average for smoothing
    plt.plot(df['step'], df['loss'].rolling(10).mean(), label='Smoothed Loss', color='red')
    plt.title("Pretraining Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path / "loss_curve.png", dpi=300)
    plt.close()
    
    # Plot Throughput
    if df['tps'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['tps'], label='Tokens / sec', color='green')
        plt.title("Training Throughput")
        plt.xlabel("Step")
        plt.ylabel("Tokens / second")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / "throughput_curve.png", dpi=300)
        plt.close()
        
    # Plot LR
    if df['lr'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['lr'], label='Learning Rate', color='purple')
        plt.title("Learning Rate Schedule")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / "lr_curve.png", dpi=300)
        plt.close()
        
    logging.info(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True, help="Path to training log text file")
    parser.add_argument("--output_dir", type=str, default="./plots")
    args = parser.parse_args()
    
    monitor_training(args.log_file, args.output_dir)
