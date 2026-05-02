#!/usr/bin/env python3
import time
import subprocess
import logging
import argparse
import sys
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def launch_training(script_path):
    """Starts the training script via a subprocess."""
    logging.info(f"Launching training: {script_path}")
    # We use shell=True to allow complex commands with arguments
    return subprocess.Popen(script_path, shell=True)

def check_preemption_status():
    """
    Checks if the VM is about to be preempted.
    Google Cloud provides a metadata endpoint that returns true if preemption is imminent.
    """
    try:
        result = subprocess.run(
            ["curl", "-s", "http://metadata.google.internal/computeMetadata/v1/instance/preempted", 
             "-H", "Metadata-Flavor: Google"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() == "TRUE":
            return True
    except Exception:
        # If we are not on GCP, or curl fails, we return False
        pass
    return False

def main():
    parser = argparse.ArgumentParser(description="Monitor training and handle preemptions.")
    parser.add_argument("--script_path", type=str, required=True, 
                        help="The full command to run the training script (e.g. 'python3 MaxText/train.py ...')")
    parser.add_argument("--project", type=str, help="GCP Project ID (optional, for logging)")
    parser.add_argument("--zone", type=str, help="GCP Zone (optional, for logging)")
    parser.add_argument("--tpu_name", type=str, help="TPU Name (optional, for logging)")
    args = parser.parse_args()

    logging.info(f"Starting preemption handler for {args.tpu_name or 'current TPU'}")
    
    process = None
    
    while True:
        if process is None or process.poll() is not None:
            # Process is not running. Let's start/restart it.
            if process is not None:
                logging.warning(f"Training process exited with code {process.returncode}. Restarting in 60s...")
                time.sleep(60)
            
            process = launch_training(args.script_path)
            
        # Check GCP preemption metadata
        if check_preemption_status():
            logging.warning("!!! PREEMPTION SIGNAL RECEIVED !!!")
            logging.warning("The TPU VM will be terminated within 30 seconds.")
            # We don't kill the process; we let it run until the VM is terminated.
            # MaxText should be configured for async checkpointing.
            time.sleep(60) 
            
        time.sleep(30)

if __name__ == "__main__":
    main()
