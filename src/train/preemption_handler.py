#!/usr/bin/env python3
import time
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def launch_training():
    """Starts the MaxText training script via a subprocess."""
    # We use stdbuf to ensure logs are written to stdout without buffering
    cmd = [
        "python3", "MaxText/train.py",
        "../src/train/maxtext_config.yml"
    ]
    logging.info(f"Launching training: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

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
    except Exception as e:
        # If we are not on GCP, or curl fails, we return False
        pass
    return False

def main():
    """
    Monitors training and handles spot instance preemptions.
    If preemption is detected, we could try to gracefully save, but MaxText handles 
    async checkpointing. So we just sleep until the VM is actually terminated.
    If the training process crashes (OOM, etc.), we restart it.
    """
    process = None
    
    while True:
        if process is None or process.poll() is not None:
            # Process is not running. Let's start/restart it.
            if process is not None:
                logging.warning(f"Training process exited with code {process.returncode}. Restarting in 60s...")
                time.sleep(60)
            
            process = launch_training()
            
        # Check GCP preemption metadata
        if check_preemption_status():
            logging.warning("!!! PREEMPTION SIGNAL RECEIVED !!!")
            logging.warning("The TPU VM will be terminated within 30 seconds.")
            # MaxText should ideally be configured to catch SIGTERM and save a checkpoint.
            # But async checkpointing every N steps is safer. We just wait for the inevitable.
            time.sleep(60) 
            
        time.sleep(30)

if __name__ == "__main__":
    main()
