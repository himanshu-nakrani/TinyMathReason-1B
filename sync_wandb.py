import wandb
import os
import sys

run_dir = sys.argv[1]
key = os.environ.get("WANDB_API_KEY")
if key:
    wandb.login(key=key, relogin=True)
else:
    wandb.login(relogin=True)
wandb.sync(run_dir)
