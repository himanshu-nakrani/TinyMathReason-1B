"""
Inspect the structure of a MaxText Orbax checkpoint.

Usage:
    python src/train/inspect_checkpoint.py \
        --orbax_dir gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362

This will print the full PyTree structure with shapes and dtypes,
which is essential before running the conversion script.
"""
import argparse
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def inspect_checkpoint(orbax_dir: str):
    """Load and inspect the PyTree structure of an Orbax checkpoint."""
    import jax
    import numpy as np
    from orbax import checkpoint as ocp

    logging.info(f"Loading checkpoint from: {orbax_dir}")

    # Use StandardCheckpointer for modern Orbax
    try:
        checkpointer = ocp.PyTreeCheckpointer()
        ckpt = checkpointer.restore(orbax_dir)
    except Exception as e:
        logging.warning(f"PyTreeCheckpointer failed: {e}")
        logging.info("Trying StandardCheckpointer...")
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(orbax_dir)

    logging.info("Checkpoint loaded successfully!\n")

    def print_tree(tree, prefix="", file=None):
        """Recursively print the PyTree structure."""
        if isinstance(tree, dict):
            for key in sorted(tree.keys()):
                print_tree(tree[key], prefix=f"{prefix}/{key}", file=file)
        elif hasattr(tree, 'shape'):
            line = f"{prefix}: shape={tree.shape}, dtype={tree.dtype}"
            print(line)
            if file:
                file.write(line + "\n")
        elif isinstance(tree, (list, tuple)):
            for i, item in enumerate(tree):
                print_tree(item, prefix=f"{prefix}[{i}]", file=file)
        else:
            line = f"{prefix}: type={type(tree).__name__}, value={tree}"
            print(line)
            if file:
                file.write(line + "\n")

    print("=" * 80)
    print("CHECKPOINT STRUCTURE")
    print("=" * 80)
    print_tree(ckpt)

    # Also print top-level keys
    print("\n" + "=" * 80)
    print("TOP-LEVEL KEYS")
    print("=" * 80)
    if isinstance(ckpt, dict):
        for key in sorted(ckpt.keys()):
            val = ckpt[key]
            if isinstance(val, dict):
                print(f"  '{key}': dict with {len(val)} keys -> {sorted(val.keys())[:10]}")
            elif hasattr(val, 'shape'):
                print(f"  '{key}': array shape={val.shape}")
            else:
                print(f"  '{key}': {type(val).__name__}")

    # If 'params' exists, explore deeper
    params = ckpt.get('params', ckpt)
    if isinstance(params, dict) and 'params' in params:
        params = params['params']

    if isinstance(params, dict):
        print("\n" + "=" * 80)
        print("PARAMETER KEYS (2 levels deep)")
        print("=" * 80)
        for k1 in sorted(params.keys()):
            v1 = params[k1]
            if isinstance(v1, dict):
                for k2 in sorted(v1.keys()):
                    v2 = v1[k2]
                    if isinstance(v2, dict):
                        subkeys = sorted(v2.keys())[:8]
                        print(f"  {k1}/{k2}: dict -> {subkeys}")
                    elif hasattr(v2, 'shape'):
                        print(f"  {k1}/{k2}: shape={v2.shape}, dtype={v2.dtype}")
                    else:
                        print(f"  {k1}/{k2}: {type(v2).__name__}")
            elif hasattr(v1, 'shape'):
                print(f"  {k1}: shape={v1.shape}, dtype={v1.dtype}")

    return ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MaxText Orbax checkpoint structure")
    parser.add_argument("--orbax_dir", type=str, required=True,
                        help="Path to MaxText Orbax checkpoint (local or GCS)")
    args = parser.parse_args()

    inspect_checkpoint(args.orbax_dir)
