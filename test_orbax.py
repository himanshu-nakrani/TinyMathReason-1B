import orbax.checkpoint as ocp
import jax
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
orbax_dir = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-tiny-test/checkpoints/1/items"

import os
os.environ["JAX_PLATFORMS"] = "cpu"

try:
    checkpointer = ocp.StandardCheckpointer()
    
    # In orbax 0.11, if we just want to restore everything as numpy arrays
    # without a target, we might just be able to mock the devices.
    # Let's use the XLA_FLAGS we found before.
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    import jax
    print(f"JAX devices: {len(jax.devices())}")
    
    ckpt = checkpointer.restore(orbax_dir)
    print("Restored successfully!")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {ckpt.keys()}")
        if 'params' in ckpt:
            if hasattr(ckpt['params'], 'keys'):
                print(f"Params keys: {ckpt['params'].keys()}")
except Exception as e:
    import traceback
    traceback.print_exc()
