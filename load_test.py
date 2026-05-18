import orbax.checkpoint as ocp
import jax
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

orbax_dir = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run9/checkpoints/54362/items"
checkpointer = ocp.StandardCheckpointer()
try:
    metadata = checkpointer.metadata(orbax_dir)
    print("Got metadata.")
    
    def create_target(x):
        if hasattr(x, 'shape') and hasattr(x, 'dtype'):
            dtype = x.dtype
            if hasattr(dtype, 'type'): # convert jax DType to numpy dtype
                dtype = np.dtype(dtype)
            return jax.ShapeDtypeStruct(shape=x.shape, dtype=dtype)
        return x

    if hasattr(metadata, 'tree'):
        target = jax.tree_util.tree_map(create_target, metadata.tree)
    else:
        target = jax.tree_util.tree_map(create_target, metadata)
        
    ckpt = checkpointer.restore(orbax_dir, target=target)
    print("Loaded ckpt:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Keys in ckpt:", list(ckpt.keys()))
        if 'params' in ckpt:
            print("Keys in params:", list(ckpt['params'].keys()))
except Exception as e:
    import traceback
    traceback.print_exc()
