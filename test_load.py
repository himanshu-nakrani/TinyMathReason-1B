import orbax.checkpoint as ocp
import jax
import numpy as np

orbax_dir = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-smoke-test/checkpoints/2/"
checkpointer = ocp.StandardCheckpointer()
try:
    # Try restoring without target, might fail with topology mismatch
    ckpt = checkpointer.restore(orbax_dir)
    print("Loaded without target!")
except Exception as e:
    print(f"Failed to load directly: {e}")
    try:
        # Get metadata and construct target with np.ndarray
        metadata = checkpointer.metadata(orbax_dir)
        print("Metadata type:", type(metadata))
        
        if hasattr(metadata, 'tree'):
            tree = metadata.tree
        else:
            tree = metadata
            
        def create_target(x):
            if hasattr(x, 'shape') and hasattr(x, 'dtype'):
                dtype = x.dtype
                if hasattr(dtype, 'type'):
                    dtype = np.dtype(dtype)
                # Create a jax.ShapeDtypeStruct to specify shape and dtype
                return jax.ShapeDtypeStruct(shape=x.shape, dtype=dtype)
            return x
        
        target = jax.tree_util.tree_map(create_target, tree)
        print("Created target tree with ShapeDtypeStructs.")
        
        ckpt = checkpointer.restore(orbax_dir, target=target)
        print("Loaded with target!")
    except Exception as e2:
        print(f"Failed to load with target: {e2}")
        import traceback
        traceback.print_exc()
