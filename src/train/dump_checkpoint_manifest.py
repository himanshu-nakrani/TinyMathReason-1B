
import os
import jax
import orbax.checkpoint as ocp
from etils import epath
import json

def dump_manifest(checkpoint_path):
    print(f"Opening checkpoint at {checkpoint_path}")
    path = epath.Path(checkpoint_path)
    
    # Check if it's a directory
    if not path.exists():
        print("Path does not exist!")
        return

    # Try to load as a PyTreeCheckpointHandler would
    handler = ocp.PyTreeCheckpointHandler()
    
    try:
        metadata = handler.inspect(path)
        print("\n--- ORBAX INSPECT METADATA ---")
        # metadata is usually a PyTree of information about the arrays
        
        def print_metadata_tree(tree, path_str=""):
            if isinstance(tree, dict):
                for k, v in tree.items():
                    print_metadata_tree(v, f"{path_str}/{k}")
            else:
                # v is likely a summary of the array
                print(f"{path_str}: {tree}")

        print_metadata_tree(metadata)
        
    except Exception as e:
        print(f"Failed to inspect with PyTreeCheckpointHandler: {e}")
        
    # Try listing files manually
    print("\n--- GCS FILE LISTING ---")
    try:
        for f in path.iterdir():
            print(f"{f}")
            if f.is_dir():
                for subf in f.iterdir():
                    print(f"  {subf}")
    except Exception as e:
        print(f"Failed to list files: {e}")

if __name__ == "__main__":
    checkpoint_path = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items"
    dump_manifest(checkpoint_path)
