
import sys
import os
import jax
import jax.numpy as jnp
from absl import app

# Add MaxText to path
sys.path.append(os.path.join(os.path.expanduser('~'), 'maxtext', 'src'))

from maxtext.utils import pyconfig, maxtext_utils
from maxtext.layers import models

def main(argv):
    # Initialize config
    config = pyconfig.initialize(argv)
    print(f"Config loaded. Model Name: {config.model_name}")
    print(f"Num Layers: {config.num_decoder_layers}")
    
    # Setup Mesh
    mesh = maxtext_utils.setup_mesh(config)
    
    with mesh:
        # Setup Model
        model, tx = maxtext_utils.setup_model(config, mesh)
        
        # Initialize variables
        print("Initializing model variables (this may take a moment)...")
        init_key = jax.random.PRNGKey(0)
        abstract_input = jnp.zeros((1, config.max_target_length), dtype=jnp.int32)
        variables = model.init(init_key, abstract_input)
        
        # Calculate params
        def count_params(tree):
            return sum(x.size for x in jax.tree_util.tree_leaves(tree))
        
        total_params = count_params(variables['params'])
        print(f"\n--- MODEL VALIDATION ---")
        print(f"Total Parameters: {total_params:,}")
        print(f"Total Parameters (Billions): {total_params / 1e9:.3f} B")
        
        if total_params > 1_000_000_000:
            print("\nSUCCESS: Model initialized with > 1B parameters.")
        else:
            print("\nFAILURE: Parameter count is too low! Layers are still not being instantiated.")
            
        # Print top-level keys
        print("\nTop-level parameter keys:")
        for k in variables['params'].keys():
            print(f" - {k}")
            if isinstance(variables['params'][k], dict):
                print(f"   Sub-keys: {list(variables['params'][k].keys())}")

if __name__ == "__main__":
    # Point to the updated config
    sys.argv = ['verify_model', 'maxtext_config.yml']
    app.run(main)
