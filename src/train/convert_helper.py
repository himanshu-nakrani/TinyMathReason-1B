
import sys
import os
import functools
import types
import importlib.abc
from unittest.mock import MagicMock
import jax
import jax.numpy as jnp
from absl import app

# 1. JAX Patches
import jax._src.api
_original_jit = jax.jit
def _patched_jit(fun=None, **kwargs):
    if fun is None: return functools.partial(_original_jit, **kwargs)
    return _original_jit(fun, **kwargs)
jax.jit = _patched_jit
jax._src.api.jit = _patched_jit

if not hasattr(jax, 'set_mesh'):
    import contextlib
    @contextlib.contextmanager
    def _set_mesh_shim(mesh):
        with mesh: yield
    jax.set_mesh = _set_mesh_shim
    jax.sharding.set_mesh = _set_mesh_shim

# 2. Robust Mock Finder
class MockFinder(importlib.abc.MetaPathFinder):
    _MOCKS = ('jax.pallas', 'jax.experimental.pallas', 'qwix', 'tokamax', 'drjax', 'pathwaysutils')
    def find_module(self, fullname, path=None):
        if any(fullname == m or fullname.startswith(m + '.') for m in self._MOCKS): return self
    def load_module(self, fullname):
        if fullname in sys.modules: return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []; mod.__file__ = '<mock>'; mod.__loader__ = self
        if fullname == 'pathwaysutils.elastic.manager':
            class _M:
                def __init__(self, *a, **k): pass
            mod.Manager = _M; mod.ScaleUpSignalError = mod.ScaleDownSignalError = Exception
        mod.__getattr__ = lambda name: MagicMock()
        sys.modules[fullname] = mod
        return mod
sys.meta_path.insert(0, MockFinder())

# 3. Setup Path and Distributed Init
sys.path.append(os.path.join(os.path.expanduser('~'), 'maxtext', 'src'))
try:
    jax.distributed.initialize()
    print(f"Worker {jax.process_index()} initialized.")
except Exception as e:
    print(f"Distributed init info: {e}")

# 4. Import MaxText
import maxtext.utils.generate_param_only_checkpoint as gpoc
from maxtext.utils import checkpointing, maxtext_utils, pyconfig
from maxtext.common import common_types

def print_tree(tree, indent=0, path=""):
    prefix = "  " * indent
    if isinstance(tree, dict):
        for k, v in sorted(tree.items()):
            print(f"{prefix}{k} (type: {type(v).__name__})")
            print_tree(v, indent + 1, f"{path}/{k}")
    elif hasattr(tree, "shape"):
        print(f"{prefix}[LEAF] path: {path}, shape: {tree.shape}, dtype: {tree.dtype}")
    elif isinstance(tree, (list, tuple)):
        print(f"{prefix}[LIST/TUPLE] length: {len(tree)}")
        if len(tree) > 0:
            print_tree(tree[0], indent + 1, f"{path}[0]")

def patched_main(argv):
    config = pyconfig.initialize(argv)
    print(f"\n--- CONFIG CHECK ---")
    print(f"base_num_decoder_layers: {config.base_num_decoder_layers}")
    print(f"model_name: {config.model_name}")
    print(f"vocab_size: {config.vocab_size}")
    
    mesh = maxtext_utils.setup_mesh(config)
    
    # We call setup_model manually to inspect it
    with mesh:
        model, tx = maxtext_utils.setup_model(config, mesh)
        
        print("\n--- MODEL TEMPLATE (params_shape) ---")
        # Initialize with zeros to see the structure
        init_key = jax.random.PRNGKey(0)
        input_shape = (1, config.max_target_length)
        # Some models require specific inputs, but zeros usually work for shape inspection
        abstract_input = jnp.zeros(input_shape, dtype=jnp.int32)
        variables = model.init(init_key, abstract_input)
        
        if 'params' in variables:
            print_tree(variables['params'])
            if 'decoder' in variables['params'] and 'layers' in variables['params']['decoder']:
                 print("\nSUCCESS: 'layers' key FOUND in model template.")
            else:
                 print("\nFAILURE: 'layers' key MISSING from model template!")
                 # Let's see what's in decoder
                 if 'decoder' in variables['params']:
                     print(f"Decoder keys: {list(variables['params']['decoder'].keys())}")
        else:
            print("Variables keys:", variables.keys())

    # Proceed to load
    print("\n--- LOADING CHECKPOINT ---")
    training_state, training_state_annotations = checkpointing.load_full_state(
        config.load_full_state_path,
        config,
        mesh,
        model,
        tx,
    )
    
    print("\n--- LOADED STATE PARAMS ---")
    print_tree(training_state.params)
    
    # If we reached here, maybe we can find where the layers went
    total_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y.size, jax.tree_util.tree_leaves(training_state.params), 0
    )
    print(f"\nTotal loaded params: {total_params / 1e9:.3f} Billion")

    # Exit early to avoid erroring out the whole script
    print("\nDiagnostic complete. Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    # We use a wrapper around gpoc.main or just our own main
    sys.argv = ['generate_param_only_checkpoint', 'maxtext_config.yml',
        'load_full_state_path=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/checkpoints/54362/items',
        'base_output_directory=gs://tinymath-reason-data-himanshu/checkpoints/tinymath-1b-prod-run2/param_only',
        'run_name=tinymath-1b-decode',
        'force_unroll=true']
    app.run(patched_main)
