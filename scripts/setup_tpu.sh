#!/bin/bash
# setup_tpu.sh - automated MaxText setup for Spot TPU VMs
# ========================================================
# This script:
#   1. Clones MaxText
#   2. Patches Python version requirement (3.10 compat)
#   3. Installs all dependencies (JAX/TPU, Flax, Grain, etc.)
#   4. Patches compatibility issues (Flax nnx, JAX reshard, etc.)
#   5. Injects runtime mocks for internal Google modules
#   6. Launches the training run
# ========================================================

set -e  # Exit on first error

# 0. Setup Python 3.12 Virtual Environment
echo "Setting up Python 3.12 virtual environment..."
python3.12 -m venv $HOME/venv312
source $HOME/venv312/bin/activate
pip install --upgrade pip

# 1. Clone MaxText (always fresh to ensure pinned version)
echo "Setting up MaxText..."
rm -rf $HOME/maxtext
git clone https://github.com/AI-Hypercomputer/maxtext.git $HOME/maxtext

# Copy the config file into the maxtext directory
if [ -f "$HOME/maxtext_config.yml" ]; then
    cp "$HOME/maxtext_config.yml" "$HOME/maxtext/"
fi

cd $HOME/maxtext

# 2. Deploy custom architecture
if [ -f "$HOME/tinymath-1b.yml" ]; then
    mkdir -p src/maxtext/configs/models/
    cp "$HOME/tinymath-1b.yml" src/maxtext/configs/models/
fi

# Patch types.py to allow tinymath-1b
sed -i 's/ModelName = Literal\[/ModelName = Literal["tinymath-1b", /' src/maxtext/configs/types.py

# Clean repository files to prevent duplicate injections on multiple runs
git checkout -- src/maxtext/trainers/pre_train/train.py 2>/dev/null || true
git checkout -- src/maxtext/utils/sharding.py 2>/dev/null || true

# 3. Install dependencies using MaxText's official setup script
echo "Installing dependencies via MaxText setup.sh..."
if [ -f "$HOME/maxtext/src/dependencies/scripts/setup.sh" ]; then
    cd $HOME/maxtext
    bash src/dependencies/scripts/setup.sh
else
    echo "ERROR: setup.sh not found in $HOME/maxtext/src/dependencies/scripts/"
    find $HOME/maxtext -name setup.sh
    exit 1
fi

# 4. Patch compatibility issues (Flax nnx, JAX reshard, etc.)
# (We still need our custom patches and mocks)

# 7. Backwards compatibility patch for Pytree -> Object and reshard
find src/ -name "*.py" -exec sed -i 's/from flax.nnx import Pytree/from flax.nnx import Object as Pytree/g' {} +

# Patch Flax 0.11+ API: .get_value() -> .value (Flax 0.10.5 uses .value property)
find src/ -name "*.py" -exec sed -i 's/\.get_value()/.value/g' {} +

# Patch jax.Ref removal in recent JAX
sed -i 's/jax\.Ref/typing.Any/g' src/maxtext/kernels/ragged/ragged_gather*.py
sed -i '1s/^/import typing\n/' src/maxtext/kernels/ragged/ragged_gather*.py



cat << 'EOF' > patch_sharding.py
import re
file_path = "src/maxtext/utils/sharding.py"
with open(file_path, "r") as f:
    content = f.read()

content = re.sub(r',\s*reshard', '', content)
content = re.sub(r'from\s+jax\s+import\s+reshard', '', content)
content = re.sub(r'from\s+jax\.sharding\s+import\s+reshard', '', content)

new_import = """
try:
    from jax.sharding import reshard
except ImportError:
    try:
        from jax.experimental.shard import reshard
    except ImportError:
        def reshard(x, *args, **kwargs):
            return x
"""
content = new_import + content

with open(file_path, "w") as f:
    f.write(content)
EOF
python3 patch_sharding.py

# 8. Create the mock injector
# This file is prepended to train.py to handle missing internal Google modules,
# JAX API differences, and grain version mismatches.
cat << 'MOCK_EOF' > mock_injector.py
import sys
import functools
import os

# ─────────────────────────────────────────────────────────────
# TensorFlow isolation: prevent TF from claiming TPU devices
# ─────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'TPU')
except Exception:
    pass

import jax
from unittest.mock import MagicMock

# ─────────────────────────────────────────────────────────────
# Pallas compatibility shim (pallas moved between JAX versions)
# We don't use Pallas GPU kernels for TPU training, so mock it.
# ─────────────────────────────────────────────────────────────
import importlib.abc
import importlib.machinery
import types

class _MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    def __getattr__(self, name):
        return MagicMock()

def mock_package(name):
    if name not in sys.modules:
        sys.modules[name] = _MockModule(name)

class _PallasMockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('jax.pallas', 'jax.experimental.pallas')):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return _MockModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _PallasMockFinder())

# Pre-populate base modules
mock_package("jax.pallas")
mock_package("jax.experimental.pallas")

# ─────────────────────────────────────────────────────────────
# JAX compatibility patches
# ─────────────────────────────────────────────────────────────
if getattr(jax, "_is_monkey_patched", False) is False:
    # Patch jax.jit to support decorator factory pattern
    _original_jit = jax.jit
    def _patched_jit(fun=None, **kwargs):
        if fun is None:
            return functools.partial(_original_jit, **kwargs)
        return _original_jit(fun, **kwargs)
    jax.jit = _patched_jit
    jax._src.api.jit = _patched_jit

    # Patch jax.config.update to ignore unsupported experimental flags
    _original_config_update = jax.config.update
    def _patched_config_update(name, value):
        try:
            _original_config_update(name, value)
        except AttributeError:
            pass
    jax.config.update = _patched_config_update

    # Shim for jax.Ref (removed in recent JAX)
    if not hasattr(jax, "Ref"):
      from typing import Any
      jax.Ref = Any

    # Shim for jax.set_mesh and jax.sharding.set_mesh (added in JAX 0.7.1, missing in 0.6.x)
    if not hasattr(jax, 'set_mesh'):
        import contextlib
        @contextlib.contextmanager
        def _set_mesh_shim(mesh):
            with mesh:
                yield
        jax.set_mesh = _set_mesh_shim
        jax.sharding.set_mesh = _set_mesh_shim

    jax._is_monkey_patched = True

# ─────────────────────────────────────────────────────────────
# Mock internal Google-only modules
# ─────────────────────────────────────────────────────────────
# pathwaysutils needs MagicMock for arbitrary attrs (e.g. .initialize())
# BUT exception classes must be real (used in except clauses)
# AND Manager must be a real class (used as type annotation)
class _ScaleUpSignalError(Exception): pass
class _ScaleDownSignalError(Exception): pass
class _Manager:
    def __init__(self, *args, **kwargs): pass

_pw_mod = MagicMock()
_pw_manager = MagicMock()
_pw_manager.ScaleUpSignalError = _ScaleUpSignalError
_pw_manager.ScaleDownSignalError = _ScaleDownSignalError
_pw_manager.Manager = _Manager
_pw_mod.elastic.manager = _pw_manager

sys.modules["pathwaysutils"] = _pw_mod
sys.modules["pathwaysutils.elastic"] = _pw_mod.elastic
sys.modules["pathwaysutils.elastic.manager"] = _pw_manager

for mod in [
    "qwix", "qwix.pallas", "qwix._src", "qwix._src.core",
    "qwix.contrib", "qwix.contrib.sparsity",
    "tokamax", "tokamax._src", "tokamax._src.ops",
    "tokamax._src.ops.experimental", "tokamax._src.ops.experimental.tpu",
    "tokamax._src.ops.experimental.tpu.splash_attention",
    "drjax",
]:
    sys.modules[mod] = MagicMock()

# ─────────────────────────────────────────────────────────────
# HuggingFace AutoTokenizer Mock (for GCS tiktoken bypass)
# ─────────────────────────────────────────────────────────────
import transformers
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
def _mock_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
    if "gs://" in str(pretrained_model_name_or_path) or "tiktoken" in str(pretrained_model_name_or_path):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.unk_token_id = 0
        mock_tokenizer.bos_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32768
        return mock_tokenizer
    return _orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
transformers.AutoTokenizer.from_pretrained = _mock_from_pretrained

# ─────────────────────────────────────────────────────────────
# Grain compatibility patches
# ─────────────────────────────────────────────────────────────
import grain.python as grain_python

# Ensure grain.experimental exists with required classes
if not hasattr(grain_python, "experimental"):
    sys.modules["grain.experimental"] = MagicMock()

import grain
if not hasattr(grain, "experimental"):
    grain.experimental = sys.modules.get("grain.experimental", MagicMock())

if not hasattr(grain.experimental, "BestFitPackIterDataset"):
    class DummyPackIterDataset:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset
        def __iter__(self):
            return iter(self.dataset)
    grain.experimental.BestFitPackIterDataset = DummyPackIterDataset
    grain.experimental.pick_performance_config = lambda *args, **kwargs: None

# Ensure PyGrainCheckpointHandler exists (may be missing in grain-nightly)
if not hasattr(grain_python, "PyGrainCheckpointHandler"):
    class DummyCheckpointHandler:
        def __init__(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): return None
        def restore(self, *args, **kwargs): return None
    grain_python.PyGrainCheckpointHandler = DummyCheckpointHandler
MOCK_EOF
# 9. Inject the mock loader at the top of train.py if not already injected
if ! grep -q "Mock internal Google-only modules" src/maxtext/trainers/pre_train/train.py; then
    cat mock_injector.py src/maxtext/trainers/pre_train/train.py > temp.py
    mv temp.py src/maxtext/trainers/pre_train/train.py
fi

# 10. Start the training run (PRODUCTION)
echo "=============================================="
echo "Starting MaxText Pretraining (PRODUCTION)..."
echo "  run_name: tinymath-1b-prod-run11"
echo "  dataset: HF pre-tokenized jsonl.zst"
echo "=============================================="

# Use a while loop to auto-restart the script in case of transient software crashes.
# (If the TPU is preempted by GCP, the entire VM will shut down, and you will
# need to recreate the TPU and re-run this script to resume from the latest checkpoint).
while true; do
    PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=0 python3 -u src/maxtext/trainers/pre_train/train.py \
        maxtext_config.yml

    echo "Training script exited. Restarting in 10 seconds to resume from latest checkpoint..."
    sleep 10
done
