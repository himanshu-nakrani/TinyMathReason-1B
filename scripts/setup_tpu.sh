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

# 3. Install only the runtime dependencies needed for training
# (Skip `pip install -e .` — it pulls in MaxText's full dev tree and takes 30+ min to resolve.
#  We use PYTHONPATH=src instead to make MaxText importable.)
echo "Installing runtime dependencies..."
pip install --upgrade pip
pip install \
    flax==0.10.5 \
    orbax-checkpoint \
    optax \
    grain-nightly \
    omegaconf \
    protobuf \
    pydantic \
    jaxtyping \
    safetensors \
    huggingface-hub \
    aqtp \
    google-cloud-storage \
    absl-py \
    tensorflow-cpu \
    tensorflow-datasets \
    datasets \
    gcsfs \
    transformers \
    tokenizers \
    tiktoken \
    sentencepiece \
    sympy \
    Pillow \
    ml_goodput_measurement \
    cloud_tpu_diagnostics \
    ml-collections \
    tensorboardX \
    zstandard

# 4. Install stable JAX with TPU support
echo "Purging conflicting JAX and TPU nightly builds..."
pip uninstall -y jax jaxlib libtpu libtpu-nightly 2>/dev/null || true
echo "Installing stable JAX with TPU support..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

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
# Uses a meta_path finder to intercept ALL pallas imports.
# ─────────────────────────────────────────────────────────────
import types
import importlib.abc
import importlib.machinery

class _PallasMockFinder(importlib.abc.MetaPathFinder):
    """Intercepts any import of jax.pallas.* or jax.experimental.pallas.*"""
    _PREFIXES = ('jax.pallas', 'jax.experimental.pallas')

    def find_module(self, fullname, path=None):
        for prefix in self._PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + '.'):
                return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        # Create a module that acts as a package and returns MagicMock for attrs
        mod = types.ModuleType(fullname)
        mod.__path__ = []
        mod.__file__ = '<pallas_mock>'
        mod.__loader__ = self
        mod.__package__ = fullname
        # Make attribute access return MagicMock for any unknown attr
        _real_mod_class = type('PallasMock', (types.ModuleType,), {
            '__getattr__': lambda self, name: MagicMock()
        })
        mock = _real_mod_class(fullname)
        mock.__path__ = []
        mock.__file__ = '<pallas_mock>'
        mock.__loader__ = self
        mock.__package__ = fullname
        sys.modules[fullname] = mock
        return mock

# Install the finder BEFORE any MaxText imports
sys.meta_path.insert(0, _PallasMockFinder())

# Set attributes on jax so direct attribute access works too
if not hasattr(jax, 'pallas'):
    jax.pallas = sys.modules.get('jax.pallas') or _PallasMockFinder().load_module('jax.pallas')
if not hasattr(jax.experimental, 'pallas'):
    jax.experimental.pallas = sys.modules.get('jax.experimental.pallas') or _PallasMockFinder().load_module('jax.experimental.pallas')

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

# ─────────────────────────────────────────────────────────────
# JAX distributed initialization
# Must happen before any Orbax checkpoint manager is created
# ─────────────────────────────────────────────────────────────
import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    try:
        jax.distributed.initialize()
    except Exception:
        pass
MOCK_EOF
# 9. Inject the mock loader at the top of train.py if not already injected
if ! grep -q "Mock internal Google-only modules" src/maxtext/trainers/pre_train/train.py; then
    cat mock_injector.py src/maxtext/trainers/pre_train/train.py > temp.py
    mv temp.py src/maxtext/trainers/pre_train/train.py
fi

# 10. Start the training run (PRODUCTION)
echo "=============================================="
echo "Starting MaxText Pretraining (PRODUCTION)..."
echo "  run_name: tinymath-1b-prod-run1"
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
