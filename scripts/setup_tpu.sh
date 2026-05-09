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

# 1. Clone MaxText if it doesn't exist
if [ ! -d "$HOME/maxtext" ]; then
    echo "Cloning MaxText..."
    git clone https://github.com/AI-Hypercomputer/maxtext.git $HOME/maxtext
fi

# Copy the config file into the maxtext directory
if [ -f "$HOME/maxtext_config.yml" ]; then
    cp "$HOME/maxtext_config.yml" "$HOME/maxtext/"
fi

cd $HOME/maxtext

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
    tensorboardX

# 4. Install stable JAX with TPU support
echo "Purging conflicting JAX and TPU nightly builds..."
pip uninstall -y jax jaxlib libtpu libtpu-nightly 2>/dev/null || true
echo "Installing stable JAX with TPU support..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 7. Backwards compatibility patch for Pytree -> Object and reshard
find src/ -name "*.py" -exec sed -i 's/from flax.nnx import Pytree/from flax.nnx import Object as Pytree/g' {} +

# Patch Flax 0.11+ API: .get_value() -> .value (Flax 0.10.5 uses .value property)
find src/ -name "*.py" -exec sed -i 's/\.get_value()/.value/g' {} +

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

    # Shim for jax.set_mesh (added in JAX 0.7.1, missing in 0.6.x)
    if not hasattr(jax, 'set_mesh'):
        import contextlib
        @contextlib.contextmanager
        def _set_mesh_shim(mesh):
            with mesh:
                yield
        jax.set_mesh = _set_mesh_shim

    jax._is_monkey_patched = True

# ─────────────────────────────────────────────────────────────
# Mock internal Google-only modules
# ─────────────────────────────────────────────────────────────
for mod in [
    "pathwaysutils", "pathwaysutils.elastic", "pathwaysutils.elastic.manager",
    "qwix", "qwix.pallas", "qwix._src", "qwix._src.core",
    "qwix.contrib", "qwix.contrib.sparsity",
    "tokamax", "tokamax._src", "tokamax._src.ops",
    "tokamax._src.ops.experimental", "tokamax._src.ops.experimental.tpu",
    "tokamax._src.ops.experimental.tpu.splash_attention",
    "drjax",
]:
    sys.modules[mod] = MagicMock()

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

# 10. Start the training run (SMOKE TEST)
echo "=============================================="
echo "Starting MaxText Pretraining (SMOKE TEST)..."
echo "  dataset_type: synthetic"
echo "  steps: 100"
echo "=============================================="
PYTHONPATH=src python3 src/maxtext/trainers/pre_train/train.py \
    maxtext_config.yml \
    run_name=tinymath-1b-smoke \
    steps=100
