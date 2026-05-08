#!/bin/bash
# setup_tpu.sh - automated MaxText setup for Spot TPU VMs

# 1. Clone MaxText if it doesn't exist
if [ ! -d "$HOME/maxtext" ]; then
    echo "Cloning MaxText..."
    git clone https://github.com/google/maxtext.git $HOME/maxtext
fi

cd $HOME/maxtext

# Clean repository files to prevent duplicate injections on multiple runs
git checkout -- src/maxtext/trainers/pre_train/train.py
git checkout -- src/maxtext/utils/sharding.py

# 2. Patch Python version requirement
sed -i 's/>=3.12/>=3.10/g' pyproject.toml

# 3. Install core dependencies bypassing hatchling isolation
echo "Installing dependencies..."
pip install --upgrade packaging hatchling hatch-requirements-txt editables
pip install -e . --no-build-isolation

# 4. Install missing runtime dependencies
pip install omegaconf 'protobuf<5.0.0' pydantic jaxtyping grain safetensors huggingface-hub aqtp google-cloud-storage absl-py optax tensorflow-cpu transformers tokenizers tiktoken sentencepiece sympy Pillow ml_goodput_measurement cloud_tpu_diagnostics
echo "Upgrading JAX to JAX-nightly TPU release..."
pip install -U --pre jax jaxlib libtpu-nightly requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 5. Force upgrade Flax from GitHub for bleeding-edge nnx
pip install --upgrade --force-reinstall git+https://github.com/google/flax.git

# 6. Backwards compatibility patch for Pytree -> Object and reshard
find src/ -name "*.py" -exec sed -i 's/from flax.nnx import Pytree/from flax.nnx import Object as Pytree/g' {} +
cat << 'EOF' > patch_sharding.py
import os
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

# 7. Inject mocks for internal Google modules (pathwaysutils, qwix)
cat << 'EOF' > mock_injector.py
import sys
import functools
import jax
from unittest.mock import MagicMock

# Monkey-patch jax.jit to support decorator factory pattern (broken in recent JAX nightly)
if getattr(jax, "_is_monkey_patched", False) is False:
    _original_jit = jax.jit
    def _patched_jit(fun=None, **kwargs):
        if fun is None:
            return functools.partial(_original_jit, **kwargs)
        return _original_jit(fun, **kwargs)
    jax.jit = _patched_jit
    jax._src.api.jit = _patched_jit
    jax._is_monkey_patched = True

# Mock internal Google pathways
sys.modules["pathwaysutils"] = MagicMock()
sys.modules["pathwaysutils.elastic"] = MagicMock()
sys.modules["pathwaysutils.elastic.manager"] = MagicMock()

# Mock qwix (quantization not used for bfloat16 pretraining)
sys.modules["qwix"] = MagicMock()
sys.modules["qwix.pallas"] = MagicMock()
sys.modules["qwix._src"] = MagicMock()
sys.modules["qwix._src.core"] = MagicMock()
sys.modules["qwix.contrib"] = MagicMock()
sys.modules["qwix.contrib.sparsity"] = MagicMock()

# Mock tokamax (experimental attention kernels not used by default)
sys.modules["tokamax"] = MagicMock()
sys.modules["tokamax._src"] = MagicMock()
sys.modules["tokamax._src.ops"] = MagicMock()
sys.modules["tokamax._src.ops.experimental"] = MagicMock()
sys.modules["tokamax._src.ops.experimental.tpu"] = MagicMock()
sys.modules["tokamax._src.ops.experimental.tpu.splash_attention"] = MagicMock()

# Mock drjax (internal Google disaster recovery module)
sys.modules["drjax"] = MagicMock()
EOF

# Inject the mock loader at the top of train.py if not already injected
if ! grep -q "Monkey-patch jax.jit" src/maxtext/trainers/pre_train/train.py; then
    cat mock_injector.py src/maxtext/trainers/pre_train/train.py > temp.py
    mv temp.py src/maxtext/trainers/pre_train/train.py
fi

# 8. Start the training run (SMOKE TEST)
echo "Starting MaxText Pretraining (SMOKE TEST)..."
PYTHONPATH=src python3 src/maxtext/trainers/pre_train/train.py maxtext_config.yml run_name=tinymath-1b-smoke steps=100
