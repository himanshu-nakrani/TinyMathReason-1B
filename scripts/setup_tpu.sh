#!/bin/bash
# setup_tpu.sh - automated MaxText setup for Spot TPU VMs

# 1. Clone MaxText if it doesn't exist
if [ ! -d "$HOME/maxtext" ]; then
    echo "Cloning MaxText..."
    git clone https://github.com/google/maxtext.git $HOME/maxtext
fi

cd $HOME/maxtext

# 2. Patch Python version requirement
sed -i 's/>=3.12/>=3.10/g' pyproject.toml

# 3. Install core dependencies bypassing hatchling isolation
echo "Installing dependencies..."
pip install --upgrade packaging hatchling hatch-requirements-txt editables
pip install -e . --no-build-isolation

# 4. Install missing runtime dependencies
pip install omegaconf 'protobuf<5.0.0' pydantic jaxtyping grain safetensors huggingface-hub aqtp google-cloud-storage absl-py optax

# 5. Force upgrade Flax from GitHub for bleeding-edge nnx
pip install --upgrade --force-reinstall git+https://github.com/google/flax.git

# 6. Backwards compatibility patch for Pytree -> Object
find src/ -name "*.py" -exec sed -i 's/from flax.nnx import Pytree/from flax.nnx import Object as Pytree/g' {} +

# 7. Inject mocks for internal Google modules (pathwaysutils, qwix)
cat << 'EOF' > mock_injector.py
import sys
from unittest.mock import MagicMock

# Mock internal Google pathways
sys.modules["pathwaysutils"] = MagicMock()
sys.modules["pathwaysutils.elastic"] = MagicMock()
sys.modules["pathwaysutils.elastic.manager"] = MagicMock()

# Mock qwix (quantization not used for bfloat16 pretraining)
sys.modules["qwix"] = MagicMock()
sys.modules["qwix.pallas"] = MagicMock()
EOF

# Inject the mock loader at the top of train.py if not already injected
if ! grep -q "mock_injector" src/maxtext/trainers/pre_train/train.py; then
    cat mock_injector.py src/maxtext/trainers/pre_train/train.py > temp.py
    mv temp.py src/maxtext/trainers/pre_train/train.py
fi

# 8. Start the training run (SMOKE TEST)
echo "Starting MaxText Pretraining (SMOKE TEST)..."
PYTHONPATH=src python3 src/maxtext/trainers/pre_train/train.py MaxText/configs/tinymath_1b.yml run_name=tinymath-1b-smoke steps=100
