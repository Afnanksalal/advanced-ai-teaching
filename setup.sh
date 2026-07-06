#!/usr/bin/env bash
# Advanced AI course: one-time setup (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Advanced AI: Course Setup ==="
python3 --version

if [ ! -d .venv ]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch..."
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch torchvision torchaudio
fi

echo "Installing requirements..."
pip install -r requirements.txt

echo "Registering Jupyter kernel..."
python configure_kernel.py

echo "Prefetching bundled datasets and sample images..."
python download_datasets.py --minimal

echo
echo "=== Setup complete ==="
echo "Start labs:  ./env.sh"
echo "Env only:    source ./env.sh --env-only"
echo
echo "Large datasets download on first lab run."
echo "Full prefetch: python download_datasets.py"
