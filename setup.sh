#!/usr/bin/env bash
# One-time setup for yolox-disease. Run this from the repo root
# (the directory containing exps/, dataset/, deploy/, requirements-*.txt).
#
# Usage: ./setup.sh
# Re-running is safe: each step is skipped if already done.
# ./setup.sh --cuda gets you the full CUDA build if you're training on a local GPU.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "==> Repo root: $REPO_DIR"

# Parse args: default to CPU-only torch (no nvidia-* packages, no CUDA
# runtime download). Pass --cuda if this machine has a GPU you're training
# on. The Avaota A1 board never needs torch at all -- it runs NCNN.
USE_CUDA=0
for arg in "$@"; do
    case "$arg" in
        --cuda) USE_CUDA=1 ;;
        *) echo "Unknown option: $arg (only --cuda is supported)"; exit 1 ;;
    esac
done

# 1. venv
if [ ! -d ".venv" ]; then
    echo "==> Creating venv"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

# 2. torch FIRST -- YOLOX's setup.py imports torch during the build step
#    ("torch is required for pre-compiling ops"); installing it before
#    `pip install -e .` and before requirements-train.txt (which also
#    pip-installs YOLOX from git) avoids that failure.
if ! python -c "import torch" >/dev/null 2>&1; then
    if [ "$USE_CUDA" -eq 1 ]; then
        echo "==> Installing torch (CUDA build)"
        pip install torch
    else
        echo "==> Installing torch (CPU-only build, no nvidia-* packages)"
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "==> torch already installed, skipping"
fi

# 3. Clone YOLOX and drop our exp config in
if [ ! -d "YOLOX" ]; then
    echo "==> Cloning YOLOX"
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
else
    echo "==> YOLOX/ already exists, skipping clone"
fi
cp exps/yolox_plant_nano.py YOLOX/exps/

# 4. Editable install + training requirements (from inside YOLOX/)
pushd YOLOX >/dev/null
pip install -v -e .
pip install -r ../requirements-train.txt

# 5. COCO-pretrained nano backbone
mkdir -p weights
if [ ! -f "weights/yolox_nano.pth" ]; then
    echo "==> Downloading yolox_nano.pth"
    wget -O weights/yolox_nano.pth \
        https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
else
    echo "==> weights/yolox_nano.pth already present, skipping download"
fi
popd >/dev/null

echo "==> Setup complete."
echo "    Next: run dataset/yolo_to_coco.py, then train from inside YOLOX/"
echo "    (see README.md steps 1-4). Remember: dataset conversion, export,"
echo "    and board deploy commands run from $REPO_DIR, not from YOLOX/."
