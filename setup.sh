#!/usr/bin/env bash
# One-time setup for yolox-disease. Run this from the repo root
# (the directory containing exps/, dataset/, deploy/, requirements-*.txt).
#
# Usage: ./setup.sh
# Re-running is safe: each step is skipped if already done.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "==> Repo root: $REPO_DIR"

# Python 3.14 (the current Ubuntu default) has no prebuilt wheels yet for
# onnx-simplifier and other deps in this chain, forcing source builds that
# hit further breakage (missing cmake, setuptools rejecting the package's
# 'unknown' version fallback). Python 3.12 has wheels for everything here,
# so prefer it, installing via deadsnakes if it's not already present.
PYTHON_BIN="python3.12"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "==> python3.12 not found, installing via deadsnakes PPA (needs sudo)"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "==> ERROR: python3.12 still not available after install attempt."
    echo "    See https://github.com/deadsnakes/deadsnakes -- your Ubuntu"
    echo "    release may not have a deadsnakes build yet; pyenv is the"
    echo "    fallback (https://github.com/pyenv/pyenv)."
    exit 1
fi

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
if [ -d ".venv" ] && ! .venv/bin/python --version 2>&1 | grep -q "3\.12"; then
    echo "==> Existing .venv is not Python 3.12, recreating"
    rm -rf .venv
fi
if [ ! -d ".venv" ]; then
    echo "==> Creating venv (python3.12)"
    "$PYTHON_BIN" -m venv .venv
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

# yolox's setup.py hard-pins onnx-simplifier==0.4.10, a version old enough
# that PyPI never published a wheel for it -- so pip always builds it from
# source, and that old sdist's setup.py derives its version via `git
# describe`, which fails outside a git checkout and falls back to the
# literal string 'unknown', which no setuptools version can both accept
# and still support current Python (older setuptools tolerates it but
# breaks on 3.12+; newer setuptools supports 3.12+ but rejects it).
# Relaxing the pin lets pip pick a modern onnx-simplifier release that
# ships a prebuilt wheel, sidestepping the source build entirely.
sed -i 's/onnx-simplifier==0\.4\.10/onnx-simplifier>=0.4.10/' setup.py

# --no-build-isolation: YOLOX's setup.py imports torch to decide whether to
# precompile ops, but torch isn't declared as a build dependency in its
# pyproject.toml. Without this flag, pip builds in an isolated env that
# can't see the torch we just installed, so the import still fails even
# though torch is present in the venv.
pip install --no-build-isolation -v -e .
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
