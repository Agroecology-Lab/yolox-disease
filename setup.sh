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
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "==> ERROR: python3.12 still not available after install attempt."
    echo "    See https://github.com/deadsnakes/deadsnakes -- your Ubuntu"
    echo "    release may not have a deadsnakes build yet; pyenv is the"
    echo "    fallback (https://github.com/pyenv/pyenv)."
    exit 1
fi

# YOLOX compiles a real C++ extension (cocoeval) that needs Python.h.
# Even if python3.12 itself was already present before this script ran,
# the -dev headers package may not be -- check separately.
if ! dpkg -l python3.12-dev >/dev/null 2>&1; then
    echo "==> python3.12-dev not found, installing (needs sudo)"
    sudo apt install -y python3.12-dev
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

pip install --upgrade pip setuptools wheel

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

# Upstream YOLOX's trainer.py unconditionally assumes a CUDA device exists
# (self.device hardcoded to "cuda:N", plus a bare torch.cuda.set_device()
# call in before_train()), so CPU-only training crashes even after
# dropping -d/--fp16 as the README suggests. This is a known, unresolved
# upstream issue (Megvii-BaseDetection/YOLOX#393). Patch it to fall back
# to CPU when no GPU is present. Idempotent: skipped if already patched.
python3 - "YOLOX/yolox/core/trainer.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "torch.cuda.is_available()" in src:
    print("==> trainer.py already patched for CPU training, skipping")
else:
    src = src.replace(
        'self.device = "cuda:{}".format(self.local_rank)',
        'self.device = "cuda:{}".format(self.local_rank) '
        'if torch.cuda.is_available() else "cpu"',
    )
    src = src.replace(
        "        torch.cuda.set_device(self.local_rank)\n",
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.set_device(self.local_rank)\n",
    )
    open(path, "w").write(src)
    print("==> Patched trainer.py for CPU training support")
EOF

# yolox/data/data_prefetcher.py is entirely CUDA-stream based (Stream(),
# .cuda(non_blocking=True), current_stream()) with no CPU path at all --
# a single is_available() guard isn't enough here, each method needs a
# CPU branch. Idempotent: skipped if already patched.
python3 - "YOLOX/yolox/data/data_prefetcher.py" <<'EOF'
import sys
path = sys.argv[1]
if "self.use_cuda" in open(path).read():
    print("==> data_prefetcher.py already patched for CPU training, skipping")
else:
    patched = '''#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.

    Patched: falls back to plain synchronous loading when no CUDA device is
    present, since upstream unconditionally assumes a CUDA stream exists.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.input_cuda()
                self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if self.use_cuda:
            if input is not None:
                self.record_stream(input)
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
'''
    open(path, "w").write(patched)
    print("==> Patched data_prefetcher.py for CPU training support")
EOF

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
sed -i 's/onnx-simplifier==0\.4\.10/onnx-simplifier>=0.4.10/' requirements.txt

# --no-build-isolation: YOLOX's setup.py imports torch to decide whether to
# precompile ops, but torch isn't declared as a build dependency in its
# pyproject.toml. Without this flag, pip builds in an isolated env that
# can't see the torch we just installed, so the import still fails even
# though torch is present in the venv.
pip install --no-build-isolation -v -e .
# requirements-train.txt redundantly re-installs yolox from git (it's
# already installed, patched, via -e . above). Re-fetching a pristine
# YOLOX checkout drags its original unpatched requirements.txt back in,
# reintroducing the onnx-simplifier==0.4.10 bug we just fixed. Strip that
# line before installing.
sed -i '/YOLOX\.git/d' ../requirements-train.txt
pip install --no-build-isolation -r ../requirements-train.txt

# yolox's own requirements pull in torchvision unpinned, from plain PyPI --
# which can be ABI-mismatched against the specific torch build we chose
# above (CPU-only or CUDA), causing errors like
# "RuntimeError: operator torchvision::nms does not exist" at import time.
# Reinstall both together from the same index so they're guaranteed
# compatible, overriding whatever yolox's deps just pulled in.
if [ "$USE_CUDA" -eq 1 ]; then
    pip install --upgrade torch torchvision
else
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

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

# 6. Download the Roboflow dataset export and convert it to COCO format.
# Needs a Roboflow API key (free account): https://app.roboflow.com/settings/api
# Get yours, then re-run as:
#   ROBOFLOW_API_KEY=your_key_here ./setup.sh
# Optional overrides (defaults shown):
#   ROBOFLOW_VERSION=1        -- dataset version number on Roboflow Universe
#   ROBOFLOW_FORMAT=yolov8    -- export format (must include a data.yaml)
#   DATA_DIR=~/data/plant-disease-coco  -- where the converted COCO dataset goes
ROBOFLOW_VERSION="${ROBOFLOW_VERSION:-1}"
ROBOFLOW_FORMAT="${ROBOFLOW_FORMAT:-yolov8}"
DATA_DIR="${DATA_DIR:-$HOME/data/plant-disease-coco}"

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
    echo "==> ROBOFLOW_API_KEY not set, skipping dataset download."
    echo "    Get a free API key at https://app.roboflow.com/settings/api"
    echo "    then re-run: ROBOFLOW_API_KEY=your_key_here ./setup.sh"
    echo "    (add ROBOFLOW_VERSION=N if the dataset version isn't 1)"
else
    pip install roboflow

    if [ -d "$DATA_DIR" ]; then
        echo "==> $DATA_DIR already exists, skipping download + conversion"
    else
        echo "==> Downloading Roboflow dataset (version $ROBOFLOW_VERSION, format $ROBOFLOW_FORMAT)"
        RF_EXPORT_DIR=$(python3 - "$ROBOFLOW_API_KEY" "$ROBOFLOW_VERSION" "$ROBOFLOW_FORMAT" <<'EOF'
import sys
from roboflow import Roboflow

api_key, version, fmt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rf = Roboflow(api_key=api_key)
project = rf.workspace("graduation-project-2023").project(
    "plants-diseases-detection-and-classification"
)
dataset = project.version(version).download(fmt)
print(dataset.location)
EOF
        )
        echo "==> Downloaded to $RF_EXPORT_DIR"
        echo "==> Converting to COCO format at $DATA_DIR"
        python3 dataset/yolo_to_coco.py --src "$RF_EXPORT_DIR" --dst "$DATA_DIR"
        echo "    ^ confirm the printed class order matches CLASS_NAMES in"
        echo "      exps/yolox_plant_nano.py and deploy/realtime_infer.py --"
        echo "      a mismatch trains fine but mislabels every detection later."
    fi
fi

echo "==> Setup complete."
if [ -d "$DATA_DIR" ]; then
    echo "    Dataset ready at $DATA_DIR. To train (from inside YOLOX/):"
    echo "      cd YOLOX"
    echo "      export PLANT_COCO_DIR=$DATA_DIR"
    if [ "$USE_CUDA" -eq 1 ]; then
        echo "      python tools/train.py -f exps/yolox_plant_nano.py -d 1 -b 32 --fp16 -o \\"
        echo "          -c weights/yolox_nano.pth"
    else
        echo "      python tools/train.py -f exps/yolox_plant_nano.py -b 32 \\"
        echo "          -c weights/yolox_nano.pth"
        echo "      (no -d/--fp16/-o: those assume a GPU. -o in particular"
        echo "      calls an unguarded torch.cuda.FloatTensor(...) and will"
        echo "      crash on CPU -- trainer.py has been patched separately"
        echo "      to handle the CUDA-device assumptions -o doesn't cover.)"
    fi
else
    echo "    Dataset not downloaded yet -- set ROBOFLOW_API_KEY and re-run"
    echo "    ./setup.sh, or run dataset/yolo_to_coco.py manually with your"
    echo "    own --src export directory and --dst destination."
fi
echo "    Export and board-deploy commands (tools/export_onnx.py,"
echo "    export_ncnn.sh, deploy/realtime_infer.py) run from $REPO_DIR,"
echo "    not from YOLOX/ -- see README.md steps 3-4."
