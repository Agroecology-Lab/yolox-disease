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

# yolox/exp/yolox_base.py's random_resize() (multiscale training) also
# unconditionally does torch.LongTensor(2).cuda() -- only needed to support
# a multi-GPU NCCL broadcast that a single-process CPU/single-GPU run never
# reaches. Idempotent: skipped if already patched.
python3 - "YOLOX/yolox/exp/yolox_base.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "tensor = torch.LongTensor(2).cuda()"
if old not in src:
    print("==> yolox_base.py already patched (or line not found), skipping")
else:
    new = (
        "tensor = torch.LongTensor(2)\n"
        "        if torch.cuda.is_available():\n"
        "            tensor = tensor.cuda()"
    )
    open(path, "w").write(src.replace(old, new))
    print("==> Patched yolox_base.py random_resize for CPU training support")
EOF

# yolox/models/yolo_head.py's simota_matching() assumes at least one anchor
# always falls inside/near every gt box. When none do -- small, edge-clipped,
# or heavily mosaic/affine-augmented boxes, especially at the smaller end of
# multiscale training -- torch.topk(cost[gt_idx], k=1) on a length-0 dim
# crashes with "selected index k out of range". This is a known, unresolved
# upstream issue (Megvii-BaseDetection/YOLOX#777, #811, #1096, #1219) that
# surfaces partway through training, not at startup, so it's easy to lose an
# hour of progress to before noticing. Idempotent: skipped if already patched.
python3 - "YOLOX/yolox/models/yolo_head.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = (
    "    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):\n"
    "        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)\n"
    "\n"
    "        n_candidate_k = min(10, pair_wise_ious.size(1))"
)
if "pair_wise_ious.shape[1] == 0" in src:
    print("==> yolo_head.py already patched (or line not found), skipping")
else:
    new = (
        "    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):\n"
        "        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)\n"
        "\n"
        "        if pair_wise_ious.shape[1] == 0:\n"
        "            # No anchor point falls inside/near ANY gt box for this image --\n"
        "            # treat as zero foreground matches instead of crashing; fg_mask\n"
        "            # is already all-False here, so nothing else needs updating.\n"
        "            num_fg = 0\n"
        "            gt_matched_classes = cost.new_zeros(0)\n"
        "            pred_ious_this_matching = cost.new_zeros(0)\n"
        "            matched_gt_inds = cost.new_zeros(0, dtype=torch.int64)\n"
        "            return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds\n"
        "\n"
        "        n_candidate_k = min(10, pair_wise_ious.size(1))"
    )
    if old not in src:
        print("==> WARNING: expected simota_matching text not found, skipped -- check yolo_head.py manually")
    else:
        open(path, "w").write(src.replace(old, new))
        print("==> Patched yolo_head.py simota_matching for zero-in-box-anchor crash")
EOF

# yolox/evaluators/coco_evaluator.py's evaluate() unconditionally builds
# torch.cuda.FloatTensor/HalfTensor for the eval image batch and the
# inference/nms-time stats tensor -- eval_interval=5 in our exp means this
# runs every 5 epochs, so a CPU-only run trains fine for epochs 1-4 and only
# crashes at the first eval, deep into a run, with "Cannot initialize CUDA
# without ATen_cuda library". Idempotent: skipped if already patched.
python3 - "YOLOX/yolox/evaluators/coco_evaluator.py" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
if "torch.cuda.is_available()" in src:
    print("==> coco_evaluator.py already patched for CPU eval, skipping")
else:
    old1 = (
        "        # TODO half to amp_test\n"
        "        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor"
    )
    new1 = (
        "        # TODO half to amp_test\n"
        "        if torch.cuda.is_available():\n"
        "            tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor\n"
        "        else:\n"
        "            tensor_type = torch.HalfTensor if half else torch.FloatTensor"
    )
    old2 = "        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])"
    new2 = (
        "        if torch.cuda.is_available():\n"
        "            statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])\n"
        "        else:\n"
        "            statistics = torch.FloatTensor([inference_time, nms_time, n_samples])"
    )
    if old1 not in src or old2 not in src:
        print("==> WARNING: expected coco_evaluator.py text not found, skipped -- check manually")
    else:
        src = src.replace(old1, new1).replace(old2, new2)
        open(path, "w").write(src)
        print("==> Patched coco_evaluator.py for CPU eval support")
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
    fi
fi

# yolo_to_coco.py uses ALL classes from the Roboflow data.yaml (29 for this
# dataset), but exps/yolox_plant_nano.py and deploy/realtime_infer.py may
# still have a smaller hand-curated CLASS_NAMES list -- that mismatch
# crashes training ("Class values must be smaller than num_classes") as
# soon as a batch contains a class outside the curated subset. Sync both
# files' CLASS_NAMES to the authoritative order already baked into the
# converted dataset's COCO json, so this can't drift out of sync. Runs
# every time (idempotent: no-ops if already matching), regardless of
# whether the dataset was just downloaded above or already existed.
ANNOTATIONS="$DATA_DIR/annotations/instances_train2017.json"
if [ -f "$ANNOTATIONS" ]; then
    python3 - "$ANNOTATIONS" exps/yolox_plant_nano.py deploy/realtime_infer.py <<'EOF'
import json
import re
import sys

ann_path, exp_path, infer_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(ann_path) as f:
    categories = json.load(f)["categories"]
names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]

block = "CLASS_NAMES = (\n" + "".join(f'    "{n}",\n' for n in names) + ")"
pattern = re.compile(r"CLASS_NAMES = \(.*?\)", re.DOTALL)

for path in (exp_path, infer_path):
    src = open(path).read()
    new_src, count = pattern.subn(block, src, count=1)
    if count == 0:
        print(f"==> WARNING: no CLASS_NAMES block found in {path}, skipped")
    elif new_src == src:
        print(f"==> {path}: CLASS_NAMES already in sync ({len(names)} classes)")
    else:
        open(path, "w").write(new_src)
        print(f"==> Updated {path}: CLASS_NAMES now has {len(names)} classes")
EOF
    # Re-copy into YOLOX/ since the exp file was already copied there earlier
    # (before this sync ran) and training reads exps/ from inside YOLOX/.
    cp exps/yolox_plant_nano.py YOLOX/exps/yolox_plant_nano.py
else
    echo "==> No converted dataset found yet at $ANNOTATIONS, skipping CLASS_NAMES sync"
    echo "    (set ROBOFLOW_API_KEY and re-run to download + convert first)"
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
