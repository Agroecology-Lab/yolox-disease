#!/usr/bin/env bash
# Convert the exported ONNX graph to NCNN (.param/.bin), then optimize and
# optionally int8-quantize it for the Avaota A1's Cortex-A55 NEON pipeline.
#
# Prerequisite: build ncnn's tools (onnx2ncnn, ncnnoptimize, ncnn2int8) from
# https://github.com/Tencent/ncnn -- these are small C++ CLI tools, build
# once on your dev machine (x86), the resulting .param/.bin files are
# architecture-independent and just get copied to the board.
#
# Usage:
#   ./tools/export_ncnn.sh plant_disease_nano.onnx calib_images/
#
# calib_images/ is a folder of ~200-500 representative field images (JPEGs),
# only needed for the int8 step -- use held-out validation images, not
# training images.
set -euo pipefail

ONNX_PATH="${1:?usage: export_ncnn.sh <model.onnx> [calib_image_dir]}"
CALIB_DIR="${2:-}"
NAME="$(basename "${ONNX_PATH}" .onnx)"

echo "[1/3] onnx2ncnn"
onnx2ncnn "${ONNX_PATH}" "${NAME}.param" "${NAME}.bin"

echo "[2/3] ncnnoptimize (fp16 storage, fuses conv+bn+activation)"
ncnnoptimize "${NAME}.param" "${NAME}.bin" "${NAME}-opt.param" "${NAME}-opt.bin" 65536

if [[ -n "${CALIB_DIR}" ]]; then
  echo "[3/3] int8 calibration + quantization"
  # ncnn2table needs a plain list of image paths, one per line
  find "${CALIB_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    > "${NAME}-calib-list.txt"

  # mean=0/norm=1 (i.e. no scaling) to match deploy/realtime_infer.py's raw
  # 0-255 BGR input -- YOLOX's default preproc does not normalize. Get this
  # wrong and the calibration table is built for a different input
  # distribution than what real inference sends, on top of quantizing an
  # already-wrong range.
  ncnn2table "${NAME}-opt.param" "${NAME}-opt.bin" "${NAME}-calib-list.txt" \
    "${NAME}.table" mean=[0,0,0] norm=[1,1,1] \
    shape=[320,320,3] pixel=BGR thread=4 method=kl

  ncnn2int8 "${NAME}-opt.param" "${NAME}-opt.bin" "${NAME}-int8.param" \
    "${NAME}-int8.bin" "${NAME}.table"

  echo "Done -- deploy ${NAME}-int8.param / ${NAME}-int8.bin (~4x smaller, fastest)"
  echo "If accuracy on real field images regresses, fall back to ${NAME}-opt.param/.bin (fp16, no calibration needed)."
else
  echo "Done -- deploy ${NAME}-opt.param / ${NAME}-opt.bin (fp16)"
  echo "Pass a calibration image directory as \$2 to also produce an int8 model."
fi
