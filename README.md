# Plant disease detection: YOLOv8 → YOLOX, ported for the Avaota A1

This replaces the Ultralytics YOLOv8 pipeline in muqadasejaz`[Plant-Detection-using-YOLOv8](https://github.com/muqadasejaz/Plant-Detection-using-YOLOv8)`
with a YOLOX pipeline sized for real-time, CPU-only inference on the Avaota A1
(Allwinner T527, 8x Cortex-A55, Mali-G56/G57). It is a re-architecture, not a
weight conversion — see "What this can't do" below before you rely on it.

## Why YOLOX instead of YOLOv8

- **No Ultralytics/AGPL runtime dependency on the board.** YOLOX (Apache-2.0)
  ships official ONNX → NCNN export paths and NCNN has no PyTorch/libtorch
  dependency at inference time — important on a 1-4GB RAM SBC.
- **NCNN is the realistic acceleration path here, not the NPU.** The T527's
  NPU (up to 2 TOPS) needs Allwinner's vendor BSP/Android kernel and closed
  toolchain to drive it. Your Avaota A1 bring-up has been on mainline U-Boot
  and Armbian — there's no mainline NPU driver on that path, so this pipeline
  targets Cortex-A55 NEON via NCNN, not the NPU. If the vendor BSP ever
  becomes viable, YOLOX still has an NPU-adjacent path forward via NCNN's
  Vulkan backend on the Mali GPU, worth revisiting then.
- **YOLOX-Nano is a genuinely small model** (~0.9M params at depth=0.33,
  width=0.25) built for exactly this class of ARM SoC, unlike YOLOv8's
  smallest variant which still assumes more headroom.

## What changed vs. the original repo

| | Original | This port |
|---|---|---|
| Model | YOLOv8 (l in notebook, n in README table — inconsistent) | YOLOX-Nano, depthwise |
| Framework at inference | `ultralytics` (torch) | `ncnn` (no torch) |
| Input size | 224 (train) / 640 (README metrics) | 320x320 |
| Serving | Streamlit image/video upload | Live V4L2 camera loop |
| Annotation format | YOLO txt | COCO json (`dataset/yolo_to_coco.py`) |

## Files

- `dataset/yolo_to_coco.py` — converts the existing Roboflow YOLO-format
  export to the COCO json YOLOX trains on. **Run this first.**
- `exps/yolox_plant_nano.py` — YOLOX experiment config: 10 classes, 320x320,
  80-epoch schedule tuned down for a ~2,600-image dataset (see comments —
  mixup disabled, mosaic reduced, meant to start from COCO-pretrained
  `yolox_nano.pth`, not random init).
- `tools/export_onnx.py` — trained checkpoint → ONNX, decode left out of the
  graph on purpose (see next file).
- `tools/export_ncnn.sh` — ONNX → NCNN `.param`/`.bin`, with an optional int8
  calibration/quantization pass.
- `deploy/realtime_infer.py` — runs **on the board**: V4L2 camera capture,
  YOLOX grid/stride decode + NMS in numpy, draws boxes, prints an FPS-tagged
  detection summary. This is the on-device equivalent of `app.py`.
- `requirements-train.txt` / `requirements-board.txt` — kept separate on
  purpose; nothing in the training file should end up installed on the SBC.

## Pipeline

```bash
# 0. One-time setup: clone YOLOX and drop this repo's exps/ into it.
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cp exps/yolox_plant_nano.py YOLOX/exps/
cd YOLOX
pip install -v -e .              # editable install, needed for tools/train.py
pip install -r ../requirements-train.txt

# Grab the COCO-pretrained nano backbone (fine-tuning from this, not random
# init, is what makes an 80-epoch run on ~2,600 images viable):
mkdir -p weights
wget -O weights/yolox_nano.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

# 1. On your dev machine, with the Roboflow export downloaded:
python ../dataset/yolo_to_coco.py --src /path/to/plants-diseases-detection-and-classification --dst /data/plant-disease-coco
# Prints the class order it read from data.yaml -- confirm it matches
# CLASS_NAMES in exps/yolox_plant_nano.py and deploy/realtime_infer.py before
# training; a mismatch trains fine but mislabels every detection later.

# 2. Train (from inside the YOLOX checkout, exps/yolox_plant_nano.py copied in above):
export PLANT_COCO_DIR=/data/plant-disease-coco
python tools/train.py -f exps/yolox_plant_nano.py -d 1 -b 32 --fp16 -o \
    -c weights/yolox_nano.pth
# -d 1: single GPU/device. Drop -d and --fp16 if training CPU-only (slow but
# works for a dataset this small). Checkpoints land in
# YOLOX_outputs/yolox_plant_nano/{latest,best}_ckpt.pth; eval_interval=5 in
# the exp means val mAP prints every 5 epochs so you can catch overfitting
# early rather than waiting for all 80.

# Optional: evaluate the best checkpoint standalone before exporting
python tools/eval.py -f exps/yolox_plant_nano.py -c YOLOX_outputs/yolox_plant_nano/best_ckpt.pth -d 1 -b 32

# 3. Export (back in this repo's directory, or with tools/export_onnx.py on your PYTHONPATH):
python tools/export_onnx.py -f exps/yolox_plant_nano.py -c YOLOX/YOLOX_outputs/yolox_plant_nano/best_ckpt.pth
./tools/export_ncnn.sh plant_disease_nano.onnx /data/plant-disease-coco/val2017

# 4. Copy plant_disease_nano-int8.param/.bin to the Avaota A1, then:
pip install -r requirements-board.txt
python deploy/realtime_infer.py --param plant_disease_nano-int8.param --bin plant_disease_nano-int8.bin --source /dev/video0 --headless
```


## What this can't do

- **No weight transplant.** `best.pt` isn't in the repo (`app.py` pulls it
  from Google Drive via a secret you don't have here), and YOLOv8's C2f
  backbone/head isn't weight-compatible with YOLOX's CSPDarknet/decoupled
  head anyway — this is a from-scratch retrain on the same dataset, not a
  conversion of the existing weights.
- **Class order is assumed, not confirmed.** `CLASS_NAMES` in both
  `exps/yolox_plant_nano.py` and `deploy/realtime_infer.py` is copied from
  `app.py`'s `DISEASE_INFO` dict order. That dict is keyed by *name string*
  at runtime in the original app, so it never actually depended on training
  order — this port does, because NCNN output is just class-index logits.
  Confirm against the real `data.yaml` (`yolo_to_coco.py` prints it) before
  trusting on-device labels.
- **No FPS numbers yet.** I haven't run this on real Avaota A1 hardware or
  the actual dataset — there's no board, no trained weights, and no dataset
  in this environment to validate against. 320x320 YOLOX-Nano int8 on 4x
  Cortex-A55 typically lands somewhere in the 15-30 FPS range on comparable
  SoCs, but treat that as a starting estimate, not a measured result — bench
  it once you have int8 weights, and drop input size to e.g. 256x256 first
  if you're short of your real-time target rather than pruning further.
- **NMS via `cv2.dnn.NMSBoxes`** assumes `opencv-python-headless` is
  available on-device (it's in `requirements-board.txt`) — if you want to
  drop OpenCV entirely for a smaller footprint, that one call is the only
  thing standing in the way; swappable for a ~15-line numpy NMS.
