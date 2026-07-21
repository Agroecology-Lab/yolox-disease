# encoding: utf-8
"""
YOLOX experiment config for the plant disease dataset, sized for real-time
CPU-only inference on the Avaota A1 (Allwinner T527, 8x Cortex-A55, no usable
NPU under mainline/Armbian -- see README.md "Why YOLOX, why Nano" section).

This replaces the notebook's:
    yolo task=detect mode=train model=yolov8l.pt data=data.yaml epochs=50 imgsz=224

Run with (from a YOLOX checkout, this repo's exps/ dropped alongside it):
    python tools/train.py -f exps/yolox_plant_nano.py -d 1 -b 32 --fp16 -o \
        -c weights/yolox_nano.pth   # COCO-pretrained backbone, recommended
"""
import os

from yolox.exp import Exp as MyExp


# NOTE: this order MUST match the `names:` list in your Roboflow data.yaml.
# These are the 10 classes from the original app.py DISEASE_INFO dict -- update
# if your data.yaml orders them differently (yolo_to_coco.py prints the order
# it found; copy it here verbatim).
CLASS_NAMES = (
    "Apple Scab",
    "Apple Rust",
    "Corn Leaf Blight",
    "Corn Gray Spot",
    "Potato Early Blight",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Healthy Leaf",
)


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---- model size ----
        # "nano" depth/width: ~0.9M params, the only YOLOX variant that runs
        # comfortably in real time (target >=10 FPS) on a Cortex-A55 CPU with
        # no NPU acceleration. If the vendor NPU BSP ever becomes viable,
        # yolox_tiny/yolox_s would trade a few FPS for meaningfully better
        # small-lesion recall -- worth revisiting then, not now.
        self.depth = 0.33
        self.width = 0.25
        self.act = "silu"

        # depthwise conv further cuts FLOPs for ARM NEON at a small accuracy
        # cost -- worth it here given the CPU-only target.
        self.depthwise = True

        # ---- input resolution ----
        # 320x320 rather than YOLOX's usual 416/640: the source dataset was
        # originally trained at imgsz=224 (see notebook), leaf lesions are
        # small low-frequency blobs, and 320 buys real-time headroom on-device.
        # Bump to (416, 416) if validation mAP on small lesions is too low
        # and you can spare the latency.
        self.input_size = (320, 320)
        self.test_size = (320, 320)
        self.random_size = (10, 20)  # multiscale range during training, x32

        # ---- dataset ----
        self.num_classes = len(CLASS_NAMES)
        self.data_dir = os.getenv(
            "PLANT_COCO_DIR", "/data/plant-disease-coco"
        )  # output of dataset/yolo_to_coco.py
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # ---- training schedule ----
        # ~2,600 images total is small for YOLOX's default 300-epoch/strong-
        # augmentation recipe -- it will overfit or the augmentation will
        # drown the (already small) signal. Shorter schedule, gentler aug,
        # and starting from COCO-pretrained yolox_nano.pth (via -c) instead
        # of random init is the important change here.
        self.max_epoch = 80
        self.no_aug_epochs = 10  # turn off mosaic/mixup for the last N epochs
        self.warmup_epochs = 3
        self.eval_interval = 5

        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0  # mixup blends two whole leaves together -- not
        # meaningful for this dataset's disease-lesion classes, disable it.
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False

        self.data_num_workers = 4
        self.seed = 42

        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        # Ensures self.num_classes / self.depthwise etc. from above are
        # actually threaded through -- default Exp.get_model() already does
        # this, kept explicit here as a reminder of what's active.
        return super().get_model()
