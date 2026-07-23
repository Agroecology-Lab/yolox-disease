"""
Convert the existing Roboflow "YOLOv8 bounding box" export (images/ + labels/ +
data.yaml, per split) into COCO json, which is what YOLOX's training pipeline
(yolox.data.datasets.COCODataset) expects out of the box.

This is the only dataset-side change needed to move from the Ultralytics
YOLOv8 pipeline to YOLOX -- the images and boxes are unchanged, only the
annotation *format* changes.

Expected input layout (this is what Roboflow exports and what the original
notebook trained from):

    plants-diseases-detection-and-classification/
        data.yaml
        train/images/*.jpg  train/labels/*.txt
        valid/images/*.jpg  valid/labels/*.txt
        test/images/*.jpg   test/labels/*.txt

Output layout (what YOLOX/COCODataset wants):

    COCO/
        train2017/*.jpg
        val2017/*.jpg
        annotations/instances_train2017.json
        annotations/instances_val2017.json

Usage:
    python dataset/yolo_to_coco.py \
        --src /path/to/plants-diseases-detection-and-classification \
        --dst /path/to/COCO

IMPORTANT: `data.yaml`'s `names:` list defines the class-id -> class-name
mapping and therefore the exact order the trained best.pt's `model.names`
used. Copy that same order into exps/yolox_plant_nano.py (CLASS_NAMES) or
detections will be labelled with the wrong disease names after conversion.
This script does that automatically by reading data.yaml, but double check
the printed class list against your Roboflow project before training.
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image


def load_class_names(data_yaml: Path):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):  # Roboflow sometimes exports {0: 'a', 1: 'b', ...}
        names = [names[i] for i in range(len(names))]
    return names


def convert_split(src_root: Path, dst_root: Path, split_src: str, split_dst: str, class_names):
    img_dir = src_root / split_src / "images"
    lbl_dir = src_root / split_src / "labels"
    if not img_dir.exists():
        print(f"[skip] {img_dir} does not exist")
        return None

    out_img_dir = dst_root / split_dst
    out_img_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    img_id = 1
    skipped = 0

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    for img_path in img_files:
        with Image.open(img_path) as im:
            w, h = im.size

        out_name = img_path.name
        shutil.copy2(img_path, out_img_dir / out_name)

        images.append(
            {"id": img_id, "file_name": out_name, "width": w, "height": h}
        )

        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    # YOLO format is normalized center-x, center-y, width, height
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    # Clip to image bounds the same way YOLOX's COCODataset
                    # loader does (yolox/data/datasets/coco.py), then drop
                    # boxes that clip down to zero width/height. That loader
                    # only rejects x2 < x1 / y2 < y1 (not <=), so an
                    # edge-touching or near-zero-size YOLO box survives into
                    # training with zero real area, zero anchors ever match
                    # it, and get_assignments() crashes with "selected index
                    # k out of range" in simota_matching(). Better to drop it
                    # here, once, than debug that traceback per bad image.
                    x1, y1 = max(0.0, x), max(0.0, y)
                    x2 = min(float(w), x1 + max(0.0, box_w))
                    y2 = min(float(h), y1 + max(0.0, box_h))
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        skipped += 1
                        continue

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id + 1,  # COCO category ids are 1-indexed
                            "bbox": [x, y, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
        img_id += 1

    categories = [
        {"id": i + 1, "name": name, "supercategory": "plant_disease"}
        for i, name in enumerate(class_names)
    ]

    if skipped:
        print(f"[{split_dst}] skipped {skipped} degenerate (zero-area after clip) boxes")

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Roboflow export root (contains data.yaml)")
    ap.add_argument("--dst", required=True, help="Output COCO-style root")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    (dst_root / "annotations").mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(src_root / "data.yaml")
    print(f"Loaded {len(class_names)} classes from data.yaml:")
    for i, n in enumerate(class_names):
        print(f"  {i}: {n}")

    # Roboflow's val split is usually "valid" -- YOLOX/COCODataset expects val2017.
    split_map = [("train", "train2017"), ("valid", "val2017"), ("test", "test2017")]

    for split_src, split_dst in split_map:
        coco = convert_split(src_root, dst_root, split_src, split_dst, class_names)
        if coco is None:
            continue
        ann_name = f"instances_{split_dst}.json"
        with open(dst_root / "annotations" / ann_name, "w") as f:
            json.dump(coco, f)
        print(
            f"[{split_dst}] {len(coco['images'])} images, "
            f"{len(coco['annotations'])} boxes -> {dst_root / 'annotations' / ann_name}"
        )

    print("\nDone. Point exps/yolox_plant_nano.py's data_dir at:", dst_root)


if __name__ == "__main__":
    main()
