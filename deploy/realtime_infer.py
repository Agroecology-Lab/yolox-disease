"""
Real-time plant disease detection on the Avaota A1, replacing app.py's
Streamlit upload-and-predict flow with a live camera loop.

Runtime is NCNN (not PyTorch/Ultralytics) -- there is no ultralytics or torch
dependency on the board at all, which matters on an 8x Cortex-A55 SBC with
1-4GB RAM. `pip install ncnn` gets you prebuilt ARM64 NEON wheels.

Decoding YOLOX's raw head output (grid + stride based, anchor-free) and NMS
are done here in plain numpy rather than in the network graph, since ncnn has
no YOLOX decode op and this keeps the .param/.bin graph minimal.

Usage:
    python deploy/realtime_infer.py \
        --param plant_disease_nano-int8.param \
        --bin plant_disease_nano-int8.bin \
        --source /dev/video0 \
        --conf 0.4

--source accepts a V4L2 device path (MIPI CSI camera exposed via the driver
as /dev/videoN), an RTSP/HTTP stream URL, or a video file for offline testing
(the equivalent of the original run_video_detection()).
"""
import argparse
import time

import cv2
import numpy as np
import ncnn

# Must match exps/yolox_plant_nano.py CLASS_NAMES order exactly.
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

DISEASE_INFO = {
    "Apple Scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "Apple Rust": "Rust fungus that creates orange/yellow spots on apple leaves.",
    "Corn Leaf Blight": "Fungal disease causing long, tan lesions on corn leaves.",
    "Corn Gray Spot": "Causes rectangular gray/brown lesions running parallel to leaf veins.",
    "Potato Early Blight": "Dark brown spots with concentric rings, usually on older leaves.",
    "Potato Late Blight": "Water-soaked lesions that turn brown/black -- highly destructive.",
    "Tomato Bacterial Spot": "Small, water-soaked spots that turn brown with yellow halos.",
    "Tomato Leaf Mold": "Pale green/yellow patches on upper leaf surface, mold on underside.",
    "Tomato Mosaic Virus": "Mottled light/dark green pattern on leaves, stunted growth.",
    "Healthy Leaf": "No disease detected. The plant appears healthy!",
}

STRIDES = (8, 16, 32)


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r


def make_grids(input_size, strides=STRIDES):
    """Anchor-free grid centers + matching stride per cell, concatenated across
    the 3 detection scales -- this is what YOLOX's head.decode_outputs() does,
    reimplemented here since we exported with decode_in_inference=False."""
    grids, expanded_strides = [], []
    for stride in strides:
        gh, gw = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride))
    return np.concatenate(grids, axis=0), np.concatenate(expanded_strides, axis=0)


def decode_and_nms(raw_output, input_size, conf_thresh=0.4, nms_thresh=0.45, num_classes=10):
    """raw_output: (N, 5 + num_classes) = [x, y, w, h, obj, cls...] in grid units."""
    grids, strides = make_grids(input_size)
    out = raw_output.copy()
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides

    boxes_xywh = out[:, :4]
    obj_conf = out[:, 4:5]
    cls_conf = out[:, 5:]
    scores = obj_conf * cls_conf  # (N, num_classes)

    cls_ids = np.argmax(scores, axis=1)
    cls_scores = scores[np.arange(scores.shape[0]), cls_ids]

    keep = cls_scores > conf_thresh
    if not keep.any():
        return []

    boxes_xywh = boxes_xywh[keep]
    cls_ids = cls_ids[keep]
    cls_scores = cls_scores[keep]

    # xywh (center) -> xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # cv2.dnn.NMSBoxes wants rects as (x, y, w, h), not (x1, y1, x2, y2) --
    # passing corners in directly (as this did before) silently corrupts the
    # IoU computation for any box not touching the origin.
    boxes_xywh_for_nms = np.stack(
        [boxes_xyxy[:, 0], boxes_xyxy[:, 1],
         boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]],
        axis=1,
    )

    detections = []
    for c in np.unique(cls_ids):
        mask = cls_ids == c
        idxs = cv2.dnn.NMSBoxes(
            boxes_xywh_for_nms[mask].tolist(),
            cls_scores[mask].tolist(),
            conf_thresh,
            nms_thresh,
        )
        for i in np.array(idxs).flatten() if len(idxs) else []:
            box = boxes_xyxy[mask][i]
            score = cls_scores[mask][i]
            detections.append((box, float(score), int(c)))
    return detections


def build_net(param_path, bin_path, num_threads=4):
    net = ncnn.Net()
    net.opt.num_threads = num_threads
    net.opt.use_vulkan_compute = False  # Mali G56/G57 vulkan path exists but is
    # unvalidated on mainline Armbian for this board -- CPU NEON only for now.
    net.load_param(param_path)
    net.load_model(bin_path)
    return net


def run_inference(net, frame, input_size, conf, nms, num_classes):
    canvas, ratio = letterbox(frame, input_size)

    # YOLOX's default (non-legacy) preproc does NOT normalize: raw 0-255
    # float32, channel order left as whatever cv2 read (BGR), only HWC->CHW.
    # No /255, no mean/std, no BGR->RGB -- match that exactly or the network
    # sees an input distribution it was never trained on. (This is the
    # opposite of YOLOv5/v8-style preprocessing -- easy to get wrong by habit.)
    mat_in = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR, canvas.shape[1], canvas.shape[0])

    ex = net.create_extractor()
    ex.input("images", mat_in)
    ret, mat_out = ex.extract("output")
    raw = np.array(mat_out).reshape(-1, 5 + num_classes)

    detections = decode_and_nms(raw, input_size, conf, nms, num_classes)
    # undo letterbox scaling to get back to original frame coordinates
    scaled = [((box / ratio), score, cls) for box, score, cls in detections]
    return scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--source", default="/dev/video0")
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--headless", action="store_true",
                     help="skip cv2.imshow -- use over SSH with no display attached")
    args = ap.parse_args()

    net = build_net(args.param, args.bin, args.threads)
    input_size = (args.input_size, args.input_size)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if isinstance(source, str) and source.startswith("/dev/video") else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps_smooth = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            detections = run_inference(net, frame, input_size, args.conf, args.nms, len(CLASS_NAMES))
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * fps

            for box, score, cls_id in detections:
                x1, y1, x2, y2 = box.astype(int)
                name = CLASS_NAMES[cls_id]
                color = (0, 200, 0) if name == "Healthy Leaf" else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {score*100:.0f}%", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, f"{fps_smooth:.1f} FPS", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if not args.headless:
                cv2.imshow("plant disease detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if detections:
                    summary = ", ".join(f"{CLASS_NAMES[c]} {s*100:.0f}%" for _, s, c in detections)
                    print(f"[{fps_smooth:.1f} FPS] {summary}")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
