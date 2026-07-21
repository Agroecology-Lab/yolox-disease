"""
Export a trained YOLOX checkpoint to ONNX for the NCNN conversion step.

Equivalent of the original pipeline's:
    yolo export model=best.pt format=onnx

Usage (run from a YOLOX checkout with this repo's exps/ alongside it):
    python tools/export_onnx.py \
        -f exps/yolox_plant_nano.py \
        -c YOLOX_outputs/yolox_plant_nano/best_ckpt.pth \
        --output-name plant_disease_nano.onnx

The decode/NMS is deliberately left OUT of the exported graph (--decode-in-inference
off) -- decoding YOLOX's raw (reg, obj, cls) head outputs and running NMS is done
in deploy/realtime_infer.py in plain numpy, which is portable to NCNN (which has
no native YOLOX decode op) and keeps the graph itself minimal for the SBC's CPU.
"""
import argparse

import torch
from yolox.exp import get_exp
from yolox.utils import fuse_model


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx export")
    parser.add_argument("--output-name", type=str, default="plant_disease_nano.onnx")
    parser.add_argument("-f", "--exp_file", required=True, type=str)
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="checkpoint path")
    parser.add_argument("--opset", type=int, default=11, help="11 is what onnx2ncnn expects")
    parser.add_argument("--no-onnxsim", action="store_true", help="skip onnx-simplifier pass")
    return parser


def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)

    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    # fuse conv+bn for a leaner exported graph, matches yolox's own export tool.
    # (Previous version of this line -- `model.head.decode_in_inference and
    # model or model` -- was a no-op tautology that always evaluated to
    # `model`; it never called fuse and the "matches yolox's own export tool"
    # comment was aspirational, not true.)
    model = fuse_model(model)
    model.head.decode_in_inference = False  # export raw head outputs, decode on host

    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=["images"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported {args.output_name}")

    if not args.no_onnxsim:
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(args.output_name)
            model_simp, check = simplify(onnx_model)
            assert check, "onnx-simplifier check failed"
            onnx.save(model_simp, args.output_name)
            print("Simplified with onnx-simplifier")
        except ImportError:
            print("onnx-simplifier not installed (pip install onnx-simplifier) -- "
                  "skipping simplification, onnx2ncnn may still work without it")


if __name__ == "__main__":
    main()
