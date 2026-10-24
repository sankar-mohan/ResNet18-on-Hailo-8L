"""Lessons A5-A7: ONNX -> HAR -> quantized HAR -> HEF for Hailo-8L. Run in WSL with the DFC venv.

Usage: python compile_hailo.py --out hf_out [--step parse|optimize|compile|all]
Each step saves its own .har, so you can run them one at a time and inspect the results.

├──────────┼────────────────────────────────────────┼───────────────────────────────────────────────┼────────────┤
│ --step   │                 Reads                  │                    Writes                     │  Time on   │
│          │                                        │                                               │    CPU     │
├──────────┼────────────────────────────────────────┼───────────────────────────────────────────────┼────────────┤
│ parse    │ resnet18.onnx                          │ resnet18.har (float)                          │ seconds    │
├──────────┼────────────────────────────────────────┼───────────────────────────────────────────────┼────────────┤
│ optimize │ resnet18.har + calib_set.npy +         │ resnet18_quantized.har (int8), and prints     │ ~10–30 min │
│          │ val_set.npz                            │ accuracy                                      │            │
├──────────┼────────────────────────────────────────┼───────────────────────────────────────────────┼────────────┤
│ compile  │ resnet18_quantized.har                 │ resnet18.hef                                  │ ~5–20 min  │
├──────────┼────────────────────────────────────────┼───────────────────────────────────────────────┼────────────┤

"""
import argparse
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

NAME = "resnet18"

# ImageNet mean/std multiplied by 255, because the HEF receives raw 0-255 pixels.
MODEL_SCRIPT = """
normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
model_optimization_flavor(optimization_level={opt_level}, compression_level=0)
"""


def accuracy(runner, ctx_type, x, y):
    with runner.infer_context(ctx_type) as ctx:
        logits = runner.infer(ctx, x.astype(np.float32), batch_size=8)
    return (np.asarray(logits).reshape(len(y), -1).argmax(1) == y).mean()


def parse(out, hw_arch):
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        str(out / f"{NAME}.onnx"), NAME,
        start_node_names=["input"], end_node_names=["logits"],
        net_input_shapes={"input": [1, 3, 224, 224]},
    )
    runner.save_har(str(out / f"{NAME}.har"))
    print(f"[parse] wrote {out / f'{NAME}.har'}")


def optimize(out, opt_level):
    runner = ClientRunner(har=str(out / f"{NAME}.har"))
    calib = np.load(out / "calib_set.npy").astype(np.float32)
    val = np.load(out / "val_set.npz")
    runner.load_model_script(MODEL_SCRIPT.format(opt_level=opt_level))
    runner.optimize(calib)
    runner.save_har(str(out / f"{NAME}_quantized.har"))
    print(f"[optimize] wrote {out / f'{NAME}_quantized.har'}")
    fp = accuracy(runner, InferenceContext.SDK_FP_OPTIMIZED, val["x"], val["y"])
    q = accuracy(runner, InferenceContext.SDK_QUANTIZED, val["x"], val["y"])
    print(f"[optimize] top-1 on {len(val['y'])} Imagenette images: float {fp:.3f}  int8 {q:.3f}")


def compile_hef(out):
    runner = ClientRunner(har=str(out / f"{NAME}_quantized.har"))
    hef = runner.compile()
    (out / f"{NAME}.hef").write_bytes(hef)
    print(f"[compile] wrote {out / f'{NAME}.hef'} ({len(hef) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hf_out")
    ap.add_argument("--step", choices=["parse", "optimize", "compile", "all"], default="all")
    ap.add_argument("--hw-arch", default="hailo8l")
    ap.add_argument("--opt-level", type=int, default=1, help="2+ is slow without a GPU")
    args = ap.parse_args()
    out = Path(args.out)

    if args.step in ("parse", "all"):
        parse(out, args.hw_arch)
    if args.step in ("optimize", "all"):
        optimize(out, args.opt_level)
    if args.step in ("compile", "all"):
        compile_hef(out)


if __name__ == "__main__":
    main()
