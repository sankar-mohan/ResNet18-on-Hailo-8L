"""Lesson A2: load Hugging Face ResNet18 weights (timm/resnet18.tv_in1k) and export ONNX.

timm's ResNet18 uses the same layer names as torchvision's, so the safetensors file loads
straight into torchvision.models.resnet18 -- no timm install needed.

pip install numpy onnxruntime torch torchvision safetensors onnx

Usage: python export_onnx.py --weights hf_models/resnet18.tv_in1k/model.safetensors --out hf_out
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from torchvision.models import resnet18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="hf_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = resnet18()
    missing, unexpected = model.load_state_dict(load_file(args.weights), strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model.eval()
    print("loaded weights:", sum(p.numel() for p in model.parameters()) / 1e6, "M params")

    onnx_path = out / "resnet18.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    # Normalization is NOT inside the ONNX; Hailo adds it on-chip during optimization.
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamo=False)

    # Sanity check: ONNX Runtime must give the same answer as PyTorch.
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = ort.InferenceSession(str(onnx_path)).run(None, {"input": dummy.numpy()})[0]
    print(f"wrote {onnx_path}  max |pytorch - onnx| = {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
