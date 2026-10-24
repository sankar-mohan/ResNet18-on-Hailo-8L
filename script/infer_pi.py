"""Lesson A9: run resnet18.hef on the Raspberry Pi 5 AI HAT (Hailo-8L).

Two modes:
  python3 infer_pi.py --hef resnet18.hef --image dog.jpg [--labels imagenet_labels.txt]
      -> top-5 ImageNet classes for one photo
  python3 infer_pi.py --hef resnet18.hef --val val_set.npz
      -> top-1 accuracy on the same 500 images used in the DFC simulator (compare with int8 0.794)

Preprocessing matches prep_data.py: short side -> 256, center-crop 224, uint8 RGB, no normalization
(the HEF normalizes on-chip).
"""
import argparse
import time

import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                            InputVStreamParams, OutputVStreamParams, VDevice)
from PIL import Image


def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    # np.array (not np.asarray): PIL's buffer is read-only, and HailoRT requires a writeable array.
    return np.array(img.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="resnet18.hef")
    ap.add_argument("--image")
    ap.add_argument("--val")
    ap.add_argument("--labels")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    if not (args.image or args.val):
        ap.error("give --image or --val")

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    print(f"input {in_info.name} {in_info.shape}  output {out_info.name} {out_info.shape}")

    with VDevice() as device:
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, params)[0]
        # Send uint8 pixels; get float32 logits back (HailoRT de-quantizes using the HEF's scale/zero-point).
        in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        with InferVStreams(network_group, in_params, out_params) as pipeline, \
                network_group.activate(network_group.create_params()):

            def infer(x):
                # HailoRT needs a C-contiguous, writeable buffer; np.array always makes such a copy.
                x = np.array(x, dtype=np.uint8, order="C")
                return pipeline.infer({in_info.name: x})[out_info.name].reshape(len(x), -1)

            if args.image:
                logits = infer(preprocess(args.image)[None])[0]
                names = open(args.labels).read().splitlines() if args.labels else None
                probs = softmax(logits)
                for i in probs.argsort()[::-1][:5]:
                    print(f"{probs[i]:6.3f}  {i:4d}  {names[i] if names else ''}")

            if args.val:
                data = np.load(args.val)
                x, y = data["x"], data["y"]
                preds, t0 = [], time.time()
                for i in range(0, len(x), args.batch):
                    preds.append(infer(x[i:i + args.batch]).argmax(1))
                dt = time.time() - t0
                acc = (np.concatenate(preds) == y).mean()
                print(f"device top-1 on {len(y)} images: {acc:.3f}  ({len(y) / dt:.0f} img/s incl. Python overhead)")


if __name__ == "__main__":
    main()
