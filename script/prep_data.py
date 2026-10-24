"""Lesson A3: build calibration + validation sets from Imagenette (a 10-class ImageNet subset).

Preprocessing matches ResNet18's ImageNet eval: resize short side to 256, center-crop 224.
Images are saved as raw uint8 RGB (0-255); normalization happens on the Hailo chip.

pip install pyarrow

Usage: python prep_data.py --parquet hf_data/imagenette/data/train-00000-of-00001.parquet --out hf_out
Produces: calib_set.npy (N,224,224,3 uint8), val_set.npz (x uint8, y = ImageNet class index)


What the script does (open prep_data.py to follow along):
1. Resizes each image so its shorter side is 256, then center-crops it to 224×224. This is the standard ImageNet preprocessing ResNet18 was trained with.
2. Keeps the pixels as raw 0–255 numbers and doesn't normalize them. The Hailo chip will do the normalization itself, in the layer that step A6 adds.
3. Saves two different sets of images:
   - calib_set.npy: 512 images for calibration, with no labels
   - val_set.npz: 500 other images, with labels, for measuring accuracy
4. Converts the labels from Imagenette's 0–9 to ImageNet's 0–999 numbering. For example, "golf ball" is class 574 in ImageNet. The model outputs 1000 scores, so its predictions have to be compared against ImageNet numbers.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Imagenette label order (WordNet IDs) -> index among ImageNet-1k's 1000 classes.
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def preprocess(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="hf_out")
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--val", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = np.array(table.column("label").to_pylist())
    print(f"{len(images)} images in parquet")

    # Random, disjoint calibration and validation subsets.
    idx = np.random.default_rng(0).permutation(len(images))
    calib_idx, val_idx = idx[:args.calib], idx[args.calib:args.calib + args.val]

    calib = np.stack([preprocess(images[i]["bytes"]) for i in calib_idx])
    np.save(out / "calib_set.npy", calib)
    val_x = np.stack([preprocess(images[i]["bytes"]) for i in val_idx])
    val_y = np.array([IMAGENETTE_TO_IMAGENET[labels[i]] for i in val_idx])
    np.savez(out / "val_set.npz", x=val_x, y=val_y)
    print(f"calib_set {calib.shape}, val_set {val_x.shape} -> {out}/")


if __name__ == "__main__":
    main()
