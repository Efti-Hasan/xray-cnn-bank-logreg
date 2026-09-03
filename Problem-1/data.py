"""Finding, decoding, caching and splitting the chest X-rays.

Two details matter more than they look:

* the shipped `val/` folder holds only 16 images, far too few to pick a model on,
  so a real validation set is carved out of `train/`.
* that carve-out is grouped by patient. Several images can come from the same
  child, and a random split would put one child on both sides of the line and
  flatter the score.

Augmentation is not here. Keras does it inside the model, in `train.py`.

Decoding goes through `tf.io` and `tf.image`, so the only things this folder needs
are tensorflow, numpy and matplotlib.
"""
from __future__ import annotations

import os
import re

import numpy as np
import tensorflow as tf

CLASSES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "test", "val")
_PNEU = re.compile(r"^(person\d+)_", re.IGNORECASE)
_NORM = re.compile(r"^(NORMAL2-IM-\d+|IM-\d+)", re.IGNORECASE)


def patient_key(fname: str) -> str:
    """Best-effort patient identifier taken from the filename."""
    m = _PNEU.match(fname) or _NORM.match(fname)
    return (m.group(1) if m else fname).lower()


def list_split(root: str, split: str, cls: str) -> list[str]:
    """Image filenames for one split/class, sorted for reproducibility."""
    folder = os.path.join(root, split, cls)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png")))


def _read_gray(path: str, img: int) -> np.ndarray:
    """Decode to one channel and resize, averaging areas when shrinking.

    These films arrive around 1,300 pixels wide, so almost every read is a
    shrink. Area averaging keeps the texture that plain sampling would drop.
    """
    arr = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
    method = "area" if max(int(arr.shape[0]), int(arr.shape[1])) > img else "bilinear"
    out = tf.image.resize(arr, (img, img), method=method).numpy()[..., 0]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def load_split(root: str, split: str, img: int, cache_dir: str | None = None):
    """Return (images uint8 [N,img,img], labels int8, filenames) for one split."""
    cache = os.path.join(cache_dir, f"{split}_{img}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["x"], z["y"], list(z["names"])
    xs, ys, names = [], [], []
    for label, cls in enumerate(CLASSES):
        for fname in list_split(root, split, cls):
            xs.append(_read_gray(os.path.join(root, split, cls, fname), img))
            ys.append(label)
            names.append(f"{cls}/{fname}")
    x = np.stack(xs).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int8)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, x=x, y=y, names=np.asarray(names))
    return x, y, names


def grouped_split(names, y, frac=0.15, seed=0):
    """Stratified-by-class, grouped-by-patient train/validation index split."""
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        groups: dict[str, list[int]] = {}
        for i in rows:
            groups.setdefault(patient_key(names[i].split("/", 1)[1]), []).append(int(i))
        keys = sorted(groups)
        rng.shuffle(keys)
        target = int(round(frac * len(rows)))
        taken = 0
        for k in keys:
            if taken >= target:
                break
            val_idx += groups[k]
            taken += len(groups[k])
    val = np.asarray(sorted(val_idx), dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    mask[val] = False
    return np.flatnonzero(mask), val
