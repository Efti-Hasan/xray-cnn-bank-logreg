"""Diagnostics behind sections 1 and 5 of the README.

    python probe.py --out outputs

Two questions the headline metrics cannot answer:

* validation scores 0.9992 and the test set scores 0.9646. Which class moved?
* the activation maps put mass on the corner holding the laterality marker. Is
  the network reading that marker, or is the map just too coarse to tell?

The second one is answered by doing something rather than by looking: blank a
border band with the image's own median grey and see whether the false positives
go away. They do not, which is the point.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import data as ds
from evaluate import load_model, probs_for
from metrics import summary

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Problem-1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--out", default="outputs")
    p.add_argument("--band", type=float, default=0.12, help="occluded border width")
    return p.parse_args(argv)


def occlude(x_u8, frac, where):
    """Replace a border band with each image's own median grey level.

    Median rather than zero: a black band is a value the network has never seen
    in a real film, so it would confound "information removed" with "impossible
    input".
    """
    out = x_u8.copy()
    n = max(1, int(round(frac * x_u8.shape[1])))
    med = np.median(x_u8.reshape(len(x_u8), -1), axis=1).astype(np.uint8)[:, None, None]
    if where in ("all", "top"):
        out[:, :n, :] = med
    if where in ("all", "bottom"):
        out[:, -n:, :] = med
    if where in ("all", "sides"):
        out[:, :, :n] = med
        out[:, :, -n:] = med
    return out


def score_spread(name, y, p):
    for label, cls in enumerate(ds.CLASSES):
        q = p[y == label]
        print(f"  {name:11s} {cls:9s} median {np.median(q):.3f}  "
              f"p10 {np.quantile(q, 0.10):.3f}  p90 {np.quantile(q, 0.90):.3f}")


def main(argv=None):
    args = parse_args(argv)
    net, cfg = load_model(args.out)
    cache = os.path.join(args.out, "cache")

    x_tr, y_tr, names = ds.load_split(args.data_root, "train", cfg["img"], cache)
    _, val_idx = ds.grouped_split(names, y_tr, cfg["val_frac"], cfg["seed"])
    x_te, y_te, _ = ds.load_split(args.data_root, "test", cfg["img"], cache)

    print("where the validation-to-test gap lives (predicted probability of pneumonia)")
    score_spread("validation", y_tr[val_idx], probs_for(net, x_tr[val_idx], cfg))
    score_spread("test", y_te, probs_for(net, x_te, cfg))
    print("  raw pixels: validation mean %.1f std %.1f, test mean %.1f std %.1f"
          % (x_tr[val_idx].mean(), x_tr[val_idx].std(), x_te.mean(), x_te.std()))

    print(f"\nborder occlusion ({args.band:.0%} band, filled with each image's median grey)")
    print("  input                 AUC  sens    spec    fp  fn")
    variants = [("original", x_te)] + [
        (f"{w} masked", occlude(x_te, args.band, w)) for w in ("top", "sides", "all")]
    for name, x in variants:
        m = summary(y_te, probs_for(net, x, cfg), 0.5)
        c = m["confusion"]
        print(f"  {name:18s} {m['roc_auc']:.4f}  {m['recall_sensitivity']:.4f}  "
              f"{m['specificity']:.4f}  {c['fp']:>3}  {c['fn']:>2}")
    print("\nMasking the marker corner raises the false-positive count, so the marker\n"
          "is not the shortcut the maps hint at. AUC falls under every mask, so the\n"
          "border does carry signal. At 6x6 feature resolution the two cannot be\n"
          "told apart. See README section 5.")


if __name__ == "__main__":
    main()
