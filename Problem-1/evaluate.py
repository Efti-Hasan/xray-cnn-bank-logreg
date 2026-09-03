"""Score the trained model on the untouched `test/` folder and draw the figures.

    python evaluate.py --data-root ../data/Problem-1 --out outputs

Training is in `train.py`. This file only reads `model.keras` back, so the test
set is touched once, at the end, by code that cannot change the weights.

The operating threshold is chosen on the held-out validation split and then used
unchanged on the test set. Picking it on the test set would be the easiest way to
publish a number nobody can reproduce.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import data as ds  # noqa: E402
from metrics import pick_threshold, roc_curve, summary  # noqa: E402

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Problem-1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--out", default="outputs")
    p.add_argument("--min-sensitivity", type=float, default=0.98)
    return p.parse_args(argv)


def load_model(out_dir):
    """The saved Keras model plus the settings train.py ran with."""
    with open(os.path.join(out_dir, "config.json")) as fh:
        cfg = json.load(fh)
    return tf.keras.models.load_model(os.path.join(out_dir, "model.keras")), cfg


def probs_for(model, x_u8, cfg=None, batch=128):
    """Pneumonia probability per image. Rescaling and augmentation sit inside the
    model, and augmentation switches itself off outside training."""
    x = np.asarray(x_u8)[..., None].astype("float32")
    return model.predict(x, batch_size=batch, verbose=0).reshape(-1)


def class_activation_map(model, x_u8):
    """One dot product, thanks to the global-average-pool head.

    The logit is the channel weights dotted with the pooled feature map. Move the
    pooling to the outside and the same weights, applied per pixel, say which part
    of the film pushed the score up.
    """
    feats_model = tf.keras.Model(model.inputs, model.get_layer("features").output)
    feats = feats_model.predict(np.asarray(x_u8)[..., None].astype("float32"), verbose=0)
    w = model.get_layer("head").get_weights()[0].reshape(-1)
    return feats @ w


def plot_confusion(cm, path, title):
    order = [["tn", "fp"], ["fn", "tp"]]
    grid = np.array([[cm[k] for k in row] for row in order], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["pred NORMAL", "pred PNEUMONIA"])
    ax.set_yticks([0, 1], ["true NORMAL", "true PNEUMONIA"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(grid[i, j]), ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black", fontsize=14)
    ax.set_title(title)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def plot_roc(y, probs, auc, path):
    fpr, tpr = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(fpr, tpr, label=f"CNN (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate"), ax.set_ylabel("true positive rate")
    ax.set_title("Test ROC"), ax.legend(loc="lower right"), ax.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def plot_cams(model, x_u8, y, probs, cfg, path, n=8):
    """Overlays: two confident hits first, then the worst mistakes."""
    pred = (probs >= 0.5).astype(int)
    wrong = np.flatnonzero(pred != y)
    right = np.flatnonzero(pred == y)
    order = list(right[np.argsort(-np.abs(probs[right] - 0.5))][:2])
    order += list(wrong[np.argsort(-np.abs(probs[wrong] - 0.5))][:n - 2])
    order += list(right[np.argsort(-np.abs(probs[right] - 0.5))][2:n - len(order) + 2])
    order = order[:n]
    cams = class_activation_map(model, x_u8[order])
    cols = min(4, len(order))
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax, (k, i) in zip(axes.ravel(), enumerate(order)):
        cam = cams[k]
        cam = np.clip((cam - cam.min()) / max(1e-8, float(np.ptp(cam))), 0, 1)
        # 6x6 map back up to the film, smoothly, so the overlay is readable.
        cam = tf.image.resize(cam[..., None], (cfg["img"], cfg["img"]),
                              method="bicubic").numpy()[..., 0]
        ax.imshow(x_u8[i], cmap="gray")
        ax.imshow(cam, cmap="inferno", alpha=0.42)
        mark = "OK" if pred[i] == y[i] else "MISS"
        ax.set_title(f"{ds.CLASSES[y[i]][:4]} p={probs[i]:.2f} {mark}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("Where the network looks (class activation maps)", fontsize=11)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def show(title, m):
    c = m["confusion"]
    print(f"\n{title}  (threshold {m['threshold']:.3f})")
    print(f"  accuracy {m['accuracy']:.4f}   balanced {m['balanced_accuracy']:.4f}   "
          f"ROC AUC {m['roc_auc']:.4f}")
    print(f"  sensitivity {m['recall_sensitivity']:.4f}   specificity {m['specificity']:.4f}   "
          f"precision {m['precision']:.4f}   F1 {m['f1']:.4f}")
    print(f"  tn {c['tn']}  fp {c['fp']}  fn {c['fn']}  tp {c['tp']}")


def main(argv=None):
    args = parse_args(argv)
    root, out = args.data_root, args.out
    model, cfg = load_model(out)
    cache = os.path.join(out, "cache")

    x_tr, y_tr, names = ds.load_split(root, "train", cfg["img"], cache)
    _, val_idx = ds.grouped_split(names, y_tr, cfg["val_frac"], cfg["seed"])
    p_val = probs_for(model, x_tr[val_idx])
    thr = pick_threshold(y_tr[val_idx], p_val, args.min_sensitivity)
    val_m = summary(y_tr[val_idx], p_val, thr)

    x_te, y_te, _ = ds.load_split(root, "test", cfg["img"], cache)
    p_te = probs_for(model, x_te)
    test_half = summary(y_te, p_te, 0.5)
    test_tuned = summary(y_te, p_te, thr)

    x_v, y_v, _ = ds.load_split(root, "val", cfg["img"], cache)
    shipped = summary(y_v, probs_for(model, x_v), thr)

    print(f"model: {cfg['n_params']:,} parameters, {cfg['img']}x{cfg['img']} input, "
          f"{cfg['epochs_run']} epochs, best at {cfg['best_epoch']}")
    print(f"operating threshold {thr:.3f} chosen on the validation split "
          f"(sensitivity floor {args.min_sensitivity})")
    show("held-out validation (patient-grouped)", val_m)
    show(f"TEST at 0.5 ({len(y_te)} images)", test_half)
    show(f"TEST at tuned threshold ({len(y_te)} images)", test_tuned)
    show("shipped val/ folder (16 images)", shipped)

    plot_confusion(test_tuned["confusion"], os.path.join(out, "fig_confusion.png"),
                   f"Test set, threshold {thr:.2f}")
    plot_roc(y_te, p_te, test_half["roc_auc"], os.path.join(out, "fig_roc.png"))
    plot_cams(model, x_te, y_te, p_te, cfg, os.path.join(out, "fig_cam.png"))

    report = {"threshold": thr, "validation": val_m, "test_at_0.5": test_half,
              "test_at_threshold": test_tuned, "shipped_val_folder": shipped,
              "n_params": cfg["n_params"], "config": cfg}
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}/metrics.json and 3 figures")


if __name__ == "__main__":
    main()
