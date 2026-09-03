"""Train the pneumonia CNN with Keras, then draw the history with matplotlib.

    python train.py --data-root ../data/Problem-1 --epochs 30

The shape follows the TensorFlow image-classification tutorial: stack Conv2D and
MaxPooling2D, put a small head on top, compile, fit, plot. Keras owns the
backward pass, so this file is about the choices, not the calculus.

Three of those choices are not the default, and each one is here for a reason.

* The validation split comes from `data.py`, so it is grouped by patient. Several
  films can come from the same child, and a plain random split would put one
  child on both sides of the line and flatter the score.
* The loss carries class weights, because pneumonia outnumbers normal 2.9 to 1.
  Without them the model learns to answer "pneumonia" and stop thinking.
* Early stopping watches validation ROC AUC, not accuracy. Accuracy depends on
  the 0.5 cut-off, and the cut-off is chosen later in `evaluate.py`.

Needs `tensorflow`. TensorFlow has no wheel for Python 3.14 yet, so use 3.12:

    python3.12 -m venv .venv
    .venv/bin/pip install tensorflow matplotlib numpy
    .venv/bin/python train.py
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
from tensorflow.keras import layers  # noqa: E402

import data as ds  # noqa: E402

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Problem-1")
WIDTHS = (16, 32, 64, 128)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--out", default="outputs")
    p.add_argument("--img", type=int, default=96)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--aug-strength", type=float, default=1.0,
                   help="scales rotation, shift and zoom. 0 turns augmentation off")
    return p.parse_args(argv)


def build_model(img, dropout=0.4, aug=1.0):
    """Four conv stages, then average the map instead of flattening it.

    Global average pooling keeps the head at one weight per channel, which is
    small enough to survive 4,400 training images. It also makes the class
    activation maps in `evaluate.py` a single dot product.

    No horizontal flip in the augmentation. A mirrored chest film is impossible,
    the heart sits on the left, so flipping would teach an invariance that the
    real films never show. Rotation, shift and zoom are all things that do vary
    between two films of the same chest.
    """
    model = tf.keras.Sequential(name="pneumonia_cnn")
    model.add(tf.keras.Input(shape=(img, img, 1)))
    model.add(layers.Rescaling(1.0 / 255))
    if aug > 0:
        model.add(layers.RandomRotation(0.02 * aug, fill_mode="nearest"))
        model.add(layers.RandomTranslation(0.06 * aug, 0.06 * aug, fill_mode="nearest"))
        model.add(layers.RandomZoom(0.1 * aug, fill_mode="nearest"))
    for i, width in enumerate(WIDTHS):
        model.add(layers.Conv2D(width, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        # The last pooling layer is named so evaluate.py can reach its feature map.
        model.add(layers.MaxPooling2D(2, name="features" if i == len(WIDTHS) - 1 else None))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout))
    # Sigmoid here rather than a raw logit, so every Keras metric below reads the
    # output as a probability without extra wiring.
    model.add(layers.Dense(1, activation="sigmoid", name="head"))
    return model


def plot_history(hist, path):
    """Two panels: what the loss did, and what the validation metrics did."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["loss"], label="train")
    ax[0].plot(hist["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("weighted BCE"), ax[0].legend()
    ax[0].set_title("Loss")
    ax[1].plot(hist["val_auc"], label="ROC AUC")
    ax[1].plot(hist["val_accuracy"], label="accuracy")
    ax[1].plot(hist["val_recall"], label="sensitivity")
    ax[1].set_xlabel("epoch"), ax[1].set_ylim(0, 1.02), ax[1].legend(loc="lower right")
    ax[1].set_title("Validation metrics")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    cache = os.path.join(args.out, "cache")
    x_tr, y_tr, names = ds.load_split(args.data_root, "train", args.img, cache)
    tr_idx, val_idx = ds.grouped_split(names, y_tr, args.val_frac, args.seed)

    # Keras wants floats with a channel axis. The /255 happens inside the model.
    xt = x_tr[tr_idx][..., None].astype("float32")
    yt = y_tr[tr_idx].astype("float32")
    xv = x_tr[val_idx][..., None].astype("float32")
    yv = y_tr[val_idx].astype("float32")
    print(f"train {len(xt)} images, validation {len(xv)}, "
          f"{len(set(ds.patient_key(names[i].split('/', 1)[1]) for i in val_idx))} "
          f"patients held out")

    counts = np.bincount(yt.astype(int), minlength=2)
    weights = {i: float(len(yt) / (2.0 * c)) for i, c in enumerate(counts)}
    print(f"class balance: normal {counts[0]}, pneumonia {counts[1]}. "
          f"weights {weights[0]:.3f} and {weights[1]:.3f}")

    model = build_model(args.img, args.dropout, args.aug_strength)
    model.summary()

    # Warm up for one epoch, then let the rate follow a cosine down to almost
    # nothing. A constant rate leaves the weights still moving when early
    # stopping fires, and the last few epochs at a small rate are what settles
    # the decision boundary. Weight decay is decoupled from the gradient here,
    # which is what AdamW does and plain Adam does not.
    steps = int(np.ceil(len(xt) / args.batch))
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.lr / 20, decay_steps=max(1, args.epochs * steps - steps),
        warmup_target=args.lr, warmup_steps=steps, alpha=0.02)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(schedule, weight_decay=args.weight_decay),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=args.patience,
        restore_best_weights=True, verbose=1)
    history = model.fit(xt, yt, epochs=args.epochs, batch_size=args.batch,
                        validation_data=(xv, yv), class_weight=weights,
                        callbacks=[stop], verbose=2)

    hist = history.history
    plot_history(hist, os.path.join(args.out, "fig_training.png"))
    best = int(np.argmax(hist["val_auc"]))
    model.save(os.path.join(args.out, "model.keras"))

    cfg = {"img": args.img, "widths": list(WIDTHS), "dropout": args.dropout,
           "seed": args.seed, "val_frac": args.val_frac, "batch": args.batch,
           "lr": args.lr, "weight_decay": args.weight_decay,
           "aug_strength": args.aug_strength, "patience": args.patience,
           "epochs_run": len(hist["loss"]), "best_epoch": best + 1,
           "best_val_auc": float(hist["val_auc"][best]),
           "n_params": int(model.count_params())}
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.items()}, fh, indent=2)

    print(f"\nbest epoch {best + 1} of {len(hist['loss'])}, "
          f"validation ROC AUC {cfg['best_val_auc']:.4f}")
    print(f"wrote {args.out}/model.keras, config.json, history.json, fig_training.png")
    print("now run: python evaluate.py --out", args.out)


if __name__ == "__main__":
    main()
