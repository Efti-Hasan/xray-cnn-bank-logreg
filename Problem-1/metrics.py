"""Reporting metrics, written out in plain numpy rather than pulled from Keras.

Keras tracks accuracy and AUC while it trains, which is what early stopping needs.
The final report needs more than that: a confusion matrix at a threshold that was
chosen on validation, and sensitivity and specificity separately. Doing it here
keeps the reported numbers independent of the library that produced the model.

Sensitivity (recall on the pneumonia class) is the headline number. A missed
pneumonia in a one-to-five-year-old costs far more than a false alarm that gets a
second read.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary labels in {0, 1}."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def roc_auc(y_true, scores):
    """AUC via the Mann-Whitney statistic, with mid-ranks for ties."""
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summary(y_true, probs, threshold=0.5):
    pred = (np.asarray(probs).reshape(-1) >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y_true, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal = 0.5 * (rec + spec)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": bal,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc(y_true, probs),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def roc_curve(y_true, scores):
    """(fpr, tpr) sampled at every distinct score, highest score first."""
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos, n_neg = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    return np.r_[0, fp / n_neg], np.r_[0, tp / n_pos]


def pick_threshold(y_true, probs, min_sensitivity=0.98):
    """Lowest-cost operating point that still catches `min_sensitivity` of cases.

    Falls back to the threshold with the best Youden's J if the constraint is
    unreachable, so this always returns something usable.
    """
    probs = np.asarray(probs).reshape(-1)
    grid = np.unique(np.round(probs, 4))
    best, best_key = 0.5, None
    for t in grid:
        m = summary(y_true, probs, t)
        if m["recall_sensitivity"] >= min_sensitivity:
            key = (1, m["specificity"])
        else:
            key = (0, m["recall_sensitivity"] + m["specificity"])
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


