"""Scoring helpers written against numpy directly, so every number in the README
traces back to a formula instead of a library default.

Accuracy is not the headline here, on purpose. Predicting "no" for everyone scores
88.3% on this dataset, so it says nothing. ROC AUC, PR AUC and the lift curve are
what a campaign manager can act on.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    y, p = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return tn, fp, fn, tp


def _ranked(y_true, scores):
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-s, kind="mergesort")
    return y[order], s[order]


def roc_auc(y_true, scores):
    """Rank-based AUC with average ranks for ties (Mann-Whitney U)."""
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s_sorted[i] != s_sorted[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _tie_cuts(y_true, scores):
    """Cumulative tp/fp counts at each *distinct* score, highest first.

    Collapsing tie groups matters: with probabilities rounded for reporting,
    hundreds of customers can share a score, and splitting such a group would
    credit the model with an ordering it never produced.
    """
    y, s = _ranked(y_true, scores)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    return tp[last], fp[last], last + 1, int((y == 1).sum()), int((y == 0).sum())


def roc_curve(y_true, scores):
    tp, fp, _, n_pos, n_neg = _tie_cuts(y_true, scores)
    return np.r_[0.0, fp / max(1, n_neg)], np.r_[0.0, tp / max(1, n_pos)]


def pr_curve(y_true, scores):
    """Precision and recall at every distinct cut point, plus average precision."""
    tp, _, called, n_pos, _ = _tie_cuts(y_true, scores)
    precision, recall = tp / called, tp / max(1, n_pos)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return recall, precision, ap



def lift_table(y_true, scores, deciles=10):
    """Response rate per score decile, the table a call centre actually uses."""
    y, _ = _ranked(y_true, scores)
    base = y.mean()
    rows = []
    edges = np.linspace(0, len(y), deciles + 1).round().astype(int)
    for d, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        chunk = y[a:b]
        rate = float(chunk.mean()) if len(chunk) else 0.0
        rows.append({"decile": d, "customers": int(b - a), "subscribers": int(chunk.sum()),
                     "response_rate": rate, "lift": float(rate / base) if base else 0.0,
                     "cumulative_recall": float(y[:b].sum() / max(1, y.sum()))})
    return rows


def summary(y_true, probs, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y_true, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    _, _, ap = pr_curve(y_true, probs)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (rec + spec),
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": roc_auc(y_true, probs),
        "pr_auc": ap,
        "brier": float(np.mean((probs - np.asarray(y_true).reshape(-1)) ** 2)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "flag_rate": float(pred.mean()),
    }


def best_threshold(y_true, probs, objective="f1"):
    """Sweep the score grid for the cut that maximises F1 (or Youden's J)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    grid = np.unique(np.round(probs, 3))
    best, best_val = 0.5, -np.inf
    for t in grid:
        m = summary(y_true, probs, t)
        val = m["f1"] if objective == "f1" else m["recall"] + m["specificity"] - 1.0
        if val > best_val:
            best, best_val = float(t), val
    return best


def calibration(y_true, probs, bins=10):
    """Mean predicted probability vs observed rate, on equal-count bins."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    order = np.argsort(p, kind="mergesort")
    y, p = y[order], p[order]
    edges = np.linspace(0, len(p), bins + 1).round().astype(int)
    pred, obs, count = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            pred.append(float(p[a:b].mean()))
            obs.append(float(y[a:b].mean()))
            count.append(int(b - a))
    return np.array(pred), np.array(obs), np.array(count)

