"""Train and evaluate the term-deposit model end to end.

    python main.py --data-root ../data/Problem-2 --out outputs

What the script establishes, in order:

1. how much of the apparent skill comes from `duration`, a field that cannot
   exist before the call is made;
2. the honest ranking quality of the model without it;
3. where to put the decision threshold, given that a wasted call is cheap and a
   missed subscriber is not.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics as M  # noqa: E402
import pipeline as P  # noqa: E402
from logreg import LogisticRegression  # noqa: E402

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Problem-2")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--out", default="outputs")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args(argv)


def kfold_indices(y, folds, seed):
    """Stratified folds, so every fold carries the same subscription rate."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)
        for i, chunk in enumerate(np.array_split(rows, folds)):
            buckets[i].append(chunk)
    return [np.sort(np.concatenate(b)) for b in buckets]


def cross_validate(x, y, grid, folds, seed):
    """Pick the ridge strength on out-of-fold ROC AUC, never on the test set."""
    parts = kfold_indices(y, folds, seed)
    table = []
    for l2 in grid:
        aucs = []
        for i, hold in enumerate(parts):
            keep = np.concatenate([p for j, p in enumerate(parts) if j != i])
            m = LogisticRegression(l2=l2, class_weight="balanced").fit(x[keep], y[keep])
            aucs.append(M.roc_auc(y[hold], m.predict_proba(x[hold])))
        table.append({"l2": float(l2), "cv_auc_mean": float(np.mean(aucs)),
                      "cv_auc_std": float(np.std(aucs))})
    best = max(table, key=lambda r: r["cv_auc_mean"])
    return best["l2"], table


def fit_and_score(data, l2, threshold_objective="f1", seed=0):
    """Fit on train, choose the threshold inside train, then score the test set."""
    x_tr, y_tr = data["x_train"], data["y_train"]
    inner_tr, inner_va = P.stratified_split(y_tr, 0.2, seed + 1)
    tuner = LogisticRegression(l2=l2, class_weight="balanced").fit(x_tr[inner_tr], y_tr[inner_tr])
    thr = M.best_threshold(y_tr[inner_va], tuner.predict_proba(x_tr[inner_va]), threshold_objective)

    model = LogisticRegression(l2=l2, class_weight="balanced").fit(x_tr, y_tr)
    p_te = model.predict_proba(data["x_test"])
    return model, thr, p_te, {
        "at_0.5": M.summary(data["y_test"], p_te, 0.5),
        "at_tuned": M.summary(data["y_test"], p_te, thr),
        "train_at_tuned": M.summary(y_tr, model.predict_proba(x_tr), thr),
    }


def coefficient_table(model, columns, scaler):
    """Coefficients, odds ratios and Wald z-scores, largest effect first.

    The coefficients live in standardised units, so an odds ratio reads as
    "per one standard deviation of this feature" for the continuous columns and
    "versus the reference level" for the dummies.
    """
    se = model.standard_errors()
    rows = []
    for j, name in enumerate(columns):
        b = float(model.coef_[j])
        s = float(se[j + 1])
        rows.append({"feature": name, "coef": b, "odds_ratio": float(np.exp(b)),
                     "std_err": s, "z": b / s if s > 0 else 0.0,
                     "scaled": bool(scaler.mask_[j])})
    rows.sort(key=lambda r: -abs(r["coef"]))
    return rows


def plot_curves(y, probs_by_name, out):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name, p in probs_by_name.items():
        fpr, tpr = M.roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {M.roc_auc(y, p):.3f})")
        rec, prec, ap = M.pr_curve(y, p)
        ax[1].plot(rec, prec, label=f"{name} (AP {ap:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax[0].set_xlabel("false positive rate"), ax[0].set_ylabel("true positive rate")
    ax[0].set_title("ROC")
    base = float(np.mean(y))
    ax[1].axhline(base, ls="--", color="grey", lw=1, label=f"base rate {base:.3f}")
    ax[1].set_xlabel("recall"), ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall")
    for a in ax:
        a.legend(fontsize=8), a.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)


def plot_diagnostics(y, probs, lift, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    pred, obs, _ = M.calibration(y, probs)
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax[0].plot(pred, obs, "o-", label="model")
    ax[0].set_xlabel("mean predicted probability"), ax[0].set_ylabel("observed rate")
    ax[0].set_title("Calibration (equal-count deciles)"), ax[0].legend(fontsize=8)

    d = [r["decile"] for r in lift]
    ax[1].bar(d, [r["lift"] for r in lift], color="#3c6e9f")
    ax[1].axhline(1.0, ls="--", color="grey", lw=1)
    ax[1].set_xlabel("score decile (1 = highest)"), ax[1].set_ylabel("lift vs base rate")
    ax[1].set_title("Lift by decile")

    cum = np.r_[0, [r["cumulative_recall"] for r in lift]]
    frac = np.linspace(0, 1, len(cum))
    ax[2].plot(frac, cum, "o-", label="model")
    ax[2].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="random calling")
    ax[2].set_xlabel("fraction of customers called"), ax[2].set_ylabel("fraction of subscribers won")
    ax[2].set_title("Cumulative gain"), ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)


def plot_coefficients(rows, out, n=18):
    top = rows[:n][::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(top) + 1.4))
    vals = [r["coef"] for r in top]
    ax.barh(range(len(top)), vals,
            color=["#b5423a" if v < 0 else "#2f7d59" for v in vals])
    ax.set_yticks(range(len(top)), [r["feature"] for r in top], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log-odds coefficient (standardised units)")
    ax.set_title(f"Largest {len(top)} effects")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)


def plot_confusion(cm, out, title):
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["pred no", "pred yes"])
    ax.set_yticks([0, 1], ["true no", "true yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(grid[i, j]):,}", ha="center", va="center", fontsize=13,
                    color="white" if grid[i, j] > grid.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)


def show(title, m):
    c = m["confusion"]
    print(f"\n{title}  (threshold {m['threshold']:.3f})")
    print(f"  accuracy {m['accuracy']:.4f}   balanced {m['balanced_accuracy']:.4f}   "
          f"ROC AUC {m['roc_auc']:.4f}   PR AUC {m['pr_auc']:.4f}   Brier {m['brier']:.4f}")
    print(f"  precision {m['precision']:.4f}   recall {m['recall']:.4f}   "
          f"specificity {m['specificity']:.4f}   F1 {m['f1']:.4f}   "
          f"calls made {m['flag_rate']:.1%}")
    print(f"  tn {c['tn']:,}  fp {c['fp']:,}  fn {c['fn']:,}  tp {c['tp']:,}")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    report = {}

    honest = P.prepare(args.data_root, include_duration=False,
                       test_frac=args.test_frac, seed=args.seed)
    df = honest["frame"]
    print(f"{len(df):,} customers, {len(honest['columns'])} model columns after one-hot")
    print(f"subscription rate {df['y'].eq('yes').mean():.4f}  "
          f"(always-'no' baseline accuracy {df['y'].eq('no').mean():.4f})")
    print(f"train {honest['n_train']:,}  test {honest['n_test']:,}")

    l2, cv_table = cross_validate(honest["x_train"], honest["y_train"], grid, args.folds, args.seed)
    print(f"\nridge strength {l2} chosen on {args.folds}-fold out-of-fold AUC "
          f"({max(r['cv_auc_mean'] for r in cv_table):.4f})")

    model, thr, p_te, scores = fit_and_score(honest, l2, "f1", args.seed)
    show("TEST, no duration, default cut", scores["at_0.5"])
    show("TEST, no duration, tuned cut", scores["at_tuned"])

    leaky = P.prepare(args.data_root, include_duration=True,
                      test_frac=args.test_frac, seed=args.seed)
    l2_leaky, _ = cross_validate(leaky["x_train"], leaky["y_train"], grid, args.folds, args.seed)
    model_l, thr_l, p_te_l, scores_l = fit_and_score(leaky, l2_leaky, "f1", args.seed)
    show("TEST, WITH duration (leaks the answer)", scores_l["at_tuned"])
    gap = scores_l["at_tuned"]["roc_auc"] - scores["at_tuned"]["roc_auc"]
    print(f"\n  duration inflates test ROC AUC by {gap:+.4f} "
          f"({scores['at_tuned']['roc_auc']:.4f} -> {scores_l['at_tuned']['roc_auc']:.4f})")

    chrono = P.prepare(args.data_root, include_duration=False,
                       test_frac=args.test_frac, seed=args.seed, split="chronological")
    _, thr_c, p_te_c, scores_c = fit_and_score(chrono, l2, "f1", args.seed)
    show("TEST, chronological split (train on earlier calls)", scores_c["at_tuned"])

    y_te = honest["y_test"]
    lift = M.lift_table(y_te, p_te)
    print("\nlift by score decile (test set, no duration)")
    print("  decile  customers  subscribers  response  lift  cum.recall")
    for r in lift:
        print(f"  {r['decile']:>6}  {r['customers']:>9,}  {r['subscribers']:>11,}  "
              f"{r['response_rate']:>7.1%}  {r['lift']:>4.2f}  {r['cumulative_recall']:>9.1%}")

    coefs = coefficient_table(model, honest["columns"], honest["scaler"])
    print("\nstrongest effects (standardised log-odds)")
    for r in coefs[:12]:
        print(f"  {r['feature']:<26} {r['coef']:+.3f}  OR {r['odds_ratio']:>6.3f}  z {r['z']:+.1f}")

    plot_curves(y_te, {"no duration": p_te, "with duration": p_te_l},
                os.path.join(args.out, "fig_roc_pr.png"))
    plot_diagnostics(y_te, p_te, lift, os.path.join(args.out, "fig_diagnostics.png"))
    plot_coefficients(coefs, os.path.join(args.out, "fig_coefficients.png"))
    plot_confusion(scores["at_tuned"]["confusion"], os.path.join(args.out, "fig_confusion.png"),
                   f"Test set, threshold {thr:.2f}")

    pd.DataFrame(coefs).to_csv(os.path.join(args.out, "coefficients.csv"), index=False)
    pd.DataFrame(lift).to_csv(os.path.join(args.out, "lift_table.csv"), index=False)
    report = {
        "n_rows": int(len(df)), "n_columns": len(honest["columns"]),
        "subscription_rate": float(df["y"].eq("yes").mean()),
        "majority_baseline_accuracy": float(df["y"].eq("no").mean()),
        "l2": l2, "cv": cv_table, "threshold": thr,
        "test_no_duration_at_0.5": scores["at_0.5"],
        "test_no_duration_tuned": scores["at_tuned"],
        "train_no_duration_tuned": scores["train_at_tuned"],
        "test_with_duration_tuned": scores_l["at_tuned"],
        "test_chronological_tuned": scores_c["at_tuned"],
        "duration_auc_inflation": gap,
        "lift": lift, "coefficients": coefs,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}/metrics.json, coefficients.csv, lift_table.csv and 4 figures")


if __name__ == "__main__":
    main()





