"""Check that the hand-written logistic regression is really correct.

Two separate checks:

* the analytic gradient against central differences, and
* the fitted coefficients against scikit-learn's `lbfgs` solver on the same
  objective. Matching a well-tested library to about 1e-6 says far more than any
  accuracy number.

    python tests/test_logreg.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logreg import LogisticRegression, sigmoid  # noqa: E402


def rel_err(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-30))


def toy(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    true = rng.standard_normal(d)
    y = (rng.random(n) < sigmoid(x @ true - 0.4)).astype(np.float64)
    return x, y


def check_sigmoid():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    s = sigmoid(z)
    ok = np.all(np.isfinite(s)) and s[0] == 0.0 and s[-1] == 1.0 and abs(s[2] - 0.5) < 1e-12
    # the naive 1/(1+exp(-z)) form overflows at -800; ours must not
    print(f"  {'ok ' if ok else 'FAIL'} sigmoid extremes      {s[0]:.1e} .. {s[-1]:.6f}")
    return ok


def check_gradient():
    x, y = toy()
    worst = 0.0
    for l2, cw in [(0.0, None), (2.5, None), (1.0, "balanced")]:
        m = LogisticRegression(l2=l2, class_weight=cw)
        xb = np.hstack([np.ones((len(x), 1)), x])
        w, mask = m._weights(y), m._penalty_mask(x.shape[1])
        beta = np.random.default_rng(1).standard_normal(xb.shape[1]) * 0.3
        analytic = m.gradient(xb, y, beta, w, mask)
        numeric = np.zeros_like(beta)
        h = 1e-6
        for j in range(len(beta)):
            up, dn = beta.copy(), beta.copy()
            up[j] += h
            dn[j] -= h
            numeric[j] = (m.objective(xb, y, up, w, mask) - m.objective(xb, y, dn, w, mask)) / (2 * h)
        worst = max(worst, rel_err(analytic, numeric))
    ok = worst < 1e-7
    print(f"  {'ok ' if ok else 'FAIL'} gradient vs numeric   {worst:.2e}")
    return ok


def check_vs_sklearn():
    """Same penalised objective, two very different solvers."""
    from sklearn.linear_model import LogisticRegression as SkLR
    x, y = toy(n=1500, d=8, seed=3)
    l2 = 4.0
    mine = LogisticRegression(l2=l2).fit(x, y)
    # Ours minimises  mean(nll) + 0.5*l2*||b||^2/n, i.e. after scaling by n,
    # sum(nll) + 0.5*l2*||b||^2. sklearn minimises 0.5*||b||^2 + C*sum(nll),
    # which is sum(nll) + ||b||^2/(2C). Equating the two gives C = 1/l2.
    # Neither implementation penalises the intercept.
    sk = SkLR(C=1.0 / l2, solver="lbfgs", tol=1e-12, max_iter=5000).fit(x, y)
    e_coef = rel_err(mine.coef_, sk.coef_.ravel())
    e_int = abs(mine.intercept_ - float(sk.intercept_[0])) / max(1e-12, abs(float(sk.intercept_[0])))
    e_prob = rel_err(mine.predict_proba(x), sk.predict_proba(x)[:, 1])
    ok = max(e_coef, e_int, e_prob) < 1e-6
    print(f"  {'ok ' if ok else 'FAIL'} vs sklearn lbfgs      coef={e_coef:.2e} "
          f"intercept={e_int:.2e} proba={e_prob:.2e}  ({mine.n_iter_} Newton steps)")
    return ok


def check_balanced_weights():
    """Balanced weighting must lift the predicted positive rate on skewed data."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3000, 4))
    y = (rng.random(3000) < sigmoid(x @ np.array([1.2, -0.8, 0.5, 0.0]) - 2.6)).astype(float)
    base = LogisticRegression(l2=1.0).fit(x, y)
    bal = LogisticRegression(l2=1.0, class_weight="balanced").fit(x, y)
    r0 = base.predict(x).mean()
    r1 = bal.predict(x).mean()
    ok = r1 > r0 and abs(bal.intercept_) < abs(base.intercept_)
    print(f"  {'ok ' if ok else 'FAIL'} balanced reweighting  flag rate {r0:.3f} -> {r1:.3f} "
          f"(true rate {y.mean():.3f})")
    return ok


def check_monotone_loss():
    """Backtracking Newton must never let the objective increase."""
    x, y = toy(n=900, d=10, seed=11)
    m = LogisticRegression(l2=0.5).fit(x, y)
    path = np.asarray(m.loss_path_)
    ok = bool(np.all(np.diff(path) <= 1e-12)) and len(path) > 2
    print(f"  {'ok ' if ok else 'FAIL'} monotone loss path    {path[0]:.4f} -> {path[-1]:.4f} "
          f"in {len(path) - 1} steps")
    return ok


def main():
    print("logistic regression checks:")
    results = [check_sigmoid(), check_gradient(), check_vs_sklearn(),
               check_balanced_weights(), check_monotone_loss()]
    bad = results.count(False)
    print("all passed" if not bad else f"{bad} check(s) FAILED")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

