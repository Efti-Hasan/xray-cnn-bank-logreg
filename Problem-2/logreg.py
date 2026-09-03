"""Binary logistic regression, fitted by Newton-Raphson (IRLS), from scratch.

Why Newton and not gradient descent: the design matrix here is about 50 columns
wide, so the Hessian is a 50x50 matrix that is cheap to build and solve. Newton
then converges in under ten steps with no learning rate to tune, and the same
Hessian hands us the standard errors for free. That matters, because the bank will
want to know which coefficients it can actually trust.

The L2 penalty covers the slopes only. Penalising the intercept would pull the
predicted base rate away from the subscription rate we observed.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    """Overflow-free logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LogisticRegression:
    """L2-regularised logistic regression with optional per-class weights.

    Parameters
    ----------
    l2 : ridge strength on the slopes (0 = unpenalised maximum likelihood)
    class_weight : "balanced" reweights each class to an equal total mass,
        which matters here because only 11.7% of customers subscribe.
    """

    def __init__(self, l2=1.0, class_weight=None, max_iter=60, tol=1e-9):
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.loss_path_ = []

    # ------------------------------------------------------------------ utils
    def _weights(self, y):
        w = np.ones_like(y, dtype=np.float64)
        if self.class_weight == "balanced":
            n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
            w[y == 1] = len(y) / (2.0 * max(n_pos, 1.0))
            w[y == 0] = len(y) / (2.0 * max(n_neg, 1.0))
        elif isinstance(self.class_weight, dict):
            for k, v in self.class_weight.items():
                w[y == k] = float(v)
        return w

    def _penalty_mask(self, n_features):
        """1 for every slope, 0 for the intercept column (which is column 0)."""
        mask = np.ones(n_features + 1)
        mask[0] = 0.0
        return mask

    def objective(self, xb, y, beta, w, mask):
        """Weighted mean negative log-likelihood plus the ridge term."""
        z = xb @ beta
        # log(1+exp(z)) evaluated stably, then the -y*z term folded in.
        nll = np.logaddexp(0.0, z) - y * z
        return float((w * nll).sum() / w.sum() + 0.5 * self.l2 * float(((beta ** 2) * mask).sum()) / w.sum())

    def gradient(self, xb, y, beta, w, mask):
        p = sigmoid(xb @ beta)
        return (xb.T @ (w * (p - y)) + self.l2 * mask * beta) / w.sum()

    # -------------------------------------------------------------------- fit
    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        xb = np.hstack([np.ones((len(x), 1)), x])
        w = self._weights(y)
        mask = self._penalty_mask(x.shape[1])
        beta = np.zeros(xb.shape[1])
        sw = w.sum()
        prev = self.objective(xb, y, beta, w, mask)
        self.loss_path_ = [prev]

        for it in range(1, self.max_iter + 1):
            p = sigmoid(xb @ beta)
            grad = (xb.T @ (w * (p - y)) + self.l2 * mask * beta) / sw
            # IRLS working weights; floored so a saturated fit stays invertible.
            s = np.maximum(w * p * (1.0 - p), 1e-10)
            hess = (xb.T * s) @ xb / sw + np.diag(self.l2 * mask) / sw
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]

            # Backtrack: Newton overshoots when the likelihood is nearly flat.
            t, ok = 1.0, False
            for _ in range(30):
                cand = beta - t * step
                cur = self.objective(xb, y, cand, w, mask)
                if cur <= prev:
                    ok = True
                    break
                t *= 0.5
            if not ok:
                break
            improvement, beta, prev = prev - cur, cand, cur
            self.loss_path_.append(cur)
            self.n_iter_ = it
            if improvement < self.tol:
                break

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        self._beta = beta
        self._hessian = hess
        self._sw = sw
        return self

    # ---------------------------------------------------------------- predict
    def decision_function(self, x):
        return np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, x):
        return sigmoid(self.decision_function(x))

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)

    def standard_errors(self):
        """Wald standard errors from the inverse Hessian at the optimum.

        Read these as indicative: the ridge penalty and the class weights both
        bias the classical covariance estimate, and one-hot dummies from the
        same source column are correlated by construction.
        """
        cov = np.linalg.pinv(self._hessian) / self._sw
        return np.sqrt(np.maximum(np.diag(cov), 0.0))

