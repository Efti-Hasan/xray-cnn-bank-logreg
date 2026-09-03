"""Load and prepare the Bank Marketing table.

Three judgement calls live here, and they matter more than the model does:

1. `duration` is the length of the call the campaign is trying to cause. Nobody
   knows it until the call ends, so keeping it makes the model look far better
   than it can ever be in production. The loader can build the matrix either way,
   so the README can measure exactly how big that illusion is.
2. `pdays == -1` does not mean "contacted 1 day ago", it means "never contacted".
   Left as a number it drags a -1 through the scaler and ruins the coefficient. It
   becomes a flag plus a distance zeroed out.
3. `unknown` is kept as its own category instead of being filled in. For `contact`
   (13,020 rows) and `poutcome` (36,959 rows) it records something real: no
   reachable phone, or no earlier campaign.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

CATEGORICAL = ("job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome")
NUMERIC = ("age", "balance", "day", "campaign", "pdays", "previous")
LEAKY = ("duration",)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def find_csv(root: str) -> str:
    """Locate bank-full.csv without assuming how deeply it is nested."""
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower() in ("bank-full.csv", "bank-additional-full.csv", "bank.csv"):
                return os.path.join(dirpath, name)
    raise FileNotFoundError(f"no bank CSV found under {root}")


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", quotechar='"')
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns that give the linear model a fairer chance."""
    out = df.copy()
    out["never_contacted"] = (out["pdays"] < 0).astype(int)
    out["pdays"] = out["pdays"].clip(lower=0)
    out["was_contacted_before"] = (out["previous"] > 0).astype(int)
    # Money and call counts are heavy-tailed; a log-ish transform keeps a single
    # millionaire from dominating the fit. signed log1p handles the overdrafts.
    out["balance_signed_log"] = np.sign(out["balance"]) * np.log1p(out["balance"].abs())
    out["campaign_log"] = np.log1p(out["campaign"])
    out["previous_log"] = np.log1p(out["previous"])
    out["month_num"] = out["month"].str.lower().map(MONTHS).astype("float64")
    # Calendar position as a smooth pair instead of 12 unordered dummies, so
    # "late autumn" can be expressed without spending 12 degrees of freedom.
    ang = 2 * np.pi * (out["month_num"] - 1) / 12.0
    out["month_sin"], out["month_cos"] = np.sin(ang), np.cos(ang)
    return out


DERIVED_NUMERIC = ("never_contacted", "was_contacted_before", "balance_signed_log",
                   "campaign_log", "previous_log", "month_sin", "month_cos")


def build_matrix(df: pd.DataFrame, include_duration: bool, columns=None):
    """One-hot the categoricals, keep the numerics, return (X, y, column names).

    `columns` pins the output to a previously seen column order so that the test
    matrix lines up with the training matrix even if a rare category is absent.
    """
    num = list(NUMERIC) + list(DERIVED_NUMERIC)
    if include_duration:
        num += list(LEAKY)
    dummies = pd.get_dummies(df[list(CATEGORICAL)], prefix=CATEGORICAL, dtype=np.float64)
    x = pd.concat([df[num].astype(np.float64), dummies], axis=1)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    y = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(np.int8).to_numpy()
    return x, y, list(x.columns)


class Standardiser:
    """Zero-mean unit-variance scaling fitted on the training rows only.

    Binary dummy columns are left alone: scaling them makes the coefficients
    unreadable ("per 0.42 of a standard deviation of being married") for no
    numerical gain.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.mask_ = None

    def fit(self, x: pd.DataFrame):
        values = x.to_numpy(dtype=np.float64)
        self.mask_ = np.array([not set(np.unique(values[:, j])) <= {0.0, 1.0}
                               for j in range(values.shape[1])])
        self.mean_ = np.where(self.mask_, values.mean(axis=0), 0.0)
        std = values.std(axis=0)
        self.scale_ = np.where(self.mask_ & (std > 0), std, 1.0)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return (x.to_numpy(dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def stratified_split(y, test_frac=0.25, seed=0):
    """Index split that preserves the 11.7% subscription rate in both halves."""
    rng = np.random.default_rng(seed)
    test = []
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)
        test.append(rows[:int(round(test_frac * len(rows)))])
    test_idx = np.sort(np.concatenate(test))
    mask = np.ones(len(y), dtype=bool)
    mask[test_idx] = False
    return np.flatnonzero(mask), test_idx


def chronological_split(df, test_frac=0.25):
    """Train on the earlier calls, test on the later ones.

    The rows arrive in campaign order but only carry day-and-month, so the year
    is recovered by counting the points where the month number goes backwards.
    A random split lets the model peek at the same weeks it is scored on; this
    one does not, which is the harder and more realistic question to ask of a
    campaign model.
    """
    month = df["month"].str.lower().map(MONTHS).to_numpy()
    year = np.cumsum(np.r_[0, (np.diff(month) < 0).astype(int)])
    stamp = year * 10000 + month * 100 + df["day"].to_numpy()
    order = np.argsort(stamp, kind="mergesort")
    cut = int(round((1 - test_frac) * len(order)))
    return np.sort(order[:cut]), np.sort(order[cut:])


def prepare(root: str, include_duration: bool, test_frac=0.25, seed=0, split="stratified"):
    """One call to go from a folder path to model-ready arrays."""
    df = engineer(load_raw(find_csv(root)))
    x_df, y, columns = build_matrix(df, include_duration)
    if split == "chronological":
        tr, te = chronological_split(df, test_frac)
    else:
        tr, te = stratified_split(y, test_frac, seed)
    scaler = Standardiser().fit(x_df.iloc[tr])
    x = scaler.transform(x_df)
    return {"x_train": x[tr], "y_train": y[tr], "x_test": x[te], "y_test": y[te],
            "columns": columns, "frame": df, "scaler": scaler,
            "n_train": len(tr), "n_test": len(te)}


