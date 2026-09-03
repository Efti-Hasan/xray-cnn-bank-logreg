"""Checks on the validation split, which is the one place leakage could hide.

    python tests/test_split.py

The model itself is Keras, so there is no hand-written backward pass left to
verify. What still needs checking is the thing Keras does not do for us: keeping
every child on one side of the train/validation line. A random split would put
two films of the same lung in train and validation, and the validation score
would then measure memory rather than generalisation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import data as ds  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "Problem-1")


def fake_names(n_normal=40, n_pneu=90, per_patient=3):
    """Filenames in the shapes the real folders use, with repeats per patient."""
    names, y = [], []
    for i in range(n_normal):
        names.append(f"NORMAL/IM-{i // per_patient:04d}-000{i % per_patient}.jpeg")
        y.append(0)
    for i in range(n_pneu):
        names.append(f"PNEUMONIA/person{i // per_patient}_bacteria_{i}.jpeg")
        y.append(1)
    return names, np.asarray(y, dtype=np.int8)


def patients_of(names, idx):
    return {ds.patient_key(names[i].split("/", 1)[1]) for i in idx}


def check(label, condition, detail=""):
    print(f"  {'pass' if condition else 'FAIL'}  {label}{detail}")
    return bool(condition)


def test_keys():
    print("patient ids come out of the filenames")
    cases = {
        "person1_bacteria_1.jpeg": "person1",
        "person1_virus_6.jpeg": "person1",
        "IM-0115-0001.jpeg": "im-0115",
        "NORMAL2-IM-0381-0001.jpeg": "normal2-im-0381",
    }
    ok = True
    for fname, want in cases.items():
        got = ds.patient_key(fname)
        ok &= check(f"{fname} -> {got}", got == want, "" if got == want else f", wanted {want}")
    return ok


def test_split(names, y, tag):
    n = len(y)
    tr, val = ds.grouped_split(names, y, frac=0.15, seed=7)
    ok = True
    print(f"{tag}: {n} images, {len(tr)} train, {len(val)} validation")
    ok &= check("every image used once", sorted(np.r_[tr, val].tolist()) == list(range(n)))
    shared = patients_of(names, tr) & patients_of(names, val)
    ok &= check("no patient on both sides", not shared, f", {len(shared)} shared" if shared else "")
    for label, cls in enumerate(ds.CLASSES):
        held = float(np.mean(y[val] == label)) if len(val) else 0.0
        want = float(np.mean(y == label))
        ok &= check(f"{cls} share held: {held:.3f} against {want:.3f} overall",
                    abs(held - want) < 0.05)
    ok &= check("validation size near 15%", abs(len(val) / n - 0.15) < 0.03,
                f", got {len(val) / n:.3f}")
    return ok


def main():
    names, y = fake_names()
    ok = test_keys() and test_split(names, y, "synthetic names")
    if os.path.isdir(os.path.join(ROOT, "train", "NORMAL")):
        names, y = [], []
        for label, cls in enumerate(ds.CLASSES):
            for f in ds.list_split(ROOT, "train", cls):
                names.append(f"{cls}/{f}")
                y.append(label)
        ok = test_split(names, np.asarray(y, dtype=np.int8), "the real train/ folder") and ok
    else:
        print("real dataset not found, synthetic check only")
    print("\nall checks passed" if ok else "\nsomething failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
