# Two problems, two folders

| Folder | Task | Method | Result |
| --- | --- | --- | --- |
| [`Problem-1/`](Problem-1/README.md) | Sort 5,856 paediatric chest X-rays into normal or pneumonia | CNN built with TensorFlow and Keras | Test ROC AUC 0.9651, sensitivity 0.9974 (1 missed case out of 390) |
| [`Problem-2/`](Problem-2/README.md) | Predict term-deposit subscription for 45,211 bank customers | Logistic regression, written from scratch (Newton-Raphson) | Test ROC AUC 0.773. Top score decile converts at 48.9% against an 11.7% base rate |

Each folder has its own README with the approach, the methodology and the
findings. Both come with a test that checks the part most likely to be quietly
wrong: that no child appears on both sides of the validation split for the CNN,
and that the hand-derived logistic regression lands on the same coefficients as
scikit-learn's L-BFGS.

## Datasets

Not committed here, since the two of them come to 1.2 GB.

- Problem 1, chest X-rays: [download](https://drive.google.com/file/d/1219EeGE1XTJVXYaulynJSa3BXGsbNCLx/view?usp=sharing)
- Problem 2, Bank Marketing: [download](https://drive.google.com/file/d/18KwSR9aVTZRNaOVF76VE9USSEkqnYzzQ/view?usp=sharing)

Unpack them like this:

```
data/Problem-1/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg
data/Problem-2/bank-data/bank-full.csv
```

Any other layout works too. Both scripts take `--data-root`, the X-ray loader
looks for the class folders and the bank loader looks for the CSV, so extra
nesting is fine.

## Running

Problem 1 needs TensorFlow, which has no wheel for Python 3.14 yet, so it gets
its own 3.12 environment:

```bash
cd Problem-1 && python3.12 -m venv .venv && .venv/bin/pip install tensorflow matplotlib numpy
```

```bash
cd Problem-1 && .venv/bin/python tests/test_split.py && .venv/bin/python train.py && .venv/bin/python evaluate.py
```

```bash
cd Problem-2 && python tests/test_logreg.py && python main.py
```

Problem 2 runs on plain `numpy`, `pandas` and `matplotlib`, with `scikit-learn`
used only as a second opinion inside its test. The CNN trains on a CPU in about
10 seconds an epoch at 96 pixels.

## What the two have in common

Both datasets reward the wrong metric, and most of the work in each folder is
about refusing to report it.

In Problem 2, 88.3% of customers said no. Say "no" to everyone and you score
88.3%, which beats the tuned model's 85.6%. So the model earns its keep on the
lift curve instead: call the top 10% of the ranked list and you reach 42% of all
subscribers at a 48.9% hit rate.

In Problem 1, validation says 0.9992 and the test set says 0.9646. That gap is
not leakage, because the validation split is grouped by patient with no overlap.
The two folders just hold different films, and the prediction spreads pin the
difference to the normal class. Quoting the 0.999 would have been easy and wrong.
