# Problem 2: Predicting term deposit subscription with logistic regression

A Portuguese bank ran a phone campaign to sell term deposits. This folder builds
a logistic regression model that scores a customer before the call is made, so
the call centre can work down a ranked list instead of dialling at random.

Accuracy is not the headline number here. Only 11.7% of the 45,211 customers
subscribed. A model that says "no" to everyone is already 88.3% accurate and
completely useless. What matters is whether the top of the ranked list holds more
subscribers than the bottom, and by how much.

## Files

| File | What it does |
| --- | --- |
| `logreg.py` | Logistic regression from scratch: Newton-Raphson (IRLS), L2 penalty, class weights, Wald standard errors |
| `pipeline.py` | Loading, cleaning, feature engineering, one-hot encoding, scaling, and the two split strategies |
| `metrics.py` | Confusion matrix, ROC and PR curves, AUC with tie correction, lift table, calibration, Brier score |
| `main.py` | The whole experiment: cross-validation, fitting, thresholding, figures, `metrics.json` |
| `tests/test_logreg.py` | Correctness checks for the solver |
| `outputs/` | Generated metrics, figures, coefficient and lift tables |

## Running it

```bash
python tests/test_logreg.py
```

```bash
python main.py --data-root ../data/Problem-2 --out outputs
```

Needs `numpy`, `pandas`, `matplotlib`, and `scikit-learn`. The last one is used
only inside the test, as a second opinion on the maths.

## The model is written by hand

`logreg.py` does not call scikit-learn. It minimises

```
  mean_i  w_i · [ log(1 + e^{z_i}) − y_i·z_i ]  +  ½·λ·‖β_slopes‖² / Σw ,   z = Xβ
```

using Newton-Raphson. Each step builds the Hessian `Xᵀ S X` with
`S = diag(w·p·(1−p))`, solves for the update, and backtracks if the objective
would go up. The design matrix has only 57 columns, so the Hessian is a 58×58
matrix. That is cheap to build and cheap to invert, which is why Newton beats
gradient descent here. It converges in six steps and there is no learning rate to
tune. The same inverse Hessian also gives the standard errors in
`outputs/coefficients.csv`.

Saying an implementation is correct is easy, so `tests/test_logreg.py` checks it
five ways:

```
logistic regression checks:
  ok  sigmoid extremes      0.0e+00 .. 1.000000
  ok  gradient vs numeric   2.39e-10
  ok  vs sklearn lbfgs      coef=1.12e-09 intercept=5.63e-09 proba=6.32e-10  (6 Newton steps)
  ok  balanced reweighting  flag rate 0.037 -> 0.327 (true rate 0.125)
  ok  monotone loss path    0.6931 -> 0.3499 in 6 steps
all passed
```

The third line is the important one. Given the same penalised objective, this
solver and scikit-learn's L-BFGS agree on every coefficient to nine decimal
places. Matching a well-tested library says far more than any accuracy figure.
The metric code is checked the same way. `roc_auc`, average precision and the
Brier score reproduce `sklearn.metrics` exactly, including when hundreds of
customers share the same rounded score.

## Preparing the data: four decisions that mattered

Nothing is missing from this dataset, which makes it easy to load it raw and get
a plausible looking but wrong answer. Four judgement calls in `pipeline.py` do
most of the real work.

**`duration` is dropped.** It holds the length of the sales call. A long call
means the customer stayed on the line to say yes, so it is a result of the
outcome rather than a cause of it. Worse, its value is unknown at the moment the
model has to pick who to ring. Keeping it gives a much better looking model that
cannot be deployed. The experiment runs both ways so the size of that illusion
can be measured (see findings).

**`pdays == -1` means "never contacted before", not "contacted yesterday".** As
a plain number it drags a −1 through the mean and standard deviation and turns a
category into a fake distance. It is split into a `never_contacted` flag plus a
distance clipped at zero, so the model can price the two things separately.

**`unknown` is kept as a category instead of being imputed.** For `contact`
(13,020 rows) and `poutcome` (36,959 rows) it is not missing data. It records
something real: no phone number on file, or no earlier campaign to have an
outcome. Filling these with the mode would erase the most informative pattern in
the file.

**Heavy tails are compressed, not clipped.** `balance` runs from −8,019 to
102,127. A signed `log1p` stops one millionaire from bending the decision
boundary while keeping the sign of an overdraft. `campaign` and `previous` get
the same treatment. Continuous columns are then standardised, with the mean and
scale fitted on training rows only. Fitting the scaler on all 45,211 rows would
leak test set information into the transform. Binary dummies are left unscaled on
purpose, so their coefficients still read as "versus the reference level" instead
of "per 0.42 standard deviations of being married".

Month is encoded twice, and that is worth flagging rather than hiding. It goes in
as eleven dummies and also as a `sin`/`cos` pair. The pair can express "late
autumn" using two degrees of freedom, the dummies can express one unusual month,
and the ridge penalty splits the weight between them. It costs nothing except
that individual month coefficients get harder to read on their own.

## How the model was chosen

A 25% test set is held out with stratification, so both halves carry the same
11.7% subscription rate. Everything below happens without touching it.

The ridge strength λ is picked by 5-fold stratified cross-validation on the
training rows, scored on out-of-fold ROC AUC:

| λ | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CV AUC | .7651 | .7651 | .7652 | .7653 | .7654 | .7656 | **.7659** | .7653 |

λ = 30 wins, but the honest reading of that table is that the choice barely
matters. 57 columns against 33,908 training rows is a comfortable ratio, so the
model is nowhere near overfitting and there is little for regularisation to fix.
The flatness is itself a result. Nothing here needs a heavy penalty.

Classes are reweighted by inverse frequency (`class_weight="balanced"`) so the
solver does not just learn to say "no". That pushes the predicted probabilities
up, so the decision threshold has to be worked out again rather than left at 0.5.
It is chosen on a further 80/20 split inside the training data by maximising F1,
and then the model is refitted on the full training set. The threshold never sees
the test set.

## Findings

### 1. `duration` is worth +0.14 AUC that the bank can never have

| Test set model | ROC AUC | PR AUC | Accuracy | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| Without `duration` (deployable) | **0.7729** | 0.4106 | 0.8558 | 0.4887 | 0.4037 |
| With `duration` (leaks the answer) | 0.9144 | 0.5402 | 0.8883 | 0.6641 | 0.5177 |

Adding call duration lifts test ROC AUC from 0.7729 to 0.9144. That is an
inflation of **+0.1414**, more than half the distance from the deployable model to
a perfect one. A report quoting 0.91 would be describing a model that answers
"did this call go well?" after the call is over. The 0.77 figure is what the
campaign can actually plan around.

### 2. Ranking works, and that is what the call centre buys

Accuracy at the tuned threshold is 0.8558, which sits below the 0.883 you get by
predicting "no" every time. That comparison is the trap this dataset sets. The
useful view is the lift table (test set, no `duration`):

| Score decile | Customers | Subscribers | Response rate | Lift | Cumulative recall |
| --- | --- | --- | --- | --- | --- |
| 1 (highest) | 1,130 | 553 | 48.9% | **4.18×** | 41.8% |
| 2 | 1,130 | 191 | 16.9% | 1.45× | 56.3% |
| 3 | 1,131 | 123 | 10.9% | 0.93× | 65.6% |
| 4 | 1,130 | 101 | 8.9% | 0.76× | 73.2% |
| ... | | | | | |
| 10 (lowest) | 1,130 | 27 | 2.4% | 0.20× | 100% |

Calling only the top decile reaches nearly half of all subscribers using one call
in ten, at a 48.9% hit rate against an 11.7% base rate. The top two deciles catch
56% of subscribers for 20% of the dialling effort. That is a real saving, and no
accuracy figure shows it.

### 3. What actually predicts a subscription

Standardised log-odds coefficients, largest effect first. Full table in
`outputs/coefficients.csv`:

| Feature | Coefficient | Odds ratio | Wald z |
| --- | --- | --- | --- |
| `poutcome_success` | +1.358 | 3.89 | +12.3 |
| `month_mar` | +0.928 | 2.53 | +8.2 |
| `month_jan` | −0.896 | 0.41 | −8.3 |
| `month_oct` | +0.804 | 2.23 | +7.3 |
| `poutcome_failure` | −0.779 | 0.46 | −7.3 |
| `contact_unknown` | −0.658 | 0.52 | −6.0 |
| `contact_cellular` | +0.450 | 1.57 | +4.2 |
| `job_retired` | +0.401 | 1.49 | +5.3 |

A customer who said yes to an earlier campaign is **3.9 times more likely** to
subscribe again, holding everything else fixed. It is the strongest signal in the
data and the least surprising one. It is also the most useful, because the bank
already has this field.

Having no phone number on record (`contact_unknown`) roughly halves the odds, and
reaching someone on a mobile beats a landline. These are channel effects rather
than customer preferences, so they argue for better data collection, not for
different targeting.

**The month coefficients are confounded and should not be read as seasonality.**
March, October, September and December look strong while May through August look
weak, but the file covers a single campaign timeline. Call volume per month is
very uneven: May alone holds 13,766 rows and March holds 477. Any change in
script, staffing or product terms over the campaign gets absorbed into whichever
month a call landed in. So the model is partly learning when the bank called well,
not when customers want term deposits. Using these coefficients to schedule next
year's campaign would be a mistake. They are shown here because dropping them
would quietly overstate how reliable the rest of the model is.

### 4. A time-ordered split shows the drift

Rows arrive in campaign order but only carry day and month, so the year is
recovered by counting the points where the month number rolls backwards. Trained
on the earlier 75% of calls and tested on the later 25%:

| Split | ROC AUC | Accuracy | Recall |
| --- | --- | --- | --- |
| Random (stratified) | 0.7729 | 0.8558 | 0.4887 |
| Chronological | 0.7266 | 0.7263 | 0.2137 |

AUC drops by 0.046 and recall falls from 0.49 to 0.21. A random split lets the
model train on the same weeks it gets scored on, which flatters it. The
chronological number is a better forecast of first-month production performance,
and it says the model needs refitting on a schedule rather than a one-time
deployment.

### 5. Calibration, and the cost of a threshold

The Brier score is 0.1830 and the calibration curve
(`outputs/fig_diagnostics.png`) sits above the diagonal throughout. That comes
straight from the balanced class weights, which trade calibrated probabilities for
a usable decision boundary. So the scores are reliable as a ranking but should not
be quoted as "this customer has a 60% chance". If a real probability is needed,
drop the weighting and re-tune the threshold instead.

The threshold is a business dial, not a fixed part of the model. At 0.5 the model
flags 28% of customers and catches 64% of subscribers. At the F1-optimal 0.605 it
flags 14% and catches 49%. Which one is right depends on the cost of a wasted call
against the margin on a term deposit. The model supplies the trade-off curve, not
the decision. Train and test metrics at the tuned threshold agree to within 0.002
accuracy (0.8540 against 0.8558), so nothing here is overfit.

## Limitations

- 57 columns of linear log-odds cannot express "young *and* high balance". A tree
  ensemble would find interactions this model structurally cannot. Logistic
  regression was the requirement, and in return it gives signed, interpretable
  effects with standard errors.
- The confound in finding 3 applies to any single-campaign dataset. More modelling
  will not fix it, only data from more than one campaign will.
- Balanced weighting is not the only way to handle 11.7% imbalance. It is the one
  that keeps the objective convex and the standard errors meaningful.
