# Problem 2: Term Deposit Subscription Prediction (Logistic Regression)

A Portuguese bank ran a phone campaign selling term deposits. The idea here is
to score customers *before* the call gets made, so the call centre can work
down a ranked list instead of dialing randomly.

Accuracy isn't really the useful number in this dataset. Only 11.7% of the
45,211 customers actually subscribed, so a model that just says "no" every
time is already 88.3% accurate and completely useless. What matters is
whether the top of the ranked list actually has more subscribers than the
bottom, and by how much.

## Files

- `logreg.py` — logistic regression from scratch (Newton-Raphson/IRLS, L2
  penalty, class weights, Wald standard errors)
- `pipeline.py` — loading, cleaning, feature engineering, encoding, scaling,
  and both split strategies
- `metrics.py` — confusion matrix, ROC/PR curves, AUC with tie correction,
  lift table, calibration, Brier score
- `main.py` — runs the whole experiment: CV, fitting, thresholding, figures,
  metrics.json
- `tests/test_logreg.py` — correctness checks on the solver
- `outputs/` — generated metrics, figures, coefficient and lift tables

## Running

```bash
python tests/test_logreg.py
python main.py --data-root ../data/Problem-2 --out outputs
```

Needs numpy, pandas, matplotlib, scikit-learn. sklearn is only used inside the
test file, as a second opinion on the math, not in the actual model.

## Why I wrote the logistic regression by hand

logreg.py doesn't call sklearn to fit anything. It minimizes:

```
mean_i  w_i * [ log(1 + e^z_i) - y_i*z_i ]  +  0.5*lambda*||beta_slopes||^2 / sum(w),   z = X*beta
```

using Newton-Raphson. Each step builds the Hessian (X^T S X) with S =
diag(w*p*(1-p)), solves for the update, backtracks if the objective would go
up instead of down. The design matrix only has 57 columns so the Hessian is a
tiny 58x58 matrix — cheap to build and invert, which is exactly why Newton
beats plain gradient descent for something this size. Converges in 6 steps,
no learning rate to babysit. The same inverse Hessian also gives the standard
errors that end up in outputs/coefficients.csv.

Saying "it's correct" isn't good enough on its own, so test_logreg.py checks
it 5 different ways:

```
logistic regression checks:
  ok  sigmoid extremes      0.0e+00 .. 1.000000
  ok  gradient vs numeric   2.39e-10
  ok  vs sklearn lbfgs      coef=1.12e-09 intercept=5.63e-09 proba=6.32e-10  (6 Newton steps)
  ok  balanced reweighting  flag rate 0.037 -> 0.327 (true rate 0.125)
  ok  monotone loss path    0.6931 -> 0.3499 in 6 steps
all passed
```

The third check is the one that actually matters — given the same penalized
objective, this solver and sklearn's L-BFGS agree on every coefficient to 9
decimal places. That agreement means a lot more than any accuracy number
would. Metric functions are checked the same way — roc_auc, average
precision, and Brier score all reproduce sklearn.metrics exactly, including
in the tie-heavy case where hundreds of customers land on the same rounded
score.

## Data prep — the parts that actually mattered

There's no missing data in this dataset technically, which is a trap — it's
easy to load it raw and get a plausible-looking but wrong model. Most of the
real work in pipeline.py comes down to four decisions.

**Dropped `duration`.** This is the length of the sales call. A long call
usually means the customer stuck around long enough to say yes — so it's a
consequence of the outcome, not something that predicts it. Also, you don't
know it until after the call happens, which is exactly when the model is
supposed to have already made its prediction. Keeping it makes the model look
much better but makes it undeployable. I ran the experiment both ways just to
measure how big that illusion actually is (see findings below).

**`pdays == -1` handled as a flag, not a number.** -1 means "never contacted
before," not "contacted yesterday." As a raw number it drags a -1 through the
mean and std and turns a category into a fake distance. Split it into a
never_contacted flag plus a distance clipped at zero so the model can treat
the two things separately.

**Kept `unknown` as its own category instead of imputing it.** For contact
(13,020 rows) and poutcome (36,959 rows), "unknown" isn't really missing data
— it means something real (no phone number on file, or no previous campaign
to have an outcome from). Filling it with the mode would erase probably the
most informative signal in the whole file.

**Compressed heavy tails instead of clipping them.** balance ranges from
-8,019 to 102,127. Used a signed log1p so one outlier account doesn't warp the
decision boundary, while still keeping the sign for overdrafts. Did the same
for campaign and previous. Continuous columns get standardized after that,
with the mean and scale fit only on the training rows — fitting the scaler on all
45,211 rows first would leak test info into the transform. Left binary dummies
unscaled on purpose so their coefficients still read as "vs the reference
category" rather than some awkward "per 0.42 std devs of being married."

Worth flagging: month is encoded twice — as 11 dummies and as a sine and cosine pair.
The pair lets the model express something like "late autumn" with 2 degrees
of freedom, the dummies can flag one specific weird month, and the ridge
penalty just splits weight between them. Doesn't cost much except that
individual month coefficients get a bit harder to interpret in isolation.

## Model selection

Held out 25% as a test set with stratification so both halves keep the same
11.7% subscription rate. Nothing below touches that test set.

Ridge strength (lambda) picked via 5-fold stratified CV on training data,
scored on out-of-fold ROC AUC:

| lambda | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|
| CV AUC | .7651 | .7651 | .7652 | .7653 | .7654 | .7656 | .7659 | .7653 |

lambda=30 technically wins but the honest read is that it barely matters at
all here. 57 columns vs 33,908 training rows is a very comfortable ratio, so
the model isn't close to overfitting and there's not much for regularization
to actually fix. The flatness of that table is itself kind of the result —
nothing here needed a heavy penalty.

Classes reweighted by inverse frequency (class_weight="balanced") so the
solver doesn't just default to predicting "no" for everyone. That pushes
predicted probabilities up across the board, so the threshold has to be
re-derived rather than left at 0.5 — picked by maximizing F1 on an 80-20 split
carved out of the training data, then the model gets refit on the full
training set. Threshold selection never touches the test set.

## Findings

### 1. `duration` is worth +0.14 AUC the bank will never actually have access to

| Test model | ROC AUC | PR AUC | Accuracy | Recall | Precision |
|---|---|---|---|---|---|
| Without duration (deployable) | 0.7729 | 0.4106 | 0.8558 | 0.4887 | 0.4037 |
| With duration (leaks the answer) | 0.9144 | 0.5402 | 0.8883 | 0.6641 | 0.5177 |

Adding call duration bumps test AUC from 0.7729 to 0.9144 — an inflation of
+0.1414, more than half the gap between the deployable model and a perfect
one. A report quoting 0.91 here would basically be describing a model that
answers "did this call go well," after the call already happened. 0.77 is
what the campaign can actually plan around.

### 2. The ranking still works — that's the actual product here

Accuracy at the tuned threshold is 0.8558, which is *below* the 0.883 you'd
get by just predicting "no" for everyone. That comparison is basically the
trap this dataset sets for you. The number that actually matters is the lift
table (test set, no duration):

| Decile | Customers | Subscribers | Response rate | Lift | Cumulative recall |
|---|---|---|---|---|---|
| 1 (top) | 1,130 | 553 | 48.9% | 4.18x | 41.8% |
| 2 | 1,130 | 191 | 16.9% | 1.45x | 56.3% |
| 3 | 1,131 | 123 | 10.9% | 0.93x | 65.6% |
| 4 | 1,130 | 101 | 8.9% | 0.76x | 73.2% |
| ... | | | | | |
| 10 (bottom) | 1,130 | 27 | 2.4% | 0.20x | 100% |

Just calling the top decile catches nearly half of all subscribers using 1
call out of 10, at a 48.9% hit rate against an 11.7% baseline. Top 2 deciles
get you 56% of subscribers for 20% of the dialing effort. That's a real,
usable saving, and accuracy alone would never show it.

### 3. What actually predicts a subscription

Standardized log-odds coefficients, biggest effects first (full table in
outputs/coefficients.csv):

| Feature | Coefficient | Odds ratio | Wald z |
|---|---|---|---|
| poutcome_success | +1.358 | 3.89 | +12.3 |
| month_mar | +0.928 | 2.53 | +8.2 |
| month_jan | -0.896 | 0.41 | -8.3 |
| month_oct | +0.804 | 2.23 | +7.3 |
| poutcome_failure | -0.779 | 0.46 | -7.3 |
| contact_unknown | -0.658 | 0.52 | -6.0 |
| contact_cellular | +0.450 | 1.57 | +4.2 |
| job_retired | +0.401 | 1.49 | +5.3 |

A customer who said yes on a previous campaign is roughly 3.9x more likely to
subscribe again, holding everything else constant. Strongest signal in the
data, and not a surprising one — but also the most useful one since the bank
already has this field for free.

Not having a phone number on record (contact_unknown) roughly halves the
odds, and mobile beats landline. These read more like channel or contactability
effects than actual customer preference, so they're really an argument for
better data collection, not for different targeting.

Month coefficients are confounded, and I don't think they should be read as
seasonality. March, October, September, and December look strong, while May through August look
weak, but this is a single campaign timeline, not multiple years of data. Call
volume per month is wildly uneven — May alone has 13,766 rows, March has 477.
Any change in script, staffing, or product terms during the campaign gets
absorbed into whichever month it happened to land in. So the model is partly
learning "when the bank happened to call well" rather than "when customers
actually want term deposits." Using these numbers to schedule next year's
campaign would probably be a mistake. Kept them in the writeup anyway because
dropping them would quietly make the rest of the model look more reliable
than it is.

### 4. Time-based split shows real drift

Rows come in campaign order but only have day and month, no year — recovered the
year by counting where the month number rolls backward. Trained on the
earlier 75% of calls, tested on the later 25%:

| Split | ROC AUC | Accuracy | Recall |
|---|---|---|---|
| Random (stratified) | 0.7729 | 0.8558 | 0.4887 |
| Chronological | 0.7266 | 0.7263 | 0.2137 |

AUC drops 0.046, recall falls from 0.49 to 0.21. Random split lets the model
train on roughly the same weeks it gets tested on, which flatters the number.
The chronological split is probably closer to what actual first-month
production performance would look like, and it suggests the model needs
periodic refitting rather than a one-time deploy-and-forget.

### 5. Calibration, and the threshold is really a business decision

Brier score is 0.1830, and the calibration curve (outputs/fig_diagnostics.png)
sits above the diagonal the whole way through. That's a direct consequence of
the balanced class weights — they buy a usable decision boundary but cost
calibrated probabilities. So the scores are good for ranking but shouldn't be
quoted as "this customer has a 60% chance of subscribing." If a real
probability estimate is needed, drop the class weighting and re-tune the
threshold separately.

The threshold itself is more of a business dial than a fixed model output. At
0.5 the model flags 28% of customers and catches 64% of subscribers. At the
F1-optimal 0.605 it flags 14% and catches 49%. Which one's "right" depends on
what a wasted call costs vs the margin on a term deposit — the model gives
you the tradeoff curve, not the actual decision. Train and test accuracy at the
tuned threshold only differ by 0.002 (0.8540 vs 0.8558), so nothing here looks
overfit.

## Limitations

- 57 columns of linear log-odds can't express something like "young AND high
  balance" — a tree ensemble would pick up interactions this model
  structurally can't. Logistic regression was the requirement here though,
  and in exchange you get signed, interpretable effects with actual standard
  errors.
- The month confound in finding 3 is a property of any single-campaign
  dataset — more modeling won't fix it, only data spanning multiple campaigns
  would.
- Balanced weighting isn't the only way to deal with 11.7% imbalance, just
  the one that keeps the objective convex and the standard errors meaningful.