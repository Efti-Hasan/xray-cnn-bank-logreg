# Problem 1: Paediatric Chest X-ray Classification (CNN)

Dataset: 5,856 chest X-rays from kids aged 1-5, labeled either Normal or
Pneumonia. Goal was to build a CNN that takes an X-ray and outputs the
probability of pneumonia, plus actually check whether that probability can be
trusted (which turned out to be the harder part).

Built with TensorFlow and Keras, fairly standard setup — Conv2D and MaxPooling
blocks, a small head, then compile, fit, and plot. Keras handles the backprop so that part
isn't really "my work" in any meaningful sense. What actually took effort was:

- splitting train and validation without leaking the same patient into both sides
- dealing with the 2.9:1 class imbalance in the loss
- the learning rate schedule (mattered way more than I expected going in)
- figuring out where the decision threshold should come from, and on what data

## Files

- `data.py` — finds/loads/caches the images, splits train/val by patient
- `train.py` — builds and trains the model, saves model + training curves
- `evaluate.py` — runs the model on test/, picks threshold, dumps metrics + plots
- `metrics.py` — confusion matrix, sensitivity/specificity, AUC, threshold search
- `probe.py` — the two extra checks behind findings 1 and 5 below
- `tests/test_split.py` — makes sure no patient leaks across train/val

## Running

TensorFlow doesn't support Python 3.14 yet so use 3.12:

```bash
python3.12 -m venv .venv && .venv/bin/pip install tensorflow matplotlib numpy
```

```bash
.venv/bin/python tests/test_split.py
.venv/bin/python train.py --data-root ../data/Problem-1 --out outputs
.venv/bin/python evaluate.py --out outputs
.venv/bin/python probe.py --out outputs
```

Only needs tensorflow, matplotlib, and numpy — images are decoded and resized with
tf.io and tf.image, so no opencv is needed. First run caches decoded images as .npz, after
that it's fast. ~10s/epoch at 96px on CPU, ~19s at 128px.

## Splitting the data properly

The dataset ships with a train folder (5,216 images), a test folder (624), and a val folder (only 16). The val folder is basically useless for picking a model — 16 images means each one is
worth over 6% of the score, so you can't actually tell two models apart with
that. So I carved a real validation set out of the train folder instead.

Can't just do it randomly though, because filenames include patient IDs.
`person1000_bacteria_2931.jpeg` and `person1000_virus_1681.jpeg` are the same
kid, and some patients have up to 30 images. Random split = same kid's X-rays
in both train and val = model can just memorize the kid instead of learning
what pneumonia looks like. So data.py pulls the ID from the filename and
splits by patient:

```
train  4,432 images  (1,140 normal, 3,292 pneumonia)
val      784 images  (  201 normal,   583 pneumonia)
patients held out: 444, overlap: 0
```

test_split.py checks this on the real folder — every image used once, no
patient split across sides, class ratio roughly preserved.

Two smaller calls I made:

**No horizontal flip.** Everyone defaults to this augmentation for images but
it doesn't make sense here — the heart is on one side, so a flipped chest
X-ray is anatomically wrong. Kept rotation, translation, and zoom since patient
positioning does vary that way in practice.

**Class weights over resampling.** Pneumonia outnumbers normal ~2.9:1 in
train. Used inverse-frequency weights (normal 1.944, pneumonia 0.673) via
class_weight instead of throwing away pneumonia images to balance the count —
didn't want to waste real data just to make the ratio look nicer.

## The model

```
Rescaling(1/255) -> RandomRotation -> RandomTranslation -> RandomZoom
Conv3x3(1->16) BN ReLU MaxPool2
Conv3x3(16->32) BN ReLU MaxPool2
Conv3x3(32->64) BN ReLU MaxPool2
Conv3x3(64->128) BN ReLU MaxPool2
GlobalAveragePooling -> Dropout(0.4) -> Dense(1, sigmoid)
```

98,241 params (97,761 trainable). Kept it small since there's only 4,432
training images — a bigger model would just memorize instead of generalizing.

Augmentation is inside the model as layers, so it runs during training and
turns off automatically at inference, no extra code needed. Head is just
GlobalAvgPool -> Dense(1), no hidden layer — which means the 128 dense weights
are literally per-channel importance scores, so a class activation map is one
dot product against the last feature map. Wasn't trying to be fancy, just
picked a head where interpretability comes for free.

Picked models based on validation ROC AUC, not accuracy — AUC doesn't need a
threshold picked first, so a model can't win selection just by looking good at
an arbitrary 0.5 cutoff. Threshold gets decided later in evaluate.py, using
only validation data.

## Findings

### 1. Val AUC 0.9992, test AUC 0.9646 — not leakage, not really overfitting either

Three configs were trained total (compared in the table further down).
Findings 1-3 and 5 below are from the baseline run specifically.

| | Val (784, patient-grouped) | Test (624, untouched) |
|---|---|---|
| ROC AUC | 0.9992 | 0.9646 |
| Accuracy | 0.9834 | 0.8734 |
| Sensitivity | 0.9828 | 0.9897 |
| Specificity | 0.9851 | 0.6795 |
| Confusion | tn 198 fp 3 fn 10 tp 573 | tn 159 fp 75 fn 4 tp 386 |

First thing I checked was leakage but there's zero patient overlap between
train and val, so that's not it. Also probably not just overfitting — a model
that memorized train wouldn't hit 0.99 on 444 patients it's never seen. Best
explanation is that the train and test folders just aren't quite the same distribution.

Ran the probe script to check where exactly the gap shows up:

| Predicted prob of pneumonia | median | 10th pct | 90th pct |
|---|---|---|---|
| Val, NORMAL | 0.000 | 0.000 | 0.008 |
| Test, NORMAL | 0.139 | 0.001 | 0.982 |
| Val, PNEUMONIA | 1.000 | 0.935 | 1.000 |
| Test, PNEUMONIA | 1.000 | 0.979 | 1.000 |

Interesting bit — the model is actually more confident on test pneumonia
cases than val ones (10th pct 0.979 vs 0.935). All the damage is on the
normal side, where the 90th percentile jumps to 0.982 on test. So it's not
that the model got worse at spotting pneumonia (sensitivity even goes up
slightly), it's that it starts calling healthy test films sick, which craters
specificity (0.9851 -> 0.6795).

Checked raw pixel stats too — nearly identical (mean 122.2, std 59.9 on val
vs mean 120.6, std 59.9 on test), so it's not a scanner or exposure
difference. Whatever counts as "normal" seems to differ a bit between the two
folders, which fits with the test set having been collected and labeled
separately. Point is, quoting just
the 0.9992 number would be misleading.

### 2. LR schedule mattered more than anything about the architecture

First version of train.py used plain Adam(1e-3), constant rate, no weight
decay, early stopping after 6 epochs. Looked totally fine — val AUC hit
0.9976.

| | Val AUC | Test AUC | Test acc | Sens | Spec | False alarms |
|---|---|---|---|---|---|---|
| Adam(1e-3) constant | 0.9976 | 0.9216 | 0.7644 | 0.9897 | 0.3889 | 143 |
| AdamW + warmup/cosine, wd 1e-4 | 0.9992 | 0.9646 | 0.8734 | 0.9897 | 0.6795 | 75 |

Val barely cared (0.0016 AUC difference). Test cared a lot — 0.043 AUC, 68
fewer false alarms out of 234 normal films.

Epoch counts explain it. Constant LR run: early stopping fired at epoch 12,
best checkpoint at epoch 6 — val AUC had basically plateaued while weights
were still bouncing. Warmup+cosine run: went to epoch 30, best at 27, and
those last low-LR epochs are what actually settle the decision boundary. No
layer changes between the two — this was the biggest single effect in the
whole project, bigger than any architecture difference below.

Caveat: three things changed at once here (schedule, weight decay, patience),
so it's not a clean isolated ablation of just the LR schedule.

### 3. Threshold tuned on val doesn't transfer cleanly to test

Tuning the threshold on val under a 98% sensitivity floor gives 0.298.
Applying that same number to test:

| Threshold | Accuracy | Sensitivity | Specificity | Missed | False alarms |
|---|---|---|---|---|---|
| 0.5 (default) | 0.8734 | 0.9897 | 0.6795 | 4 | 75 |
| 0.298 (val-tuned) | 0.8429 | 0.9923 | 0.5940 | 3 | 95 |

It does move sensitivity up (3 missed instead of 4 out of 390), but costs 20
extra false alarms because the normal-class scores have shifted underneath
the threshold. Same story as finding 1, in practical terms — the ranking
survives the shift, the calibration doesn't.

Reporting both thresholds rather than just the flattering one. For a
screening tool the sensitivity trade is probably worth it anyway — a false
alarm costs a second look, a missed pneumonia case in a toddler costs way
more.

### 4. Validation ranked the three runs in exactly the wrong order

| Run | Input | Aug | Dropout | Val AUC | Test AUC | Test acc | Sens | Spec | Missed | False alarms | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 96² | 1.0x | 0.40 | 0.9992 | 0.9646 | 0.8734 | 0.9897 | 0.6795 | 4 | 75 | 27 |
| strong-aug | 96² | 1.8x | 0.50 | 0.9959 | 0.9669 | 0.8285 | 0.9974 | 0.5470 | 1 | 106 | 13 |
| higher-res | 128² | 1.4x | 0.45 | 0.9985 | 0.9651 | 0.8670 | 0.9974 | 0.6496 | 1 | 82 | 28 |

Weight decay 1e-4 in all three. Two things worth pulling out:

Val put baseline first, but on test baseline is worst of the three. Both runs
val liked less actually cut missed cases from 4 down to 1. Differences are
small either way (0.0033 AUC on val, 0.0023 on test) so I wouldn't read too
much into any single number, but the ordering flipping is the actual point —
once val AUC is sitting at 0.999 there's basically no signal left to select
on.

Also, more resolution didn't really help. 96px -> 128px moved test AUC from
0.9646 to 0.9651, for roughly double the training time (19 seconds vs 10 seconds per epoch).
Rules out one explanation for finding 1 — it's not that the model is throwing
away useful detail at 96px, since giving it back barely changes the number.
What 128px did improve was the operating point (3 fewer missed cases, 24
fewer false alarms than strong-aug).

These are three different configs, not a clean sweep — resolution, aug
strength, and dropout all changed together in each run.

Training curves back this up — in outputs/fig_training.png val AUC is above
0.99 by epoch 3 and barely moves after, while accuracy and sensitivity keep
swinging for another dozen epochs. Those swings are just the 0.5 cutoff
sliding around under a model whose ranking is already locked in, which is
basically the whole argument for selecting on AUC and handling the threshold
separately.

### 5. What the model actually looks at (and a shortcut-learning scare)

outputs/fig_cam.png overlays class activation maps on 2 confident correct
calls and the 6 worst mistakes. Correct pneumonia calls: heat sits over the
lower lung field, as expected. False positives: heat spreads across both
lungs and in several cases lands right on the image border — bottom edge
below the diaphragm, and the top-left corner where the "R" laterality marker
is burned into the film. That corner thing is a textbook shortcut-learning
red flag, so I didn't want to just show the maps and call it done.

Ran an occlusion test — replaced a border band with the image's own median
grey value (a value the model has actually seen before) and rescored:

| Input | ROC AUC | Sens | Spec | FA | Missed |
|---|---|---|---|---|---|
| original | 0.9646 | 0.9897 | 0.6795 | 75 | 4 |
| top 12% masked | 0.9572 | 0.9949 | 0.3974 | 141 | 2 |
| left and right 12% masked | 0.9271 | 0.9462 | 0.6923 | 72 | 21 |
| whole border masked | 0.9111 | 0.9487 | 0.6581 | 80 | 20 |

Honestly this test is inconclusive and I think that's the actual finding.
Masking the corner with the marker makes false positives almost double (75 ->
141), so the marker itself doesn't look like the shortcut. Masking the sides
trims false alarms a bit (75 -> 72) but pushes missed cases from 4 to 21. AUC
drops under every masking condition, so the border clearly does carry real
signal — but blanking it is also an input the model has never seen, which
muddies things. Can't cleanly separate "reading the marker" from "reading the
outer lung fields" with this setup. At 96px with 4 pooling stages the final
feature map is 6x6, so one cell covers a 16x16 pixel block — big enough to
hold both the marker and the lung apex at once.

Would need to actually paint the marker out (not just mask a band) to settle
this properly. Didn't want to just claim "the model looks at the lungs" when
I can't fully back that up.

### 6. Which one I'd ship

outputs_128, the 128px run. Test AUC 0.9651, sensitivity 0.9974, 1 missed
pneumonia out of 390, 82 false alarms out of 234 normal films.

To be upfront — picking "best on test" after comparing 3 models against the
test set isn't a clean generalization estimate, it's a choice made looking at
that exact data. And the AUC isn't really the reason for the pick anyway —
all three runs sit within 0.0023 of each other, a handful of images out of
624. The real reason is that in paediatric screening the two error types
aren't equally bad, and both regularized runs tie at 1 missed case vs 4 for
baseline. Between those two, 128px gets there with 24 fewer false alarms.

So: 0.9651 is a test AUC that was observed while 3 models were being
compared against each other, so treat it as a bit optimistic. The range
across the three runs (0.9646-0.9669) is probably the more honest number for
what this architecture actually does on this data.

## Reproducing

```bash
.venv/bin/python train.py --out outputs
.venv/bin/python evaluate.py --out outputs
.venv/bin/python probe.py --out outputs

.venv/bin/python train.py --out outputs_aug --epochs 40 --aug-strength 1.8 --dropout 0.5
.venv/bin/python evaluate.py --out outputs_aug

.venv/bin/python train.py --out outputs_128 --img 128 --aug-strength 1.4 --dropout 0.45
.venv/bin/python evaluate.py --out outputs_128
```

All seeded with --seed 7 (tf.keras.utils.set_random_seed) so split,
initialization, and augmentation stream are reproducible. Each outputs*/
folder has model.keras, config.json, history.json, metrics.json, and 4
figures (training curves, test confusion matrix, ROC, CAM).

## Limitations

- Test set is only 624 images — one misclassification moves accuracy by 0.16
  points, so don't over-read the small differences in finding 4. AUC is the
  steadier comparison.
- The distribution shift (finding 1) is diagnosed here, not fixed. Actually
  fixing it needs training data from the same source as the test set, or some kind
  of domain adaptation. In a real deployment you'd want to recalibrate the
  threshold on local data.
- Patient IDs only come from filenames — it's the only info available. If any
  filenames were inconsistent some leakage could theoretically slip through.
- One binary label mixes bacterial (2,530) and viral (1,345) pneumonia,
  clinically pretty different things — a 3-class setup would probably be more
  useful since that info is already in the filenames.
- Not a diagnostic device. Trained on one hospital's paediatric AP films,
  tested on 624 images. Triage aid at best — the actual call belongs to a
  clinician, not an argmax.