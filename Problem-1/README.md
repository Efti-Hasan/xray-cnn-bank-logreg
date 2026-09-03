# Problem 1: Classifying paediatric chest X-rays with a CNN

5,856 chest films from children aged one to five. Each film is either normal or
shows pneumonia. This folder holds a convolutional network that reads a film and
returns the probability of pneumonia, plus the evaluation that decides whether
that probability can be trusted.

The network is built with TensorFlow and Keras, in the shape of the standard
image-classification tutorial: stack `Conv2D` and `MaxPooling2D`, put a small head
on top, compile, fit, then draw the curves with matplotlib. Keras owns the
backward pass, so the work in this folder is not the calculus. It is the four
decisions that the framework leaves to you, and each one of them changed the
answer:

- how the validation set is carved out, which is where leakage would hide
- what the loss is told about a 2.9 to 1 class imbalance
- how the learning rate moves, which was worth more here than the architecture
- where the operating threshold comes from, and which set it is allowed to see

## Contents

| File | What it does |
| --- | --- |
| `data.py` | Finds the images, decodes and caches them, splits train and validation by patient |
| `train.py` | Builds the Keras model, fits it, saves `model.keras` and the training curves |
| `evaluate.py` | Scores the untouched `test/` folder, picks the threshold, writes `metrics.json` and the figures |
| `metrics.py` | Confusion matrix, sensitivity, specificity, ROC AUC, threshold search |
| `probe.py` | The two diagnostics behind findings 1 and 5 |
| `tests/test_split.py` | Checks that no child appears on both sides of the validation line |

## Running it

TensorFlow has no wheel for Python 3.14 yet, so this needs 3.12:

```bash
python3.12 -m venv .venv && .venv/bin/pip install tensorflow matplotlib numpy
```

```bash
.venv/bin/python tests/test_split.py
.venv/bin/python train.py --data-root ../data/Problem-1 --out outputs
.venv/bin/python evaluate.py --out outputs
.venv/bin/python probe.py --out outputs
```

Only `tensorflow`, `matplotlib` and `numpy` are needed. The JPEGs are decoded with
`tf.io.decode_image` and resized with `tf.image.resize`, so there is no OpenCV
dependency. The first run caches the decoded images as an `.npz` next to the
outputs, and later runs start straight away. About 10 seconds an epoch on a CPU at
96 pixels, 19 seconds at 128.

## The validation set is the most important decision here

The dataset ships as `train/` (5,216), `test/` (624) and `val/` (16). Sixteen
images cannot select a model: one image is 6.25% of the score, so the difference
between two architectures is indistinguishable from noise. A real validation set
has to come out of `train/`.

It cannot be cut at random, though. The filenames carry the patient ID.
`person1000_bacteria_2931.jpeg` and `person1000_virus_1681.jpeg` are two films of
the same child, and the pneumonia class has up to 30 images from one patient. A
random split would leave some of a child's films in train and the rest in
validation, so the network could score well by recognising the child instead of
the disease. `data.py` reads the ID out of the filename and splits by patient,
keeping the class balance:

```
train  4,432 images  (1,140 normal / 3,292 pneumonia)
val      784 images  (  201 normal /   583 pneumonia)
patients held out: 444.  patient overlap between the two: 0
```

`tests/test_split.py` checks that on the real folder, not just on a toy example:
every image used once, no patient on both sides, and each class holding its share.

Two smaller decisions follow from the medical context:

- **No horizontal flipping.** It is the reflex augmentation for image models and it
  is wrong here. The heart sits on the left, so a mirrored chest film is
  anatomically impossible. Flipping would teach an invariance that no real film
  shows. `RandomRotation`, `RandomTranslation` and `RandomZoom` stay, because
  patient positioning really does vary that way.
- **Class weights, not resampling.** Pneumonia outnumbers normal 2.9 to 1 in the
  training rows. Inverse-frequency weights (normal 1.944, pneumonia 0.673) go into
  the loss through `class_weight`, so every image is still seen every epoch.
  Throwing away two-thirds of the pneumonia films to even up the count would waste
  real data.

## Architecture

Four downsampling stages, then global average pooling straight into one output:

```
Rescaling(1/255)  RandomRotation  RandomTranslation  RandomZoom
Conv3x3(1->16)   BN  ReLU  MaxPool2
Conv3x3(16->32)  BN  ReLU  MaxPool2
Conv3x3(32->64)  BN  ReLU  MaxPool2
Conv3x3(64->128) BN  ReLU  MaxPool2
GlobalAveragePooling  Dropout(0.4)  Dense(1, sigmoid)
```

**98,241 parameters, 97,761 of them trainable.** Small on purpose. There are 4,432
training images, so a ResNet-sized model would carry far more weights than examples
and would memorise instead of generalise.

The augmentation lives inside the model as layers, so it runs on the training
batches and switches itself off at prediction time without any extra code. The head
is `GlobalAveragePooling` into `Dense(1)` with no hidden layer, and that is
deliberate: it makes the 128 dense weights per-channel importances, so a class
activation map is one dot product against the last feature map. The
interpretability comes out of the architecture rather than a bolted-on method.

Model selection uses **validation ROC AUC**, not accuracy. AUC needs no threshold,
so selection cannot be won by a checkpoint that happens to sit well against an
arbitrary 0.5 cut. The threshold is chosen afterwards, in `evaluate.py`, and only
on validation.

## Findings

### 1. Validation says 0.9992, the test set says 0.9646, and the gap is the result

Three configurations were trained, and §4 compares them. Findings 1, 2, 3 and 5 use
the baseline run so the numbers stay concrete.

| | Validation (784 films, patient-grouped) | Test (624 films, untouched) |
| --- | --- | --- |
| ROC AUC | **0.9992** | **0.9646** |
| Accuracy | 0.9834 | 0.8734 |
| Sensitivity | 0.9828 | 0.9897 |
| Specificity | 0.9851 | 0.6795 |
| Confusion | tn 198 fp 3 fn 10 tp 573 | tn 159 fp 75 fn 4 tp 386 |

Validation came out of `train/` with no patient overlap, so this is not leakage and
it is not plain overfitting either. A model that had only memorised would not reach
0.99 on 444 unseen patients. `train/` and `test/` simply do not hold the same kind
of films.

The prediction spreads say exactly where the difference sits
(`python probe.py --out outputs`):

| Predicted probability of pneumonia | median | 10th pct | 90th pct |
| --- | --- | --- | --- |
| Validation, NORMAL films | 0.000 | 0.000 | 0.008 |
| Test, NORMAL films | 0.139 | 0.001 | **0.982** |
| Validation, PNEUMONIA films | 1.000 | 0.935 | 1.000 |
| Test, PNEUMONIA films | 1.000 | **0.979** | 1.000 |

The pneumonia films in `test/` are picked up *more* confidently than the ones in
validation, with a 10th percentile of 0.979 against 0.935. All of the damage is in
the normal class: films that should score near zero have a 90th percentile of 0.982.
The model has not got worse at finding pneumonia. It has started calling healthy
test films sick, which is why sensitivity actually rises on test (0.9828 to 0.9897)
while specificity falls off a cliff (0.9851 to 0.6795).

Raw pixel statistics are nearly identical in the two sets (mean 122.2 and standard
deviation 59.9 on validation, 120.6 and 59.9 on test), so this is not a difference
in exposure or scanner. What changed is what a normal film looks like between the
two folders, which fits `test/` having been collected and labelled separately.
Anyone quoting only the 0.9992 would be quoting the wrong number.

### 2. The learning-rate schedule was worth more than anything in the architecture

The first version of `train.py` used plain `Adam(1e-3)` at a constant rate, no
weight decay, and stopped after 6 epochs of no improvement. It looked fine while it
trained. Validation ROC AUC reached 0.9976, which is a number nobody would question.

| Same model, same split, same seed | Val AUC | Test AUC | Test acc @0.5 | Sensitivity | Specificity | False alarms |
| --- | --- | --- | --- | --- | --- | --- |
| `Adam(1e-3)` constant, no decay | 0.9976 | 0.9216 | 0.7644 | 0.9897 | 0.3889 | 143 |
| `AdamW`, warmup then cosine, decay 1e-4 | **0.9992** | **0.9646** | **0.8734** | 0.9897 | **0.6795** | **75** |

Validation barely noticed the difference, 0.0016 of AUC on 784 images. The test set
noticed a lot: 0.043 of AUC, and 68 fewer false alarms out of 234 normal films.

The reason is visible in the epoch counts. With a constant rate, early stopping
fired at epoch 12 and the best checkpoint was epoch 6, because validation AUC was
already saturated and the weights were still bouncing around. With warmup and a
cosine decay the run went to epoch 30 with its best at 27, and the last epochs at a
small learning rate are the ones that settle the decision boundary. Nothing about
the layers changed. This is the single largest effect measured in this folder, and
it is bigger than the gap between the three architectures in §4.

The honest caveat: three things changed at once, the schedule, the weight decay and
the patience, so this is a before-and-after and not a clean ablation of the
schedule on its own.

### 3. A threshold tuned on validation does not survive the move to test

Choosing the operating point on validation under a 98% sensitivity floor gives
0.298. Applied unchanged to the test set:

| Threshold | Accuracy | Sensitivity | Specificity | Missed pneumonia | False alarms |
| --- | --- | --- | --- | --- | --- |
| 0.5 (default) | 0.8734 | 0.9897 | 0.6795 | 4 | 75 |
| 0.298 (val-tuned) | 0.8429 | 0.9923 | 0.5940 | **3** | 95 |

The threshold does move the way it was meant to. Sensitivity rises to 0.9923, three
missed cases out of 390 instead of four. It costs 20 extra false alarms, because the
normal-class scores have shifted underneath it. That is §1 in practical terms: on a
shifted set you keep the ranking and lose the calibration.

Both operating points are reported, not just the flattering one. For a screening
tool the trade is worth making. A false alarm costs a radiologist a second look,
while a missed pneumonia in a one-to-five-year-old costs a great deal more.

### 4. Three runs, and validation ranks them in the wrong order

| Run | Input | Aug | Dropout | Val AUC | **Test AUC** | Test acc @0.5 | Sens | Spec | Missed / false alarms | Epochs (best) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline `outputs` | 96² | 1.0× | 0.40 | **0.9992** | 0.9646 | 0.8734 | 0.9897 | 0.6795 | 4 / 75 | 30 (27) |
| strong-aug `outputs_aug` | 96² | 1.8× | 0.50 | 0.9959 | **0.9669** | 0.8285 | 0.9974 | 0.5470 | **1** / 106 | 21 (13) |
| higher-res `outputs_128` | 128² | 1.4× | 0.45 | 0.9985 | 0.9651 | 0.8670 | 0.9974 | 0.6496 | **1** / 82 | 30 (28) |

Weight decay was 1e-4 in all three. Two things fall out of the table.

**Validation puts them in exactly the wrong order.** It ranks the baseline first, and
on test the baseline comes last of the three. Both runs that validation liked less
cut the missed cases from four to one. The validation spread is 0.0033 of AUC and the
test spread is 0.0023, so neither is a large difference, but the ordering is reversed,
and the ordering is what a model search would act on. Once a validation score sits at
0.999 there is nothing left to select on and a search against it is measuring noise.

**More pixels did not buy more AUC.** Test AUC went from 0.9646 to 0.9651 for double
the epoch time, 19 seconds against 10. That rules out the cheapest explanation for
§1: the gap is not the model throwing away fine texture at 96×96, because giving it
back changes nothing. What the 128² run did improve is the operating point, three
fewer missed cases at 24 fewer false alarms than the strong-aug run.

These are three configurations, not a controlled sweep. Each one changes resolution,
augmentation and dropout together, so no single row isolates one knob.

The training curves make the selection problem visible. In
`outputs/fig_training.png` validation ROC AUC is above 0.99 from about epoch three
and never really moves again, while validation accuracy and sensitivity keep swinging
for another dozen epochs. Those swings are the 0.5 cut point sliding around under a
model whose ranking is already stable. That is the reason to select on AUC and decide
the threshold separately.

### 5. What the network looks at, and a warning sign in the maps

`outputs/fig_cam.png` overlays class activation maps on two confident correct calls
and the six worst mistakes. On the two correct pneumonia calls the heat sits over a
lower lung field, which is where it should be. On the six false positives it does
not: it spreads across both lungs, and in several of them it sits on the edge of the
film, along the bottom border below the diaphragm and on the top-left corner where
the radiographic laterality marker ("R") is burned in. That corner is the classic
shortcut-learning pattern, and presenting these maps as a clean bill of health would
be dishonest.

So an occlusion test was run to settle it. Replace a border band with the image's own
median grey value, which is a value the network has seen in real films, then score
again:

| Test-set input | ROC AUC | Sensitivity | Specificity | False alarms | Missed |
| --- | --- | --- | --- | --- | --- |
| original | 0.9646 | 0.9897 | 0.6795 | 75 | 4 |
| top 12% masked | 0.9572 | 0.9949 | 0.3974 | 141 | 2 |
| left and right 12% masked | 0.9271 | 0.9462 | 0.6923 | 72 | 21 |
| whole 12% border masked | 0.9111 | 0.9487 | 0.6581 | 80 | 20 |

The result is inconclusive, and that is the finding. Covering the corner that holds
the marker makes the false positives nearly twice as bad, 75 up to 141, so the marker
is not a shortcut the model uses to say "pneumonia". Covering the sides trims false
alarms a little, 75 down to 72, but turns four missed cases into 21. AUC falls under
every mask, so the outer band does carry signal the model relies on, and blanking it
is itself an input the network has never seen. The test cannot separate "reading the
marker" from "reading the outer lung fields". At 96×96 with four pooling stages the
final feature map is 6×6, so one cell of the map covers a 16×16 block of pixels,
which is large enough to hold the marker and the lung apex at the same time.

Settling it would need the marker painted out rather than blanked. Writing "the
network looks at the lungs" would have been easier and less true.

### 6. Which model to ship

**`outputs_128`, the 128×128 run.** Test ROC AUC 0.9651, sensitivity 0.9974 at the
default threshold, one missed pneumonia out of 390, and 82 false alarms out of 234
normal films.

This needs saying carefully, because "best on test" is a choice made on the test set
and so it is not a clean estimate of how the model will generalise. The AUC margin is
not the reason. All three runs sit within 0.0023 of each other, which is a handful of
images on a 624-image set. The reason is the rule the margin happens to agree with:
in paediatric screening the two mistakes are not worth the same. On missed cases the
two heavily regularised runs tie at one and the baseline is four times worse, so the
choice is between those two, and the 128² run gets there with 24 fewer false alarms.

What the number is worth, plainly: 0.9651 is a test AUC on a set that was looked at
while three models were compared, so read it as optimistic. The range 0.9646 to
0.9669 across the three runs is the fairer description of what this architecture does
on this data.

## Reproducing the numbers

```bash
.venv/bin/python train.py --out outputs
.venv/bin/python evaluate.py --out outputs
.venv/bin/python probe.py --out outputs
```

```bash
.venv/bin/python train.py --out outputs_aug --epochs 40 --aug-strength 1.8 --dropout 0.5
.venv/bin/python evaluate.py --out outputs_aug
```

```bash
.venv/bin/python train.py --out outputs_128 --img 128 --aug-strength 1.4 --dropout 0.45
.venv/bin/python evaluate.py --out outputs_128
```

All three are seeded with `--seed 7` through `tf.keras.utils.set_random_seed`, so the
split, the initialisation and the augmentation stream are reproducible. Each
`outputs*/` folder holds `model.keras`, `config.json`, `history.json`, `metrics.json`
and four figures: training curves, test confusion matrix, test ROC and the class
activation maps. The other two runs are kept because the comparison in §4 is part of
the finding.

## Limitations

- **The test set is 624 images.** One misclassification moves accuracy by 0.16 of a
  percentage point, so the differences in §4 are worth a few images each and should
  not be over-read. The AUC comparison is steadier than the accuracy comparison, and
  §6 says plainly what the shipped number is and is not evidence for.
- **The distribution shift is diagnosed, not fixed.** Fixing it properly needs
  training data from the same source as `test/`, or a domain-adaptation step.
  Augmentation only softens it. In a real deployment the right move is to recalibrate
  the threshold on data from the site where the model will run.
- **Patient IDs come from filenames.** That is the only patient information the
  dataset gives. If two children's films were named inconsistently, a little leakage
  could survive. The grouping removes all of it that can be seen.
- **One binary label, no severity or cause.** The pneumonia class mixes bacterial
  (2,530 training films) and viral (1,345) cases. These are clinically different and
  would be better as three classes, since the labels are already sitting in the
  filenames.
- **Not a diagnostic device.** It is trained on one hospital's paediatric
  anterior-posterior films and tested on 624 images. It is a triage aid at best, and
  the operating point belongs to a clinician, not to `argmax`.
