# Classifier training experiments 2 and 3: plan fixed before any result

Written 2026-10-01, before any model for these experiments was trained.

Motivation (official-test error breakdown of the EdgeTAM pipeline, 211 errors): 77 never tracked (confident wrong), 85
tracked but already wrong, 49 tracked and broken by the tracker. The classifier is trained on ground-truth masks but
applied to tracker masks at test time. Both experiments change training only; no gate, fusion or decision rule is touched.

## Experiments

| arm | change to the training crops (training split only) |
|---|---|
| P | with probability 0.5 the annotated mask is replaced by a corrupted copy: dilate or erode by 2 to 8% of the instrument size, or cut 5 to 15% off one end along its major axis (`perturb_mask`) |
| N | with probability 0.5 the sample is replaced by a stored EdgeTAM crop of the same physical instrument 1 to 5 frames away (1 Hz), from the same training cases (`scripts/build_temporal_neighbors.py`); a neighbour is dropped when its mask is empty or its area leaves [1/3, 3] of the annotated area |
| P+N | both, only run if P and N each help |

Everything else is identical to the baseline: the three members resnet50_320, baseline (MobileNetV3-small, stretched crop),
letterbox_crop (MobileNetV3-small), the evidential loss (lambda 0.01, KL anneal 10 of 20 epochs), optimiser, epochs, class
weights, image sizes, seeds 42, 43, 44. The baseline is the already-trained fold1 evidential ensembles (seeds 42, 43, 44).

## Protocol

1. Development on fold1: members trained on the fold2 cases, evaluated on the fold1 cases. Neighbour crops come from the
   fold2 cases only.
2. Confirmation on fold2: the same arms trained on the fold1 cases, evaluated on the fold2 cases, seed 42 against the
   existing baseline. An arm that wins on fold1 but not on fold2 is reported as not confirmed.
3. The official test is touched once, at the end, with the single winning configuration trained on the 8 official
   training cases, in the same pipeline as tracker A (3-member evidential, EdgeTAM, causal 20 frames, gate 2.85e-4). The
   gate, look-back and tracker settings are not changed.

## Metrics and decision rule

Primary: three-member ensemble alone, accuracy and macro-F1 on the held-out fold, mean over 3 seeds, with the seed
spread. Secondary: accuracy of the three-member ensemble (weights 0.40 / 0.30 / 0.30, evidential combine) on every EdgeTAM neighbour
crop (1 to 5 frames either side, same area guard) of the held-out fold's instruments, each crop one sample, which measures
the tracker-mask robustness directly with no gate (scripts/eval_on_neighbour_crops.py). An arm counts as an improvement only if its mean accuracy beats the baseline mean by more
than the larger of the two seed standard deviations and the gain has the same sign on fold2. Anything smaller is
reported as no resolvable effect. Every arm is reported, including negatives.

## Not done, on purpose

No change to the gate threshold, the fusion rule, the look-back, the member weights or the number of members. No selection
on the official test.

## Run order and code

scripts/overnight_arms.sh runs everything unattended: neighbour generation for the fold2 and fold1 cases (EdgeTAM, K = 5),
arms P and N on fold1 (scripts/run_training_arms.py), scoring (scripts/evaluate_training_arms.py), then the fold2
confirmation. Results land in docs/reports/training_arms/. Neighbour crops of a fold are generated from the ground-truth
masks of that fold's instruments, so the held-out fold's neighbour crops are used only for the secondary metric and
never for training.
