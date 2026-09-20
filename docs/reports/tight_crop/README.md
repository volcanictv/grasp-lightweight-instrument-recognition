# Tight instrument crops with a background buffer: fold1 result

Branch `exp/tight-crop`. Verdict: on the standard evaluation the tight-band
crop is **worse**, not better, and it does not learn faster. The new
augmentation (random masking + lighting diversity) is a small, noisy gain on
the existing crop. Fold2 and the official confirmatory run were deliberately
skipped (see "Deliberate skip").

## Hypothesis and interpretation used

Hypothesis: training on crops that hold the instrument plus a buffer of real
background, with the instrument the majority of pixels, one instrument per
crop, plus random masking and lighting augmentation, helps the model learn to
identify instrument pixels faster or more accurately.

- **Training crop ("tight band").** Ground-truth instance mask, plus the
  nearest background pixels until they number at most 0.5 x the instrument's own
  pixel count, never farther than 24 native pixels from the mask. Pixels of any
  other annotated instrument in the frame are removed from the band, so each
  crop shows exactly one instrument. Everything outside instrument + band is
  black (the same convention as the existing crops). The kept region is
  cropped to its bounding box, padded to a square, resized to 224.
- **Standard crop (baseline).** The project's existing mask-only letterbox crop
  (`letterbox_crop: true`): instrument pixels only, background black.
- **Augmentation.** `default` is the project's recipe. `masklight` adds, after
  ToTensor and before Normalize: lighting diversity (gain 0.6-1.4, gamma
  0.7-1.5, a smooth illumination gradient, small per-channel gains, each applied
  with p=0.8, all multiplicative so a black background stays black) and random
  masking (p=0.5, 1-3 rectangles of 2-15% of the crop set to black). No vertical
  flips, large rotations or hue shifts (CLAUDE.md limits).
- **Test.** Primary: the project's standard evaluation, unchanged (mask-only
  letterbox crops of the validation split, default eval transform), so numbers
  compare with every earlier Task B result. Secondary, as a robustness check:
  raw untouched pixels inside the annotated box (no mask, no buffer), and the
  tight-band view as a matched-distribution reference.
- **Design.** 2x2 of {standard, tight band} x {default, masklight}, MobileNetV3-small at
  224, identical seed, epochs (20), batch 32, lr 0.001, backbone lr 0.0001 and
  class-weighted cross entropy as `configs/region_letterbox_crop.yaml`. `data.split:
  fold1` (train on the fold2 annotation file, validate on fold1), never the
  official split. 3 seeds (42, 43, 44) per condition, 12 runs. The
  seed-42 standard baseline reproduces the earlier run of the same config
  (0.8134 vs 0.8161). Every checkpoint's standard-view score reproduces its
  manifest to 1e-3.

## Crop audit (all 2,935 fold1-training instances)

`audit.json`, `preview.png` (standard | tight band | tight band after masklight).

- The instrument outnumbers the band for every instance (asserted in the audit
  and unit-tested). Band pixels / instrument pixels: mean 0.38, median 0.38, p10
  0.24, p90 0.50. No empty masks and no instance without a band.
- The 24 px cap binds for 2,074 of 2,935 instances (71%): for most instruments
  the band is a 24 px rim and stays under the 0.5 limit. Mean band radius 21 px.

## Results (fold1, mean +/- SD over 3 seeds)

| crop | aug | best-epoch F1 | final F1 (mean of last 3 epochs) | best-epoch acc | Grasper F1 | Suction F1 |
|---|---|---|---|---|---|---|
| standard | default | 0.822 +/- 0.009 | 0.801 +/- 0.013 | 0.878 +/- 0.005 | 0.546 +/- 0.032 | 0.928 +/- 0.008 |
| standard | masklight | 0.833 +/- 0.006 | 0.808 +/- 0.021 | 0.884 +/- 0.008 | 0.564 +/- 0.044 | 0.934 +/- 0.007 |
| tight band | default | 0.780 +/- 0.015 | 0.745 +/- 0.028 | 0.839 +/- 0.012 | 0.495 +/- 0.021 | 0.896 +/- 0.011 |
| tight band | masklight | 0.794 +/- 0.033 | 0.759 +/- 0.015 | 0.846 +/- 0.023 | 0.509 +/- 0.052 | 0.915 +/- 0.014 |

Best-epoch F1 is the maximum over epochs of validation macro-F1 and is
optimistic (selected on the validation split), equally for every condition.
Final F1 involves no selection.

### Learning speed (`learning_curves.png`)

| crop | aug | mean F1 over all 20 epochs | epochs to reach 90% of the baseline's final F1 |
|---|---|---|---|
| standard | default | 0.760 +/- 0.009 | 3.0 +/- 0.0 |
| standard | masklight | 0.756 +/- 0.008 | 4.7 +/- 1.2 |
| tight band | default | 0.708 +/- 0.008 | 7.0 +/- 1.7 |
| tight band | masklight | 0.710 +/- 0.024 | 6.7 +/- 2.1 |

The target is 0.9 x that seed's own standard+default final F1. The tight-band
curves sit below the standard curves at essentially every epoch after epoch 2
and reach the target later.

### Seed-level noise (paired by seed)

Differences vs standard+default, per seed (42, 43, 44):

| comparison | final F1 | best-epoch F1 | curve mean |
|---|---|---|---|
| tight band, default aug | -0.079, -0.039, -0.050 | -0.049, -0.042, -0.036 | -0.049, -0.052, -0.055 |
| tight band, masklight | -0.031, -0.053, -0.042 | -0.016, -0.062, -0.007 | -0.041, -0.068, -0.041 |
| standard, masklight | +0.035, -0.011, -0.006 | +0.025, +0.013, -0.005 | +0.005, -0.002, -0.012 |

The tight-band deficit is negative in all 3 of 3 seeds on every metric and is
several times the seed SD. The masklight gain on the standard crop is +0.006
(final) to +0.011 (best) with per-seed SD 0.015-0.025 and mixed signs, which is
not distinguishable from zero with 3 seeds. Masklight on the tight band is
+0.014 (final), 2 of 3 seeds positive, also inconclusive.

### Secondary evaluation: other views of the validation split (`variant_eval.json`)

Macro-F1, mean +/- SD over 3 seeds.

| trained on | standard view (primary) | raw bbox, no mask, no buffer | tight-band view |
|---|---|---|---|
| standard, default | 0.822 +/- 0.009 | 0.367 +/- 0.041 | 0.784 +/- 0.016 |
| standard, masklight | 0.833 +/- 0.006 | 0.381 +/- 0.026 | 0.778 +/- 0.009 |
| tight band, default | 0.780 +/- 0.015 | 0.483 +/- 0.020 | 0.808 +/- 0.015 |
| tight band, masklight | 0.794 +/- 0.033 | 0.495 +/- 0.021 | 0.835 +/- 0.008 |

## Verdict on the hypothesis

Not supported on the primary metric. Training on tight-band crops lowers
fold1 macro-F1 by about 0.04-0.06 (final and best epoch), lowers accuracy
by about 0.04, hurts both hard classes (Grasper 0.55 -> 0.50, Suction 0.93 ->
0.90), and reaches 90% of the baseline F1 about 4 epochs later. Confidence that the tight band is not better than the
existing crop under this evaluation: high (3 of 3 seeds, deficit several times
the noise). "Learns instrument pixels faster" is not visible: the curves are
slower, not faster.

The masking + lighting augmentation is a possible small gain on the existing
crop (+0.006 to +0.011 F1, +0.006 accuracy), but 3 seeds cannot separate it
from noise, and it does not speed learning (curve mean 0.756 vs 0.760).

One result points the other way and should not be over-read: tight-band
training is far better on raw, unmasked boxes (0.48-0.50 vs 0.37-0.38), as
expected since those models saw real background. All views on raw pixels are
still far below the mask-based scores, so raw-box inputs are a poor operating
point for every model here.

## Caveats

- **Train/test mismatch.** The standard evaluation feeds mask-only crops,
  while the tight-band models trained with a background rim. Part of their
  deficit may be this distribution shift, not a worse training signal. The
  matched-view column shows the shift is real (the tight-band models gain
  +0.03 to +0.06 on their own view), but scores on different views are not
  comparable to each other, so this experiment cannot separate the two
  explanations. Both crops need the ground-truth mask at inference.
- One backbone (MobileNetV3-small), one band setting (0.5, 24 px, binding for
  71% of instances), one dataset fold, 3 seeds. Larger bands or a resnet
  backbone were not tried.
- "Original test sequences with no alteration" was interpreted as the
  project's standard evaluation, with raw boxes as the robustness check.

## Deliberate skip

Per the stop-if-not-promising design, fold2 and the single official confirmatory
run were **not run**: fold1 shows the tight band is worse in every seed, so
neither is warranted. The official split was never touched.

## Files

- Code: `src/surgical_ai/data/tight_crop.py`, `src/surgical_ai/data/tight_transforms.py`,
  `scripts/train_tight_crop.py` (wrapper around `scripts/train.py`, which is
  unchanged), `scripts/preview_tight_crops.py`, `scripts/analyze_tight_crop.py`,
  `scripts/eval_tight_crop_variants.py`, `tests/test_tight_crop.py` (10 tests).
- Configs: `configs/tight_crop/` (4 conditions x seeds 42-44 on fold1; the fold2
  configs exist but were not run).
- Results: `results.json` and `summary.md` (per-run and per-condition),
  `variant_eval.json`, `audit.json`, `learning_curves.png`, `preview.png`.
- Runs, checkpoints and logs live on titanxp in `~/Desktop/tight_crop_exp`
  (not in git).

## Deviations from the brief

3 seeds for all four conditions (not only the headline pair); fold2 and the
official run skipped as above; the augmentation is added on top of the default
recipe rather than replacing it.
