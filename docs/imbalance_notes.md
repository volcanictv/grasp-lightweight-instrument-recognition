# Imbalance and data-quality literature notes

Companion to `docs/dataset_report.md`. That doc establishes what's wrong with
the data (class imbalance, few-shot rare class, case-count confounding,
test-set concentration, bbox overlap). This doc covers what the literature
does about it, and whether each technique is worth adopting here. Most of
this is standard long-tail-recognition material, not novel — said plainly
per CLAUDE.md, not dressed up.

## What the GraSP authors themselves did

Checked directly against arXiv 2401.11174 (Ayobi et al., TAPIS). They use
**no class-balancing technique** for instrument recognition or segmentation —
no weighted loss, no balanced sampling, no oversampling — despite showing the
same long-tailed distribution we found. They report a single aggregate
mAP@0.5IoU_segm (89.85%) with no per-class breakdown, so there is no
published Clip Applier number to benchmark against. That gap is ours to
report, not theirs to compare to.

TAPIS's classification head cross-attends per-instance segment embeddings
against spatio-temporal video features, then applies a linear classifier —
a genuinely heavy transformer stage. Confirms it as the correct "heavy
baseline" for the tradeoff curve (PROJECT_SPEC.md §10), not just a
stand-in.

**Update, 2026-08-31:** this was checked again and no longer holds as
stated. Third-party work does exist now — LACOSTE (Zhou et al., arXiv
2409.09360, stereo+temporal surgical instrument segmentation) and a 2026
foundation-model paper (ZEN) both benchmark on GraSP, and LACOSTE's results
table gives TAPIS numbers (AP50_segm 90.34-91.71 across two variants) that
don't exactly match the 89.85% this doc originally cited from the paper
directly — likely a different reported variant/split, not investigated
further. But the correction that matters for our numbers specifically:
every third-party GraSP benchmark found (LACOSTE, ZEN, a referring-motion-
segmentation paper) evaluates **instance/semantic segmentation** (AP@IoU,
Dice, mcIoU, Hausdorff distance) — none evaluate pure instrument
classification (mAP or F1 given a frame or a known instance) the way this
project does. So "nothing to replicate against for our specific numbers"
still holds, just not for the reason originally stated (it's not that no
one else uses GraSP — it's that no one else poses the same classification-
only task). Full detail and sources in `docs/findings.md`.

## Problem 1: severe + few-shot imbalance

Clip Applier: 64 train instances, 25.6x gap vs. the most common class.

| Technique | Citation | Cost | Fit |
|---|---|---|---|
| Weighted BCE (inverse frequency) | standard; already in `configs` schema as `loss.class_weights` | training-time only | Do this first. Simplest, cleanly isolatable, exactly the "baseline + weighted loss" ablation row already planned. |
| Class-balanced loss (effective number of samples) | Cui, Jia, Lin, Song, Belongie, *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019 | training-time only | Generic CV, not surgical-specific, but handles near-duplicate/low-diversity samples better than naive inverse frequency — relevant given our case-concentration problem. Worth a second weighted-loss variant, not a separate ablation axis. |
| Focal loss | Urrea, Garcia-Garcia, Kern, *Improving Surgical Scene Semantic Segmentation through a Deep Learning Architecture with Attention to Class Imbalance*, Biomedicines 2024 (CholecSeg8K) | training-time only | Applied to surgical imagery specifically. A third loss-shape option — only worth adding if weighted loss and weighted sampler both underperform. Don't add speculatively; violates one-variable-per-run if bundled in early. |
| Synthetic rare-instrument compositing | Zhao et al., *Rethinking Data Imbalance in Class Incremental Surgical Instrument Segmentation*, Medical Image Analysis 105, 2025 | training-time only, but real engineering cost | Built for class-incremental segmentation, not our classification task. Overkill for now — revisit only if the cheap interventions plateau. |

## Problem 2: small case count (13 total) → case-identity confounding

No training technique fixes this — it's a methodology question, not an
algorithmic one.

Bradshaw, Huemann, Hu, Rahmim, *A Guide to Cross-Validation for Artificial
Intelligence in Medical Imaging*, Radiology: AI, 2023, documents accuracy
inflated by up to 41% when patient identity leaks across splits, and
recommends leave-one-patient-out CV for small cohorts. This validates that
GraSP's official case-level splits (already what we use) are the right call.
With only 4 cases per fold, expect real fold-to-fold variance, especially
for rare classes.

No cost, because there's nothing to add: report per-fold variance
explicitly instead of a single pooled CV number, and treat the
frozen-vs-fine-tuned comparison already planned (PROJECT_SPEC.md §6) as the
main regularization lever rather than reaching for exotic few-shot methods.

## Problem 3: test-set rare-class concentration

Clip Applier is 45% supplied by one test case (CASE053); Laparoscopic
Grasper is 56% supplied by one test case (CASE050) — both disproportionate
to those cases' share of test frames.

Same conclusion as Problem 2: a small held-out set, not something a training
technique corrects. The fix is reporting practice, not modeling — break out
per-case performance for these two classes alongside the pooled test
macro-F1, so a good or bad number reads as "mostly CASE053" rather than a
representative average. State it as a limitation in the results writeup
rather than papering over it.

## Problem 4: bbox overlap threatens Task B crop purity

~30% of co-occurring instrument pairs have overlapping bboxes; 10% have the
smaller box >50% covered by the other.

| Technique | Citation | Cost | Fit |
|---|---|---|---|
| Mask-based crop (multiply the instance RLE mask over the bbox before classifying, instead of a raw rectangular crop) | none needed — direct use of annotations GraSP already provides | training/eval preprocessing only for the classification-only task now; becomes a real inference-time dependency once Milestone 9's segmenter feeds this stage | The practical fix. Cheap, no new architecture, consistent with "lightweight, no novelty" mandate. Use this for Task B instead of raw bbox crops. |
| Occlusion-aware instance segmentation (BCNet) | Ke, Tai, Tang, *Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers*, CVPR 2021, arXiv:2103.12340 | adds architecture | Adjacent domain (generic instance segmentation, not surgical). Explicitly models occluder/occludee layers, which is conceptually the right idea, but it's real architectural weight for a problem the masked-crop approach solves at near-zero cost. Not worth adopting. |

## Flagged and rejected

**MixUp / CutMix.** Evidence is mixed even within medical imaging — helps
organ segmentation in *Cut to the Mix* (MICCAI 2024), but is explicitly
discouraged in pathology work where spatial/anatomical plausibility matters.
Our own augmentation constraints already ban implausible transforms
(no vertical flip, no large rotation, no hue shift) for the same reason —
CutMix's blended-frame artifacts are in the same spirit of implausibility.
Not adopting without a specific reason to revisit.

## Net recommendation for Milestone 5

Ablation order, one variable at a time, per PROJECT_SPEC.md §7's grid:

1. baseline
2. baseline + weighted loss (inverse frequency first; effective-number-of-samples as a documented variant if time allows, not a separate grid row)
3. baseline + weighted sampler
4. baseline + augmentation (within the existing constraints, nothing new needed)
5. baseline + weighted loss + augmentation

Everything else in this doc (focal loss, synthetic compositing,
occlusion-aware segmentation) stays parked unless the cheap interventions
plateau. The two concentration problems (case count, test-set skew) aren't
ablation variables — they're reporting obligations: per-fold and per-case
breakdowns alongside every pooled metric for Clip Applier and Laparoscopic
Grasper.

## Milestone 5 results (2026-08-31)

Ran rows 1-4 plus the weighted-loss+augmentation combo (results and
comparison table in `README.md`). The cheap interventions plateaued in
the expected direction, not the pessimistic one: every intervention beat
the baseline on both mean AP and macro-F1, so there's no case yet for
reaching into the parked techniques (focal loss, class-balanced loss,
synthetic compositing).

Weighted loss and weighted sampler landed within ~1 point of each other on
every metric (macro-F1 0.673 vs 0.667, Clip Applier F1 0.545 vs 0.545) —
unsurprising since both are inverse-frequency-driven and this doc already
flagged them as close cousins (Problem 1 table folds B and C together for
the same reason). Augmentation alone acted more as an overfitting fix
(flat val loss across training) than a targeted imbalance fix, and
combining it with weighted loss gave the best macro-F1 (0.708) and by far
the best Clip Applier F1 (0.735) — the two interventions attack different
parts of the problem (which class gets attended to vs. whether the model
memorizes the ~64 Clip Applier training instances) and stack productively.

Laparoscopic Grasper did not respond to any intervention (F1 stayed in
0.40-0.44 across all five runs). That's consistent with this doc's Problem 3
already predicting it: 56% of its test instances come from a single case
(CASE050), so this reads as a reporting/evaluation problem (need the
per-case breakdown before trusting the pooled number) rather than something
a loss or sampler change fixes. Not escalating to a targeted technique for
this class until that breakdown is done.
