# Error analysis: what is actually causing the errors

Prompted by a direct question, 2026-09-02: before spending more compute on
ensembling or a new segmentation architecture, look at the actual error
cases from the existing detection and classification models and find the
dominant failure mode, rather than assuming it's occlusion because that's
the mechanism the project has spent the most time on. This is a diagnostic
document, not a new benchmark -- every number here comes from re-running
inference with an already-trained checkpoint against ground truth
(read-only; nothing here retrains anything). Scripts:
`scripts/analyze_detection_errors.py`, `scripts/analyze_classification_errors.py`,
`scripts/extract_confusion_crops.py`. See `docs/DECISIONS.md`, 2026-09-02, for
the full run log.

Checkpoints used (all on the official test set, matching `docs/findings.md`'s
existing numbers, reproduced here as a correctness check before trusting the
breakdown): `detection_weighted_loss` (best detection-only mAP@50, Milestone 8),
`imbalance_weighted_loss_augmentation` (best Task A macro-F1, Milestone 5),
`region_baseline` (Task B, Milestone 7).

## Detection: the occlusion/NMS mechanism is not what's causing misses

347 of 2861 GT instances were missed (score/IoU threshold 0.5/0.5, matching
the operating point every other detection number in this project uses).
Root cause per miss:

| reason | count | % of misses |
|---|---|---|
| low confidence (right box, right class, scored under 0.5) | 126 | 36.3% |
| not proposed (no candidate box near the instance at all) | 103 | 29.7% |
| localization error (box nearby, not tight enough) | 71 | 20.5% |
| classification error (box on the instrument, wrong class) | 47 | 13.5% |
| **stolen by a higher-scoring same-class neighbor** | **0** | **0%** |

The last row is the important one. This project's working hypothesis since
Milestone 8 (`docs/detection_literature_notes.md`, the soft-NMS experiment)
has been that hard NMS discards one of two real, overlapping instruments as
a "duplicate." Soft-NMS gave a null result (mAP@50 0.826 vs. 0.829,
`docs/DECISIONS.md` 2026-08-31), which pointed away from this mechanism but
didn't rule it out directly. This decomposition does: across all 347 misses,
zero were a same-class box that existed, scored above threshold, and lost to
a neighboring instance's detection. **The mechanism this project spent the
most effort investigating (occlusion-driven NMS suppression) does not occur
in this checkpoint's errors at all.**

Stratifying by occlusion bucket (box-overlap proxy, `docs/imbalance_notes.md`
Problem 4) shows what does change under occlusion:

| reason | isolated | light | heavy |
|---|---|---|---|
| not_proposed | 44 | 16 | 43 |
| low_confidence | 65 | 32 | 29 |
| localization_error | 24 | 20 | 27 |
| classification_error (+ low-conf variant) | 22 | 14 | 11 |

For isolated misses, 42% (65/155) are pure low-confidence -- the model finds
the right box and the right class, it's just under-scored. For heavy-occlusion
misses, not_proposed + localization_error together are 64% (70/110) -- under
occlusion the candidate box itself degrades (or never forms), which is a
proposal-quality problem, not a confidence-calibration one. Two different
failure modes depending on occlusion, and neither is NMS suppression.

False positives are the bigger surprise: 61.4% (328/534) are **a correctly
located box on a real instrument, just the wrong class** -- not background
hallucination (12.0%) and not duplicates (1.5%). The confusion is
concentrated in specific pairs, not spread evenly:

| true class | predicted as | count |
|---|---|---|
| Bipolar Forceps | Prograsp Forceps | 110 |
| Prograsp Forceps | Bipolar Forceps | 39 |
| Large Needle Driver | Bipolar Forceps | 31 |
| Large Needle Driver | Monopolar Curved Scissors | 24 |
| Large Needle Driver | Prograsp Forceps | 20 |
| Monopolar Curved Scissors | Suction Instrument | 19 |

## Task A: co-occurrence hurts recall as suspected, but Suction Instrument is the actual biggest problem

Overall recall (all classes pooled) drops as more instruments co-occur in a
frame, consistent with `docs/findings.md`'s "framing problem" interpretation
of why Task B outperforms Task A on rare classes:

| instruments in frame | recall |
|---|---|
| 1 | 0.890 |
| 2 | 0.846 |
| 3 | 0.805 |
| 4 | 0.814 |

But the co-occurrence effect (roughly 8-9 points) is smaller than one
specific class's collapse: **Suction Instrument recall is 0.371 at n=2 and
0.529 at n=3+**, against instrument counts (310 test instances) that aren't
especially rare. It produces 151 false negatives -- more than double any
other class (next is Bipolar Forceps at 104) -- making it the single largest
contributor to Task A's missed detections, ahead of the two classes
(Clip Applier, Laparoscopic Grasper) that have gotten the attention so far
for being rare/hard.

## Task B: the same confusion pairs show up in an independently trained model

Task B's confusion matrix (region classifier, different architecture head,
different training run, ground-truth-cropped input) shows the same pairs
detection's false positives found:

| true class | predicted as | count | % of true class |
|---|---|---|---|
| Large Needle Driver | Bipolar Forceps | 65 | 14.5% |
| Bipolar Forceps | Prograsp Forceps | 56 | 6.9% |
| Prograsp Forceps | Bipolar Forceps | 48 | 14.5% |
| Monopolar Curved Scissors | Suction Instrument | 48 | 5.7% |
| Large Needle Driver | Prograsp Forceps | 26 | 5.8% |

Two independently trained models converging on the same specific pairs is
evidence this is a property of the visual data (or the backbone's feature
representation), not a fluke of one training run. It groups into two visual
families: forceps/grasper-type tools with a jawed tip (Bipolar Forceps,
Prograsp Forceps, Laparoscopic Grasper) and shaft-type tools with a long thin
silhouette (Large Needle Driver, Monopolar Curved Scissors, Suction
Instrument).

A second, previously unflagged finding: **error rate scales strongly with
instance crop size**, independent of class --

| crop area quartile | median area | error rate |
|---|---|---|
| smallest | 35,685 px² | 27.6% |
| 2nd | 113,096 px² | 13.3% |
| 3rd | 189,685 px² | 7.8% |
| largest | 316,096 px² | 8.7% |

A >3x gap between the smallest and largest quartile. Error concentration by
case is comparatively flat (13.3%-15.8% across the 5 test cases) -- ruling
out "one bad case is dragging the average down."

## Visual inspection: three distinct causes behind the confusion pairs, not one

`scripts/extract_confusion_crops.py` pulled actual misclassified crops (both
the mask-multiplied crop the Task B model sees and the raw bbox crop for
context) for the top pairs above. Manually inspecting them shows the
confusion is not one mechanism:

1. **Tip compressed out of the crop by extreme aspect ratio.** Several
   Bipolar-Forceps-as-Prograsp-Forceps and Large-Needle-Driver-as-Bipolar-Forceps
   examples are long diagonal shafts (e.g. a 730x788px bbox) where the actual
   distinguishing feature -- the jaw/tip shape -- occupies a small fraction of
   the crop, at one extreme corner. The dataset's fixed-size 224x224 square
   resize stretches the whole bbox including the mostly-empty background,
   which compresses exactly the region that would let the model tell the
   classes apart.
2. **Tissue occlusion, not instrument occlusion.** One
   Monopolar-Curved-Scissors-as-Suction-Instrument example's mask is split
   into two disconnected fragments; the raw crop shows why -- a mass of
   tissue/blood covers the middle of the instrument, leaving only two small,
   ambiguous slivers of shaft visible. This is a real occlusion mode this
   project's existing occlusion-stratified recall metric does not capture at
   all (it only measures instrument-vs-instrument box overlap, per
   `docs/detection_literature_notes.md`).
3. **Genuine fine-grained visual similarity at the given resolution.** Other
   examples (a Laparoscopic-Grasper-as-Suction-Instrument case, a
   Large-Needle-Driver-as-Monopolar-Curved-Scissors case) show the tip clearly,
   unoccluded, well-lit -- and it's still a plausible confusion at 224x224:
   metallic jawed tips and smooth curved shafts look similar to each other
   across these classes without a resolution or framing problem to blame.

## What this rules out and what it points to

Ruled out: hard-NMS/occlusion-driven detection suppression as a source of
missed instances (0 occurrences across 347 misses, directly measured, not
inferred). Also ruled out: a single problem case or a single mechanism
behind the confusion pairs.

Confirmed and now quantified: Task A's recall does degrade with
co-occurrence count, as `docs/findings.md` suspected, but the effect is
smaller than one class's (Suction Instrument) collapse.

New findings not previously in this project's docs:
- Detection's false positives are dominated by classification confusion on
  real, correctly-located instruments (61.4%), not background hallucination.
- The exact same confusion pairs (Bipolar/Prograsp/Grasper family,
  Needle-Driver/Scissors/Suction family) appear independently in both the
  detector and the region classifier.
- Task B error rate has a >3x gap between the smallest and largest instance
  crops.
- At least one of the three visible causes behind the confusion pairs (crop
  aspect ratio compressing the tip) is a data-preprocessing issue, not a
  capacity or architecture one -- fixable without a bigger model.

## Recommended next step

Of the three visual causes, aspect-ratio compression is the cheapest, most
mechanical one to test: change Task B's crop-to-224x224 resize to preserve
aspect ratio (pad/letterbox instead of stretch) and re-train, one variable,
against the exact same `region_baseline` config otherwise. If the top
confusion pairs shrink specifically for the long-thin-bbox cases (Large
Needle Driver, Bipolar/Prograsp shafts) while the tissue-occlusion and
genuine-similarity cases don't move, that confirms the mechanism rather than
just correlating with it. This is the investigation started immediately
following this document -- see `docs/DECISIONS.md` for the outcome.

Not pursued here because it's a bigger lift for uncertain payoff: a
tissue-occlusion-aware metric (parallel to occlusion-stratified recall but
for instrument-vs-tissue coverage) would need a way to detect tissue
coverage from RGB alone, no annotation exists for it.

## Outcome: letterbox crop confirms the mechanism, but trades against the rarest classes

`configs/region_letterbox_crop.yaml`, official split, otherwise identical to
`region_baseline.yaml`. Result: accuracy 0.857 -> **0.882** (+2.5 points),
macro-F1 0.825 -> 0.827 (flat). Full run: `docs/DECISIONS.md`, 2026-09-02.

The confusion matrix confirms the mechanism directly, not just by
correlation -- the exact pairs the aspect-ratio hypothesis predicted would
shrink, did:

| pair | before | after |
|---|---|---|
| Large Needle Driver -> Bipolar Forceps | 65 (14.5%) | 24 (5.3%) |
| Prograsp Forceps -> Bipolar Forceps | 48 (14.5%) | 28 (8.5%) |
| Monopolar Curved Scissors -> Suction Instrument | 48 (5.7%) | 22 (2.6%) |
| Bipolar Forceps -> Prograsp Forceps | 56 (6.9%) | 41 (5.1%) |

But two others got worse, and they land on the project's already-weakest
class:

| pair | before | after |
|---|---|---|
| Bipolar Forceps -> Large Needle Driver | 21 (2.6%) | 41 (5.1%) |
| Prograsp Forceps -> Laparoscopic Grasper | 12 (3.6%) | 23 (7.0%) |
| Laparoscopic Grasper -> Suction Instrument | 8 (9.9%) | 10 (12.3%) |

Per-class F1: five of seven classes improved (Bipolar Forceps +0.029,
Prograsp Forceps +0.045, Large Needle Driver +0.044, Monopolar Curved
Scissors +0.016, Suction Instrument +0.032) -- exactly the classes the
confusion-pair analysis implicated. But **Clip Applier (-0.076) and
Laparoscopic Grasper (-0.079, down to F1 0.634, the worst of any class in
either run) got worse.** Macro-F1 ends up flat because it weights all seven
classes equally, so the gains on the well-represented, previously-confused
classes are offset by losses on the two smallest classes.

**Honest read**: this is a real, confirmed, single-variable causal result
(the crop-aspect-ratio mechanism reduces exactly the pairs it should), not
a wash. But it is not a clean win to adopt as the new baseline without
qualification -- it shifts the error distribution rather than removing it,
and the class it makes worse (Laparoscopic Grasper) was already this
project's hardest class across every prior milestone
(`docs/findings.md`, Task A section). Both classes it hurt are also the two
smallest in the dataset (Clip Applier, Laparoscopic Grasper), which fits a
plausible mechanism: letterbox padding shrinks the instrument's effective
in-canvas resolution for already-elongated crops, and there isn't enough
training data in these two rare classes for the model to compensate the way
it can for the well-represented classes. Not confirmed further (would need
a per-class effective-resolution measurement to verify rather than infer).

Next honest step, not yet done: check whether combining `letterbox_crop`
with the existing `class_weights: true` (already on in this config) needs a
stronger rare-class weight specifically to recover Laparoscopic Grasper and
Clip Applier, or whether letterboxing should be applied selectively (only to
crops above some aspect-ratio threshold, sparing already-square-ish rare
class crops from the padding/resolution cost). Not implemented -- a decision
point, not a default next action.

## Negative result: focal loss for Task A's Suction Instrument calibration problem

A second, parallel diagnostic (same session) found Task A's Suction
Instrument misses are mostly borderline, not confidently wrong -- mean
predicted probability 0.507 when the class is actually present (only 14% of
misses scored under 0.1). That specific pattern is exactly what focal loss
(Lin et al. 2017) is built to sharpen, and it hadn't been tried anywhere in
this project (Milestone 5 tried weighted loss, weighted sampler, and
augmentation, not a loss-shape change). Tested as `configs/imbalance_focal_loss.yaml`
(`loss.type: focal_bce`, gamma=2.0) -- one variable against
`imbalance_weighted_loss_augmentation.yaml`, same pos_weight mechanism kept
so focal loss is a strict addition, not a replacement.

**Result: worse across the board, including on the target class.**

| class | baseline F1 | focal-loss F1 |
|---|---|---|
| Bipolar Forceps | 0.894 | 0.899 |
| Prograsp Forceps | 0.569 | 0.552 |
| Large Needle Driver | 0.859 | 0.838 |
| Monopolar Curved Scissors | 0.941 | 0.944 |
| **Suction Instrument** | **0.530** | **0.495** |
| Clip Applier | 0.735 | 0.677 |
| Laparoscopic Grasper | 0.425 | 0.333 |
| **macro-F1 / mean AP** | **0.708 / 0.745** | **0.677 / 0.736** |

Suction Instrument's recall did move in the intended direction (0.531 ->
0.553), but precision collapsed (0.529 -> 0.448), so F1 net *dropped*.
Clip Applier and Laparoscopic Grasper -- the two smallest classes, already
carrying the largest `pos_weight` values -- got substantially worse
(Laparoscopic Grasper F1 0.425 -> 0.333, the worst score recorded for that
class in this project). Val macro-F1 was also visibly noisier epoch-to-epoch
than the baseline run.

**Likely mechanism, not fully confirmed**: focal loss's hard-example
up-weighting compounds with `pos_weight`'s already-large inverse-frequency
multiplier for rare classes (both push the same direction -- more gradient
on rare/hard positives). For classes with very few training instances (Clip
Applier, Laparoscopic Grasper), that combination likely concentrates
gradient on a handful of genuinely ambiguous or borderline examples rather
than sharpening a real decision boundary, destabilizing training on exactly
the classes with the least data to absorb it. Not confirmed by a separate
ablation (e.g. focal loss without `pos_weight`, or a lower gamma) -- this is
the plausible explanation, not a verified one.

**Decision: not adopted.** Closes this specific lever for Task A's
calibration problem; a per-class decision threshold (chosen from a
validation split, not touched here) is the more promising untried
alternative for a borderline-calibration problem like Suction Instrument's,
since it doesn't touch training dynamics or interact with `pos_weight` at
all.

## Follow-up: per-class threshold, small isolated win for Suction Instrument, unreliable for the rarest classes

`scripts/tune_per_class_thresholds.py`. Tuned per-class thresholds on the
TRAIN split's own predictions (in-sample -- this checkpoint used
`data.split: official`, so no leakage-free held-out split exists for it
specifically), applied to official test, compared against an oracle
(thresholds tuned directly on test, reference only, never reportable as an
achievable result).

Tuning all 7 thresholds at once is net negative (macro-F1 0.701 -> 0.690,
vs. an oracle of 0.731 -- real headroom exists but in-sample tuning doesn't
reliably capture it): Bipolar Forceps and Prograsp Forceps improved close
to their oracle values, but Clip Applier (0.706 -> 0.596) and Laparoscopic
Grasper (0.408 -> 0.383) got worse, landing on extreme train-optimal
thresholds (0.94, 0.89) that read as overfit to a handful of positive
training examples -- the same two classes every other rare-class
intervention this session has hurt.

Applied only to Suction Instrument (threshold 0.41, every other class left
at 0.5): macro-F1 0.701 -> **0.703**, Suction Instrument F1 0.532 -> **0.542**,
zero effect elsewhere by construction. Modest, but the first genuinely
non-negative result this session for this specific problem, and the only
one that didn't cost something else to get it. Adopted for Suction
Instrument only; the other six classes' in-sample thresholds are not
trusted given the overfitting evidence. Full numbers: `docs/DECISIONS.md`,
2026-09-02.

## Negative result: combining selective letterbox + sharper class weights made things worse, not better

Direct follow-up attempt to recover Clip Applier/Laparoscopic Grasper
without losing the letterbox fix's gain: `configs/region_letterbox_selective_weighted.yaml`
combines `letterbox_min_aspect: 2.0` (only pad crops with long:short ratio
>=2:1, ~33% of test crops instead of 100%) with `class_weight_power: 1.5`
(sharpens the balanced class-weight formula beyond the default). Run in
parallel with the focal-loss experiment above.

Result: **worse than the unconditional letterbox run on almost every
metric, including the two classes it specifically targeted.**

| class | region_baseline | region_letterbox_crop (unconditional) | selective + sharper weights |
|---|---|---|---|
| Bipolar Forceps | 0.859 | 0.888 | 0.862 |
| Prograsp Forceps | 0.727 | 0.772 | 0.725 |
| Large Needle Driver | 0.830 | 0.874 | 0.824 |
| Monopolar Curved Scissors | 0.939 | 0.955 | 0.928 |
| **Clip Applier** | 0.861 | 0.785 | **0.775** |
| **Laparoscopic Grasper** | 0.713 | 0.634 | **0.615** |
| macro-F1 / accuracy | 0.825 / 0.857 | 0.827 / 0.882 | **0.801 / 0.853** |

Laparoscopic Grasper's recall did rise (0.716 -> 0.728) but precision fell
further (0.569 -> 0.532), so F1 dropped again rather than recovering --
**the same recall-up/precision-down/F1-net-negative signature as the focal
loss result above**, now seen twice from two different rare-class-weight-
boosting mechanisms (a loss reshaping and a weight-formula exponent) on two
different tasks. Restricting letterbox's scope also gave back most of its
gain on the classes it was fixing (all four confused classes moved back
toward baseline). Net effect: worse than either single-variable ablation it
was meant to combine.

**Decision: not adopted.** Aggressively up-weighting a class this project
has already tried twice (here and with focal loss) consistently trades
precision for recall without a net F1 gain -- this looks like a real,
repeatable pattern for GraSP's rarest classes (too few training examples to
support a sharper decision boundary), not noise from one run. Stronger
weighting is not a promising direction for Clip Applier/Laparoscopic
Grasper specifically; see the ensemble result below for what actually
worked instead.

## What actually worked: ensembling the two Task B checkpoints, no retraining

Simplest idea on the table, tried last: average the softmax outputs of
`region_baseline` (plain stretch) and `region_letterbox_crop` (pad-to-square)
-- the same "architecture/preprocessing diversity beats single-model tuning"
pattern already validated for the segmentation ensemble
(`docs/DECISIONS.md`, four-way Mask R-CNN + SAM2 ensemble), tested here for
Task B instead of assumed to transfer. `scripts/evaluate_region_ensemble.py`,
no training, ~2 minutes of inference.

| class | region_baseline | region_letterbox_crop | **ensemble** |
|---|---|---|---|
| Bipolar Forceps | 0.859 | 0.888 | **0.890** |
| Prograsp Forceps | 0.727 | 0.772 | **0.775** |
| Large Needle Driver | 0.830 | 0.874 | 0.874 |
| Monopolar Curved Scissors | 0.939 | 0.955 | **0.961** |
| Suction Instrument | 0.847 | 0.879 | **0.890** |
| Clip Applier | 0.861 | 0.785 | 0.846 |
| Laparoscopic Grasper | 0.713 | 0.634 | 0.701 |
| **macro-F1 / accuracy** | 0.825 / 0.857 | 0.827 / 0.882 | **0.848 / 0.889** |

Beats both individual models on five of seven classes outright, and on the
two it doesn't outright beat (Large Needle Driver ties Model B; Clip
Applier/Laparoscopic Grasper land between the two individual models rather
than below both), recovering nearly all of what the letterbox model cost on
the two classes it hurt while keeping nearly all of its gain on the classes
it fixed. **Best Task B result in the project**, both by macro-F1 and
accuracy, and it cost no additional training -- the two attempts to fix the
letterbox trade-off by changing training (stronger weights, selective
scope) both made it worse; combining the two *already-trained* models at
inference time is what worked. Reproduction:

```
python scripts/evaluate_region_ensemble.py \
    --checkpoint-a experiments/region_baseline_20260831-182451/best.pt --letterbox-a false \
    --checkpoint-b experiments/region_letterbox_crop_20260902-152750/best.pt --letterbox-b true
```

**Latency, measured (Titan Xp, single image, warmed up, synchronized,
median/p95 over 200 runs -- CLAUDE.md's standard benchmark protocol)**: an
accuracy win reported without its latency cost is incomplete per this
project's own rule, and this ensemble hadn't been benchmarked yet.

| | Model A alone | Model B alone | Combined (sequential) |
|---|---|---|---|
| Titan Xp median / p95 | 4.156 / 4.227 ms | 4.237 / 4.308 ms | **8.302 / 8.415 ms** |
| ONNX CPU median | 1.432 ms | 1.420 ms | **2.853 ms** |
| Params | 1.525M | 1.525M | 3.050M |
| Model size | 5.94 MB | 5.93 MB | 11.87 MB |
| Peak VRAM | -- | -- | 22.5 MB |

Combined latency is within 0.1ms of the naive sum of the two individual
medians (8.302ms measured vs. 8.393ms summed) -- no meaningful overhead
from running both sequentially, and peak VRAM barely rises above a single
model's (22.5MB either way, both models are ~6MB). At 8.3ms/frame the
ensemble is still far under any real-time budget relevant to this project
(the 33ms/30fps figure used elsewhere) -- unlike the segmentation
ensemble's 768ms, doubling a lightweight classifier costs almost nothing.

## Full end-to-end pipeline: detect -> classify -> segment, wired together for the first time

Every number in this project up to this point evaluates one stage in
isolation -- Task B's 0.889 accuracy uses oracle ground-truth boxes/masks,
the Mask R-CNN's own class head has the same confusion issues as the box
detector (above), and no prior benchmark measured the actual cost of
running detection, classification, and segmentation as one pipeline on a
real image. Built directly on request: `scripts/evaluate_end_to_end_pipeline.py`.

Pipeline: the official box-then-mask Mask R-CNN checkpoint
(`instance_segmentation_maskrcnn_official`, AP50_segm 0.8101) gives boxes
+ masks in one forward pass; its own class head is discarded and replaced
by the Task B ensemble (region_baseline + region_letterbox_crop) applied
to each detected box+mask crop, since the ensemble is the more accurate
classifier per this document's own findings. No new training -- this
wires together checkpoints that already existed.

**Latency, measured on 100 real official-test frames (native 800x1280,
Titan Xp, warmed up, synchronized)** -- not a fixed dummy tensor, since
instance count varies frame to frame and that variability is a real cost
of running this pipeline:

| stage | median | p95 |
|---|---|---|
| Detect + segment (Mask R-CNN forward pass) | 30.63ms | 33.82ms |
| Classification (all instances in frame) | 36.74ms | 75.65ms |
| **Total, per frame** | **67.26ms** | **108.87ms** |

Mean 2.59 instances/frame (1-6 observed), ~17.3ms per instance for
classification. **This pipeline is not real-time as constructed** -- 67ms
median is roughly 2x the 33ms/30fps budget used elsewhere in this project,
and p95 is over 3x. Two components of that cost are notably higher than
what summing this project's previously-reported numbers would predict:

1. **Classification's real per-instance cost (~17.3ms) is roughly 2x the
   previously-benchmarked ensemble latency (8.3ms combined, docs/DECISIONS.md).**
   That 8.3ms figure timed two GPU forward passes on a tensor already
   resident on the GPU -- no PIL image construction, no transform pipeline,
   no per-instance (non-batched) CPU-to-GPU transfer. This end-to-end run
   includes all of that, and it roughly doubles the real cost. The 8.3ms
   number was correct for what it measured (pure GPU compute); it was
   never a deployment-latency claim on its own, and this is the number
   that actually reflects the cost of using the ensemble in a real
   pipeline.
2. **The Mask R-CNN forward pass measured here (30.63ms) does not match
   this project's previously-documented figure for the same architecture
   family (84.8ms, the four-way segmentation ensemble's latency table,
   docs/DECISIONS.md).** Re-benchmarked fresh, directly, on three related
   checkpoints (`instance_segmentation_maskrcnn_official`,
   `..._official_scratch`, and the original `instance_segmentation_maskrcnn`
   run) using the project's own `benchmark_detection_gpu_latency` helper at
   its documented settings (warmup=50, runs=200): all three measure
   27-30ms, not 84.8ms. **This is an open, unresolved discrepancy, not
   something this document is asserting an explanation for** -- possible
   causes include a different checkpoint/config being the true source of
   the 84.8ms figure, GPU contention at the time of that original
   measurement (it was taken during an extended multi-experiment session),
   or a difference in the exact benchmarking invocation. Flagged honestly
   rather than silently overwritten; if the ensemble latency table needs
   correcting, that should be a deliberate re-check, not an inference from
   this script's incidental finding.

**Accuracy: classification on real detected instances vs. oracle boxes.**
Matched 219 detections to ground truth (IoU >= 0.5, score >= 0.5) across
the same 100 frames and checked the ensemble's classification accuracy on
those real, non-oracle boxes/masks: **0.8493**, against Task B's own
oracle-box ensemble accuracy of **0.889** -- a real, quantified ~4-point
cost of classifying from the detector's actual (imperfect) localization
instead of ground-truth boxes. This is the first number in this project
that measures "how accurate is the classifier when fed what the detector
actually gives it," rather than what it's given a perfect box.

**Honest summary**: the components exist and can be wired together today
without new training, but as a naive sequential pipeline it is
~2-3x over a real-time budget, and its real classification accuracy is a
few points below the oracle-box number reported everywhere else in this
document. Both the latency-discrepancy and the oracle-vs-real accuracy gap
are exactly the kind of thing that only shows up once stages are actually
connected -- neither would be visible from the individual-stage numbers
this project had before.

### Same pipeline, the project's actual best segmentation result (4-way ensemble)

The run above used the single official MobileNetV3 Mask R-CNN for
detect+segment, not this project's best available result -- flagged
directly and corrected: `scripts/evaluate_end_to_end_pipeline_full_ensemble.py`
wires the same Task B classification stage onto the actual 4-way ensemble
(3 Mask R-CNN checkpoints + fine-tuned SAM2, `weighted_fusion_merge`,
AP50_segm 0.8594) instead, discarding the ensemble's own fused class label
the same way. Measured on the same 100 real official-test frames:

| stage | median | p95 |
|---|---|---|
| Detect + segment (4-way ensemble, fused) | 604.39ms | 636.89ms |
| Classification (all instances) | 47.49ms | 90.79ms |
| **Total per frame** | **651.16ms** | **731.86ms** |

Classification accuracy on the 217 real TP-matched instances: **0.8802** --
notably higher than the single-model pipeline's 0.8493, and closer to Task
B's oracle-box accuracy (0.889). This makes sense: the 4-way ensemble's
better localization and mask quality (AP50_segm 0.8594 vs. 0.8101) gives
the classifier cleaner crops, so more of the oracle-box accuracy survives
contact with real detections.

The obvious tradeoff: ~604ms vs. ~31ms for detect+segment, ~19x slower,
almost entirely SAM2's image encoder (previously measured standalone at
~398ms/frame). This is the real accuracy/latency choice this project's
components currently offer for a full pipeline -- not a hypothetical one:
0.8493 accuracy at 67ms/frame (~15fps-equivalent workload, still 2x over
the 33ms budget) vs. 0.8802 accuracy at 651ms/frame (~1.5fps-equivalent,
~20x over budget). Neither is real-time yet; closing that gap is squarely
Milestone 10's (deferred efficiency phase) territory, not something to
solve by picking a bigger model.

## Negative result: tip-crop is worse than both the plain crop and the letterbox crop

`configs/region_tip_crop.yaml` (`crop_mode: tip`, `tip_crop_frac: 0.45`) --
described above, visually validated before training (correctly located the
tip on known hard examples, including the diagonally-held one whose
axis-aligned bbox is misleadingly close to square). One variable against
`region_baseline.yaml` (`crop_mode: bbox` there).

| class | region_baseline | region_letterbox_crop | **region_tip_crop** |
|---|---|---|---|
| Bipolar Forceps | 0.859 | 0.888 | 0.834 |
| Prograsp Forceps | 0.727 | 0.772 | 0.691 |
| Large Needle Driver | 0.830 | 0.874 | 0.822 |
| Monopolar Curved Scissors | 0.939 | 0.955 | 0.935 |
| Suction Instrument | 0.847 | 0.879 | 0.872 |
| Clip Applier | 0.861 | 0.785 | **0.615** |
| Laparoscopic Grasper | 0.713 | 0.634 | **0.437** |
| **macro-F1 / accuracy** | 0.825 / 0.857 | 0.827 / 0.882 | **0.744 / 0.831** |

Worse than *both* other variants on every single class except Suction
Instrument (where it's within noise of letterbox). Laparoscopic Grasper's
F1 (0.437) is the single worst Task B result recorded in this project --
this class had been Task B's strongest result relative to Task A (F1 0.713
in the original baseline, `docs/findings.md`'s "single most important
finding" section) before this run.

Training curves point at overfitting, not underfitting: train macro-F1
reached 0.973 by epoch 20 while val macro-F1 oscillated noisily in the
0.64-0.74 band with no clear late-epoch improvement, and val loss (up to
1.1+) was consistently higher and noisier than the letterbox run's.

**Likely mechanism, not confirmed**: the crop was designed to fix one cause
(the tip's pixels being a small fraction of an elongated crop), but
discarding the shaft/wrist region also throws away context the model
apparently relies on -- shaft diameter, wrist-joint articulation style,
color transitions -- for classes where that context matters more than a
tighter view of the tip. The effect lands hardest on Clip Applier and
Laparoscopic Grasper, which are also the two smallest classes with the
least training data to compensate for a harder, more information-sparse
crop -- the same two classes every other rare-class intervention this
session has hurt, which suggests the mechanism here may be more "less
signal, less data to compensate" than something specific to tip-cropping
per se. Not verified against a less aggressive `tip_crop_frac` (a bigger
window that keeps more shaft context) -- that's the natural next variant if
tip-cropping is revisited, not attempted here.

**Decision: not adopted.** Closes the tip-crop idea as tested. The Task B
ensemble above (region_baseline + region_letterbox_crop, macro-F1 0.848)
remains the best available result; tip-crop's checkpoint was not added as a
third ensemble member since its standalone performance is below both
existing members on most classes, unlike SAM2's addition to the
segmentation ensemble (weaker overall but still additive) -- worth checking
empirically rather than assuming, if a third Task B member is wanted later.
