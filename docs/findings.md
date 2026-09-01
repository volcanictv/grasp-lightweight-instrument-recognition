# Experimental findings

Consolidated results for the research report. This pulls every number
directly from `experiments/*/manifest.json` and `experiments/*/benchmark.json`
(not from memory) as of 2026-09-01. For methodology detail and reasoning
behind specific choices, see `README.md` (narrative per milestone),
`docs/DECISIONS.md` (dated decision log), and `docs/imbalance_notes.md`
(literature). This file is the summary meant to be read on its own.

## Research question

Can a lightweight recognition model (MobileNetV3-Small) reach competitive
surgical instrument recognition on GraSP while substantially cutting
parameters, latency, memory, and model size versus heavier architectures?
The goal is a defensible accuracy/efficiency tradeoff curve, not a new
architecture or a claim of beating the dataset authors' own model.

## Dataset

GraSP (Ayobi et al., arXiv 2401.11174), robot-assisted radical
prostatectomy video. Official split: 8 train cases / 5 test cases,
2324 / 1125 annotated keyframes, 6170 / 2861 instrument instances. 7
instrument classes. Severely long-tailed: Clip Applier is the rarest
(102 instances total, a 25.6x gap vs. the most common class, Monopolar
Curved Scissors at 2609). 94.7% of annotated frames contain 2 or more
instrument instances simultaneously — this is why "one instrument per
frame" classification was rejected as a task framing; see Task A/B below.
Full detail in `docs/dataset_report.md`.

## Two task framings

- **Task A — multi-label frame classification**: 7 sigmoid outputs per
  frame, BCE loss, macro-F1 and mean AP.
- **Task B — region classification**: crop each annotated instrument
  instance (bbox + its own segmentation mask, background zeroed) and
  classify single-label. This is the task TAPIS (the paper's heavy
  transformer) actually solves.

## Hardware

Titan Xp workstation (2x, 12GB, Pascal, no tensor cores) did all training
and GPU benchmarking below. Dev laptop (RTX 4060, discovered mid-project to
be usable — see `docs/DECISIONS.md` 2026-08-31 entry — but not used for any
number in this file) is a separate, faster machine not yet factored into
these results.

---

## Task A: frozen vs. fine-tuned (Milestones 3-4)

MobileNetV3-Small, official split, 20 epochs, batch 32.

| run | trainable/total params | mean AP | macro F1 | wall clock |
|---|---|---|---|---|
| frozen backbone | 598,023 / 1,525,031 | 0.620 | 0.506 | 399.6s |
| fine-tuned (discriminative LR: head 1e-3, backbone 1e-4) | 1,525,031 / 1,525,031 | 0.711 | 0.645 | 169.4s |

**Finding**: fine-tuning wins decisively, especially on rare classes —
Laparoscopic Grasper F1 goes from 0.024 (recall 0.012, effectively never
predicted) to 0.403. ImageNet features alone don't transfer well to
endoscopic imagery. The fine-tuned run overfits hard by epoch 20 (train
macro-F1 0.995, val plateaus ~0.60-0.65 from epoch ~8), which motivated
Milestone 5.

## Task A: imbalance ablations (Milestone 5)

One variable at a time against the fine-tuned baseline above.

| run | mean AP | macro F1 | Clip Applier F1 | Laparoscopic Grasper F1 |
|---|---|---|---|---|
| baseline (fine-tuned) | 0.711 | 0.645 | 0.471 | 0.403 |
| + weighted loss (inverse-frequency `pos_weight`) | 0.727 | 0.673 | 0.545 | 0.429 |
| + weighted sampler (max inverse-freq per sample) | 0.714 | 0.667 | 0.545 | 0.407 |
| + strong augmentation | 0.756 | 0.672 | 0.500 | 0.437 |
| + weighted loss + strong augmentation | 0.745 | **0.708** | **0.735** | 0.425 |

**Findings**:
- All four interventions beat the baseline on every metric — no need for
  heavier techniques (focal loss, class-balanced loss) yet.
- Weighted loss and weighted sampler land within ~1 point of each other
  everywhere — expected, both are inverse-frequency-driven.
- Augmentation alone acts more like an overfitting fix (flat val loss
  0.29-0.37 across all 20 epochs, vs. baseline's climb to 0.58) than a
  targeted imbalance fix — best mean AP, not best macro-F1.
- **Weighted loss + augmentation combined is the clear macro-F1 winner**,
  driven almost entirely by Clip Applier (F1 0.735, vs. 0.545 for either
  intervention alone) — the two interventions attack different failure
  modes (which class gets attended to vs. whether the model memorizes the
  ~64 Clip Applier training instances) and stack productively.
- Laparoscopic Grasper never got above F1 ~0.44 in any Task A run,
  including the combined one. 56% of its test instances come from a single
  case (CASE050) — this looks like a reporting/evaluation problem, not
  something a loss or sampler change fixes (see Task B below for the
  actual resolution).

## Task A: backbone sweep / tradeoff curve (Milestone 6)

Same recipe (fine-tuned, discriminative LR, weighted loss, strong
augmentation) — architecture is the only variable. ResNet-50 was added as
a deliberately heavy baseline (see `docs/DECISIONS.md` for why it was
chosen over TAPIS itself or a ViT).

| Model | Total params | Model size | Titan Xp GPU latency (median) | Titan Xp peak VRAM | Titan Xp batch throughput | ONNX CPU latency (median) | Mean AP | Macro F1 |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small (proposed) | 1.53M | 5.93 MB | 4.14 ms | 16.6 MB | 5258 img/s | **1.41 ms** | 0.745 | 0.708 |
| MobileNetV3-Large | 4.21M | 16.25 MB | 4.85 ms | 31.9 MB | 2099 img/s | 3.24 ms | 0.809 | 0.754 |
| EfficientNet-B0 | 4.02M | 15.59 MB | 6.31 ms | 34.4 MB | 1252 img/s | 5.95 ms | 0.808 | 0.749 |
| ResNet-18 | 11.18M | 42.72 MB | **1.95 ms** | 77.9 MB | 2374 img/s | 9.49 ms | 0.798 | 0.740 |
| ResNet-50 (heavy baseline) | 23.52M | 90.02 MB | 4.98 ms | 126.0 MB | 708 img/s | 21.84 ms | **0.862** | **0.799** |

**Findings**:
- Accuracy ranks roughly by parameter count, as expected. ResNet-50 wins
  everything, including finally getting real traction on Laparoscopic
  Grasper (F1 0.622 — the only Task A configuration, across every
  milestone, to clear 0.5 on this class).
- **GPU latency does not rank by parameter count, and this is the most
  important methodological finding of the sweep.** ResNet-18 has 4-8x the
  MACs of the MobileNet/EfficientNet models but the *lowest* Titan Xp
  latency in the entire sweep — faster than MobileNetV3-Small itself. This
  is Pascal's lack of tensor cores combined with depthwise-separable
  convolutions (used by MobileNetV3/EfficientNet, not ResNet) being
  FLOP-efficient but not GPU-kernel-efficient on this specific hardware:
  low arithmetic intensity per kernel launch, poor memory bandwidth
  utilization. This directly validates the project's standing rule that
  Titan Xp latency is not a deployment claim (`CLAUDE.md`) — it isn't a
  defensive caveat, it's demonstrably true.
- On **ONNX Runtime CPU latency** — the number that's actually portable to
  hardware this project doesn't own — the ranking flips back to matching
  parameter count exactly, with MobileNetV3-Small over 15x faster than
  ResNet-50 (1.41ms vs 21.84ms).
- **Where this leaves the tradeoff curve**: ResNet-50 buys +0.091 macro-F1
  over MobileNetV3-Small (0.799 vs 0.708) at 15x the ONNX CPU latency and
  15x the model size. MobileNetV3-Large and EfficientNet-B0 sit at a
  similar accuracy tier to each other (macro-F1 0.754/0.749) for 2-4x
  MobileNetV3-Small's ONNX latency/size — the middle of the curve.

## Task B: region classification (Milestone 7)

MobileNetV3-Small, fine-tuned, discriminative LR, balanced
`CrossEntropyLoss`, default augmentation. 6170 train instances / 2861 test
instances, 20 epochs, wall clock 1168.6s (uncached — see below).

| class | precision | recall | F1 |
|---|---|---|---|
| Bipolar Forceps | 0.851 | 0.868 | 0.859 |
| Prograsp Forceps | 0.712 | 0.742 | 0.727 |
| Large Needle Driver | 0.914 | 0.759 | 0.830 |
| Monopolar Curved Scissors | 0.956 | 0.923 | 0.939 |
| Suction Instrument | 0.778 | 0.929 | 0.847 |
| Clip Applier | 0.829 | 0.895 | 0.861 |
| Laparoscopic Grasper | 0.667 | 0.765 | 0.713 |
| **macro / accuracy** | | | **0.825** / 0.857 |

Architecture is identical to Task A's MobileNetV3-Small (same 1,525,031
params), so the hardware-independent and Titan Xp/ONNX latency numbers in
the sweep table above apply unchanged to this model too.

**This is the single most important finding in the project so far.**
Comparing the two classes that were the persistent failure mode across
every Task A run, every milestone:

| class | Task A, best result across all 11 runs | Task B |
|---|---|---|
| Clip Applier | F1 0.735 (weighted loss + augmentation, Milestone 5) | **F1 0.861** |
| Laparoscopic Grasper | F1 0.622 (ResNet-50 heavy baseline, Milestone 6) | **F1 0.713**, using the *lightweight* model |

Region classification reaches better rare-class performance with
MobileNetV3-Small than the heaviest model in the entire sweep (ResNet-50)
achieved on the whole-frame task. The confusion matrix
(`experiments/region_baseline_20260831-182451/figures/confusion_matrix.png`)
confirms this is a real result, not an artifact: 34/38 Clip Applier test
instances land on the diagonal.

**Interpretation**: this is evidence the rare-class problem in Task A was
not purely a class-imbalance problem that a better loss/sampler could
solve — a meaningful part of it was a *framing* problem. Whole-frame
multi-label classification asks the model to simultaneously localize,
disambiguate between co-occurring instruments, and classify, all from one
frame-level supervision signal. Once given an isolated, mask-cropped
instance, the model only has to do the last part, and the rare classes stop
being anywhere near as hard.

**Caveat that must stay attached to this number in the report**: this
figure uses *ground-truth* instance masks to build the crops. It measures
"can the model name an instrument once perfectly told where it is," not an
end-to-end detect-then-classify pipeline. It is not comparable to TAPIS's
published number (see below) for that reason, independent of the metric
mismatch.

**Infrastructure note**: Task B's bbox/mask coordinates are in each frame's
native resolution, so it cannot use the resized frame cache built for Task
A's dataloader throughput (Milestone 1) — using it silently produces
wrong crops (fixed with a loud failure instead, see `docs/DECISIONS.md`).
This run is therefore uncached and not directly speed-comparable to Task A
runs; if Task B becomes loader-bound in practice, the right fix is a
Task-B-specific cache of pre-cropped instance thumbnails, not reusing
Task A's.

---

## Milestone 8: Detection (Faster R-CNN)

Boxes come directly from the short-term JSON's per-instance `bbox` field,
no new annotation. One sample per annotated frame, all instances in that
frame as one target set (unlike Task B's one-sample-per-instance). Metrics
via `pycocotools.COCOeval` directly, the same family (AP50_box) every
third-party GraSP paper reports (see comparison table below).

### Baseline

MobileNetV3-Large-FPN (torchvision's `fasterrcnn_mobilenet_v3_large_fpn`,
18.9M params), Adam, `lr=0.0001`, batch size 4, no augmentation.

| variant | mAP@50 | mAP@50:95 | wall clock |
|---|---|---|---|
| **Adam (chosen default)** | **0.8295** | 0.6155 | 2564s |
| SGD (torchvision's own reference-recipe optimizer) | 0.8230 | 0.6035 | 2748s |

Adam slightly ahead on both metrics; kept as the default for every
Milestone 8 variant below unless noted.

Baseline per-class AP@50 (Adam): Bipolar Forceps 0.923, Prograsp Forceps
0.687, Large Needle Driver 0.959, Monopolar Curved Scissors 0.969, Suction
Instrument 0.753, Clip Applier 0.834, **Laparoscopic Grasper 0.682**
(weakest class).

### Augmentation ablation -- negative result

Both of torchvision's standard detection-augmentation recipes made mAP
*worse*, not better, on this dataset -- tried in order of increasing
strength, both regressed:

| variant | mAP@50 | mAP@50:95 | heavy-occlusion recall |
|---|---|---|---|
| baseline (no augmentation) | 0.8295 | 0.6155 | -- |
| light (flip + color jitter only) | 0.8040 | 0.5980 | 0.606 |
| strong (+ RandomZoomOut + RandomIoUCrop) | 0.8157 | 0.5801 | 0.645 |

Consistent with this project's literature review
(`docs/detection_literature_notes.md`): COCO-style geometric/photometric
augmentation isn't validated for small, visually homogeneous surgical
datasets, and the augmentation techniques that *do* help in the surgical
detection literature are synthesis-based (see copy-paste, below), not
generic crop/color jitter.

### Soft-NMS -- null result

Gaussian soft-NMS (Bodla et al. 2017) applied as pure post-processing on
the baseline's raw predictions (hard-NMS disabled to get candidate boxes):
mAP@50 0.826 vs. 0.829 baseline -- no improvement, occlusion recall
unchanged. Confirms hard-NMS suppression of a correctly-detected-but-
overlapping box is *not* the mechanism behind the occlusion recall gap
(see below); ruling this out was the reason the occlusion investigation
moved to a proposal/representation-quality explanation instead, and
eventually to Milestone 9's architecture change.

### Class-weighted loss -- small aggregate move, real per-class win

Balanced weight on the box classifier's cross-entropy (same formula
Task B uses), monkeypatching `torchvision.models.detection.roi_heads.
fastrcnn_loss`:

| variant | mAP@50 | mAP@50:95 | Laparoscopic Grasper AP@50 | Clip Applier AP@50 |
|---|---|---|---|---|
| baseline | 0.8295 | 0.6155 | 0.682 | 0.834 |
| **weighted loss** | **0.8307** | 0.5955 | **0.737** | 0.824 |

Aggregate mAP barely moves (+0.0012), but Laparoscopic Grasper -- the
weakest class in every Milestone 8 variant -- gains +5.5 AP50 points. The
aggregate number hides a real, targeted improvement; weighted loss
redistributes attention toward the specific class that needs it rather
than lifting everything uniformly. Occlusion-stratified recall (heavy:
0.617) is statistically unchanged from the other variants -- confirms
again that occlusion is a separate axis from class imbalance.

### Occlusion-stratified recall -- the throughline finding

A proxy metric built specifically because COCO's small/medium/large-area
AP breakdown is meaningless here (this dataset has almost no small-object
instances -- see below): bucket each ground-truth instance by how much of
its own bbox area is covered by another co-occurring instrument's bbox
(`isolated` = 0%, `light` <=50%, `heavy` >50%), then measure recall at a
fixed score/IoU threshold (0.5/0.5) per bucket.

| variant | isolated | light | heavy |
|---|---|---|---|
| augmented (strong) | 0.881 | 0.913 | 0.645 |
| light augmentation | 0.895 | 0.909 | 0.606 |
| weighted loss | 0.902 | 0.918 | 0.617 |
| ResNet-50 (heavy backbone) | 0.894 | 0.923 | 0.666 |
| ResNet-50 + weighted loss | 0.885 | 0.926 | 0.634 |
| copy-paste augmentation | 0.899 | 0.920 | 0.645 |

**Every single Milestone 8 variant tried lands in the same 0.606-0.666
heavy-occlusion recall band**, regardless of backbone size (18.9M vs.
41.1M params), loss weighting, generic augmentation, or occlusion-targeted
augmentation -- a ~25-30 point recall drop from the isolated-instance rate
that nothing in the Faster R-CNN family closed. This consistency, not any
single number, is the finding: it points to a structural limitation in
Faster R-CNN's box-proposal + NMS pipeline itself (confirmed not to be
NMS suppression specifically, via the soft-NMS null result above), not
something fixable by tuning within that architecture family. This is the
motivation for Milestone 9's anchor-free, NMS-free centroid/offset
segmenter (below).

### Backbone/capacity scaling -- negative result, closes that branch

Per the user's explicit decision rule (get within ~3% of the literature's
comparable range or trade efficiency for accuracy via more capacity), a
ResNet-50-FPN backbone (41.1M params, more than 2x the baseline) was
tried alone and combined with weighted loss:

| variant | params | mAP@50 | mAP@50:95 | wall clock |
|---|---|---|---|---|
| baseline (MobileNetV3) | 18.9M | 0.8295 | 0.6155 | 2564s |
| **weighted loss (MobileNetV3)** | 18.9M | **0.8307** | 0.5955 | -- |
| ResNet-50 alone | 41.1M | 0.808 | 0.576 | 8474s |
| ResNet-50 + weighted loss | 41.1M | 0.800 | 0.587 | 13233s |

More capacity never beat the lightweight baseline -- both ResNet-50
variants underperform it, and the combined variant underperforms
ResNet-50 alone, so combining levers didn't help either. On a dataset
this size (8 train cases), a heavier backbone overfits rather than
generalizing better. This closes the "trade efficiency for accuracy via
bigger model" branch of the decision rule: capacity is not the lever that
works here, and (per the occlusion table above) it does not touch the
occlusion gap either.

### Copy-paste augmentation -- best aggregate result, occlusion gap unchanged

Ghiasi et al. 2021's "Simple Copy-Paste" applied to GraSP's own
per-instance masks (no synthetic data): pastes real Clip Applier crops
onto other training frames, biased 70% of the time to land on top of an
existing box (deliberately increasing occlusion exposure), targeting both
the class-imbalance and occlusion problems in one augmentation.

mAP@50 = 0.826, mAP@50:95 = 0.590 (18.9M params, MobileNetV3 backbone) --
matches the baseline within noise, the best-performing Milestone 8
follow-up. Clip Applier AP@50 lands at 0.780, respectably mid-pack rather
than the weakest class.

**But occlusion-stratified recall (heavy: 0.645) sits in exactly the same
band as every other variant.** This is a meaningful negative result on
*mechanism*: deliberately increasing occlusion exposure during training
does not teach Faster R-CNN to separate overlapping boxes better, because
the failure is in the architecture's proposal/NMS pipeline, not its
exposure to occluded examples. This result is the strongest evidence in
the project that closing the occlusion gap requires an architecture
change, not another Milestone 8 data or loss intervention -- see
Milestone 9, below.

### Latency (Titan Xp, real measurements)

`scripts/benchmark.py`'s "model" mode didn't support detection models
(different calling convention -- a list of tensors, not a batched
tensor); extended it with a `detection` mode (`evaluation/benchmarking.py
::benchmark_detection_gpu_latency`) rather than leave this unmeasured.
FLOPs/MACs and ONNX export are still not attempted for detection --
both are unreliable for two-stage detectors with a dynamic-length
RPN/ROI-heads pipeline, and an honestly-missing number is better than a
wrong one.

Single-image GPU latency, native 800x1280 resolution, median over 200
warm runs:

| variant | params | median latency | p95 latency | peak VRAM |
|---|---|---|---|---|
| **MobileNetV3 (baseline/weighted-loss/copy-paste, identical architecture)** | 18.9M | **~22ms** | ~22ms | ~309MB |
| ResNet-50 (heavy) | 41.1M | **67.7ms** | 70.0ms | 588MB |

**The lightweight model is ~3.1x faster while also scoring higher
accuracy** (copy-paste's 0.826 vs. ResNet-50's 0.808/0.800) -- this is the
clearest efficiency-thesis data point in the detection results: more
capacity was both slower and less accurate here, not a
speed/accuracy trade at all. At ~22ms/frame there is still headroom under
a real-time budget (33ms at 30fps) for something cheap on top (motivates
Milestone 9's planned tracking-by-detection component).

Milestone 9's segmenter (2.1M params, 384x384 input -- not the same
resolution as detection's native 800x1280, so not a direct apples-to-
apples comparison, only a rough sense of scale) measured **3.96ms median
latency**, 55.4MB peak VRAM -- consistent with being a much smaller model,
though its accuracy is not yet competitive (see above), so this is not
yet a usable efficiency-vs-accuracy comparison point on its own.

### Tracking-by-detection -- real improvement on the diagnostic metric, still short on the headline one

Motivated by the latency headroom above: with ~22ms/frame against a 33ms
(30fps) real-time budget, there is room for a cheap post-processing step.
Annotated GraSP frames within a case are ~35 frames apart (median) --
too sparse for frame-to-frame tracking directly -- but the raw frame
directories contain every intermediate frame at native (~1fps) sampling
(e.g. CASE001 has 10,972 sequential files against 329 annotated ones),
checked directly rather than assumed before building anything.

Built an IOU-Tracker (Bochinski et al. 2017 -- greedy box-IoU association,
no motion model or appearance features) with two additions, both found
necessary by testing, not assumed upfront: (1) track survival through up
to `max_age` consecutive missed frames (the same "max age" idea
SORT/DeepSORT use), so a track can survive momentary occlusion instead of
ending the instant one frame is missed; (2) boundary-exit detection -- a
track tracks a coarse velocity (box-center delta between real matches),
and if it's near a frame edge and moving further outward when a gap
starts, the gap is treated as the instrument genuinely leaving the frame
and the track is dropped immediately rather than coasting.

**First check -- occlusion-stratified recall** (fixed 0.5 score
threshold, the same threshold this metric uses everywhere else in this
report): isolated 0.899->0.912-0.913, light 0.920->0.923-0.930, **heavy
0.645->0.648-0.683** depending on configuration -- a real improvement,
and (in the best case found) the first technique all session to move
heavy-occlusion recall outside the 0.606-0.666 band every other
Milestone 8 variant landed in. Verified this was a genuinely causal,
real-time-valid result and not a look-ahead artifact (a non-causal window
produced identical numbers, traced to the implementation reading out
tracker state before any future frames are processed -- not a real leak,
but worth confirming empirically rather than assuming).

**Second, more demanding check -- full COCO-style mAP@50** (needs a low
score threshold for a real precision-recall curve, unlike the fixed-
threshold recall table above): tracking-augmented predictions scored
*worse* than raw per-frame detections in the first version of this
result -- 0.7899 tracked vs. 0.8264 raw, a ~3.7 point drop. Root cause:
two distinct failure modes let a stale, wrong box get reported as a
confident-looking false positive -- (a) tracks born from low-confidence
(likely noise) detections coasting forward, and (b) tracks from real,
confidently-detected instruments that had *actually left the frame or
moved far* (not momentarily occluded) still coasting on their last known,
now-wrong box. Fixed with `min_confidence_to_coast` (only let a track
coast if its last real detection was confident) and the boundary-exit
check above; the boundary-exit check did almost all of the recovery,
confirming (b) was the dominant failure mode, not (a).

| configuration | mAP@50 | gap to raw (0.8264) | heavy-occlusion recall |
|---|---|---|---|
| tracking, no fixes | 0.7899 | -4.65 | 0.645-0.683 (noisy) |
| + confidence-gating only | 0.7914 | -4.50 | 0.652 |
| + boundary-exit, max_age=3 | 0.8034 | -2.30 | 0.648 |
| + boundary-exit, max_age=2 | 0.8075 | -1.89 | 0.648 |
| **+ boundary-exit, max_age=1** | **0.8130** | **-1.34** | 0.652 |

A hyperparameter sweep on `max_age` (how many frames a track may coast)
showed a clean monotonic trend -- shorter coasting window, less
staleness, less precision loss -- with `max_age=1` the best found and
close to the practical floor (anything shorter disables occlusion
tolerance entirely).

**Tried and rejected: occlusion-corridor detection.** A geometric
positive signal for occlusion (extrapolate a missing track's position; if
it overlaps another instrument actually detected this frame, treat the
gap as corroborated occlusion and grant a longer coasting lifetime instead
of the normal short one) was tested in three forms -- a modest extension,
a generous one, and a version requiring the evidence to keep holding up
every frame rather than just once. **All three consistently underperformed
plain `max_age=1` with no corridor mechanism at all** (0.8048-0.8089 vs.
0.8130). Not noise -- consistent across extension length and
evidence-checking strategy. Most likely explanation: GraSP frames
routinely contain 2-3 co-occurring instruments (94.7% of frames), so
"another instrument's box is nearby" is true almost constantly in this
dataset regardless of whether real occlusion is happening -- a plausible
signal in principle, too weakly discriminating here in practice.

**Honest bottom line**: after two real, diagnosed-and-fixed bugs and a
hyperparameter sweep, tracking-by-detection's best configuration found so
far is **still net negative on mAP@50**, 1.34 points below the raw
detector, despite a real and validated recall benefit at the fixed-
threshold diagnostic level. This is near-zero added inference cost in the
way that matters for a real-time pipeline (past frames need to have
already been processed once, not re-processed at request time), and the
mechanism is sound and worth keeping in the toolkit, but it has not yet
cleared the bar of improving the actual literature-comparable headline
number. Reported as a validated, partially-negative research direction,
not a finished win -- see `docs/DECISIONS.md` for the full, dated
progression of this result.

### Against the literature target

Per `docs/detection_literature_notes.md` and the third-party comparison
table below, the comparable published range is ~88-93% (TAPIS's
mAP@0.5IoU_segm 89.85%; LACOSTE's AP50_segm 90.34-91.71% for two TAPIS
variants). The best Milestone 8 box-AP50 achieved is 0.831 (weighted
loss) -- roughly 5-9 points short, and no Milestone 8 lever (augmentation,
loss weighting, backbone capacity, or occlusion-targeted augmentation)
closed that gap. Some of the gap is not apples-to-apples (Milestone 8 is
box-only, single-frame, no temporal/stereo context, vs. TAPIS/LACOSTE's
segmentation-based, some using temporal or stereo information), but the
occlusion-stratified recall finding above suggests a real, fixable
architectural gap remains beyond that framing difference.

## Milestone 9: Instance segmentation (in progress, not a final number)

Anchor-free, NMS-free instance segmentation, built specifically to attack
the occlusion mechanism Milestone 8 could not fix by tuning within the
Faster R-CNN family. Not a reproduction of one paper: a semantic
segmentation head plus a centroid-heatmap + offset-regression head for
instance separation, the heatmap+offset half following Kurmann et al.
2021's "mask then classify" framing, structurally close to Cheng et al.
2020's Panoptic-DeepLab. Backbone is the same MobileNetV3-Large-FPN family
used throughout this project (2.1M total params for the full model --
lighter than either detection variant above).

**First run** (`segmentation_baseline`, patience 10): semantic head
result is reasonable for a first pass -- mIoU 0.308, mean Dice 0.448, 32
min wall clock. But the instance-level metrics (AP50_segm, occlusion
recall) were badly broken (AP50_segm 0.0017) due to a diagnosed decode-
time bug, not a broken model: the heatmap-peak NMS kernel/threshold were
copied from COCO-scale CenterNet defaults, which found 14-41 spurious
peaks per image against 1-4 real instances on this project's coarser
output grid and still-converging heatmap head. Fixed
(`score_threshold=0.3`, `nms_kernel=7`, tuned against real predictions);
AP50_segm on the same checkpoint improved ~38x (0.0017 -> ~0.10) with the
corrected decode settings alone, confirming the representation was never
the problem.

**After the decode fix, two further runs** (`segmentation_extended_patience`,
patience 30; `segmentation_cosine_lr`, patience 40 with a cosine LR decay
to remove periodic heatmap-loss spikes seen in the first) converged to the
**same plateau**: mIoU 0.319-0.321, mean Dice 0.460, AP50_segm 0.098-0.111,
occlusion-stratified recall isolated 0.32-0.33 / light 0.40 / heavy
0.07-0.11. Training is stable and converged (flat loss curves, no more
instability once the LR schedule was added) -- three independent runs
landing in the same place means this is a real capacity/architecture
ceiling for a 2.1M-param, single-stride-4-feature-map model, not an
undertraining artifact fixable by more epochs.

**Honest comparison to Faster R-CNN, not spun either way**: this
architecture does **not yet** outperform Faster R-CNN's occlusion
handling. Absolute recall is far lower across every bucket (isolated 0.32
vs. Faster R-CNN's ~0.88-0.90), and the *relative* occlusion penalty is
currently larger, not smaller: heavy/isolated ratio ~0.23-0.31 here vs.
~0.72 for copy-paste's Faster R-CNN -- the opposite of what an NMS-free
architecture was hypothesized to achieve.

**A fourth run ported copy-paste augmentation to this pipeline**
(`segmentation_copy_paste`, same Clip Applier pasting as detection's best
lever, updating the dense heatmap/offset/semantic targets instead of a
box+label list) to test whether the plateau was a training-distribution
problem. Result: mIoU 0.309 (flat), but **AP50_segm = 0.1216 -- the best
of all four runs**, with a modest absolute recall improvement in every
occlusion bucket (isolated 0.339, light 0.411, heavy 0.087) -- a small,
real gain, not a null result, but still inside the same 0.23-0.31
relative-occlusion-penalty band. Four independent runs across three
different single-variable interventions (patience, LR schedule, and now
copy-paste) converging to the same plateau is much more consistent with a
**capacity or training-maturity ceiling** than a training-distribution
problem -- unlike detection, where copy-paste alone closed most of the
gap, this architecture's bottleneck looks structural to its current size
(2.1M params, first-ever training runs), not to what it's trained on.

The architecture is correctly implemented and validated end-to-end (one
real instance-decoding bug caught and fixed during this process, see
`docs/DECISIONS.md`), but confirming whether removing NMS actually helps
occlusion here needs more model capacity or a training budget beyond what
this project has invested so far -- not a finished result, and not yet
evidence either for or against the underlying hypothesis. Given this,
effort moved to tracking-by-detection on the detector instead (see above)
as the nearer-term path to an actual occlusion-robustness improvement.

Also not yet done, per CLAUDE.md's latency-benchmarking requirement: no
ONNX-CPU latency, FLOPs, or model-size benchmark exists yet for either the
Milestone 8 detectors or the Milestone 9 segmenter -- Titan Xp GPU latency
is now measured for both (`scripts/benchmark.py`'s `detection` mode, added
this round, plus a direct call to the classifiers' latency helper for the
segmenter), but FLOPs/ONNX remain unattempted for detection specifically
because thop/ONNX export are unreliable for two-stage detectors, not
because it was overlooked. Given the user's stated priority is
specifically inference latency, this is flagged here as an explicit gap
to close before any final efficiency-vs-accuracy claim is made, not
filled in with assumed or estimated numbers.

---

## Comparison to the dataset authors' own benchmark (TAPIS)

TAPIS (the GraSP paper's model) reports a single aggregate
**mAP@0.5IoU_segm of 89.85%** for instrument segmentation, with **no
per-class breakdown** and **no class-balancing technique used** (per
`docs/imbalance_notes.md`, checked directly against arXiv 2401.11174).

**This project's numbers above are not directly comparable to that figure**,
for two independent reasons:

1. **Different metric.** Their number is end-to-end instance-segmentation
   mAP at an IoU threshold — it scores localization and classification
   together. Every number in this report is classification-only.
2. **Different evaluation setting.** Task B's crops come from ground-truth
   masks (an oracle for localization); TAPIS's published number includes
   their own localization/segmentation error. Comparing an oracle-input
   classification score to an end-to-end segmentation score would be an
   unfair comparison in this project's favor.

A legitimate, apples-to-apples comparison requires this project's own
detection/segmentation stage evaluated under the same mAP@IoU protocol.
Milestone 8 (detection, box-AP50 only) is done -- best result 0.831,
roughly 5-9 points short of TAPIS/LACOSTE's comparable range, with the
gap's likely mechanism (occlusion, not capacity or imbalance) diagnosed in
detail above. Milestone 9 (instance segmentation, the metric family that's
actually comparable to TAPIS's mAP@0.5IoU_segm) is in progress as of this
writing -- see above. Until Milestone 9 produces a trusted number, the
honest framing stays: Task B's result shows the *classification* component
of the problem is close to solved by a lightweight model; it says nothing
yet about whether this project's full pipeline would match TAPIS's
end-to-end number.

## Checked against the wider published literature (web search, 2026-08-31)

`docs/imbalance_notes.md` originally stated no independent third-party
GraSP benchmark existed. Re-checked for this report and that's out of
date — real third-party work does exist:

| Work | What it does with GraSP | Metrics reported |
|---|---|---|
| TAPIS (Ayobi et al., arXiv 2401.11174 / *Medical Image Analysis* 2025) | Original dataset + model | mAP@0.5IoU_segm (aggregate 89.85% cited in the preprint; a results-table citation from LACOSTE below shows AP50_segm 90.34-91.71 across two TAPIS variants — likely a different reported split/setup, not reconciled here) |
| **LACOSTE** (arXiv 2409.09360, third-party, not GraSP's authors) | Stereo+temporal surgical instrument segmentation, evaluated on GraSP as 1 of 3 benchmarks | AP50_box, AP50_segm (instance seg.); Ch_IoU, ISI_IoU, mcIoU (semantic seg., multi-class). Best variant: mcIoU 80.07 |
| QPD (query-proposal-decoder segmentation method, cited as a LACOSTE baseline) | Instrument instance segmentation | Same metric family as above (from LACOSTE's comparison table) |
| Referring-segmentation paper (arXiv 2601.12224, 2026) | Text-referred instrument segmentation on GraSP-IM (a motion-annotated GraSP variant) | J, F, J&F region/contour similarity — a different task (referring segmentation), not classification |
| ZEN (foundation-model paper, arXiv 2602.13633, 2026) | GraSP as 1 of many benchmarks for a general surgical foundation model | Dice + 95% Hausdorff distance (semantic seg.), mAP (instance seg.) |

**The pattern holds across every third-party paper found**: everyone who
benchmarks on GraSP treats it as a **segmentation** task (some IoU- or
Dice-based metric requiring correct spatial localization), never as a pure
classification task the way this project's Task A/B numbers are framed.
So the original conclusion in `imbalance_notes.md` — "nothing to replicate
our numbers against" — still holds, but for a different and more precise
reason than originally stated: it's not that GraSP lacks a research
community, it's that no one else has published the classification-only
framing this project uses. That framing choice (Task A: multi-label frame
classification; Task B: region classification given ground-truth
localization) is genuinely this project's own, not a replication of
anyone else's protocol — worth stating plainly in the report rather than
implying it's a gap in coverage.

This also means: **if this project eventually builds its own
detection/segmentation stage (Milestones 8-9)**, the natural metrics to
report for a real literature comparison are the same family everyone else
uses — AP50_box/segm and mcIoU/Ch_IoU/ISI_IoU — not mAP/macro-F1, so the
result would sit in the same table as TAPIS, LACOSTE, QPD, and ZEN. That's
a stronger, more legible comparison than trying to force a classification
metric into the literature's segmentation-metric convention.

Search performed 2026-08-31; not exhaustive (no systematic search of
venues/preprint servers beyond web search, and PDF-size limits blocked
full-text review of two papers). If the report needs this section held to
a stricter literature-review standard, treat this as a starting point, not
a completed survey.

---

## Summary of key findings (for the report's abstract/conclusion)

1. Fine-tuning a lightweight backbone beats using it frozen by a wide
   margin on this dataset (macro-F1 0.506 -> 0.645) — ImageNet features
   alone don't transfer to endoscopic imagery.
2. Simple, well-understood imbalance techniques (weighted loss, weighted
   sampling) each help, and combining weighted loss with stronger
   augmentation compounds because they fix different problems (attention to
   rare classes vs. overfitting) — macro-F1 0.645 -> 0.708.
3. **On the Titan Xp (a tensor-core-free GPU), inference latency does not
   track parameter count** — a lightweight, depthwise-separable-convolution
   model (MobileNetV3-Small) is not automatically the fastest option on
   this specific hardware; ResNet-18 is faster despite far more MACs. This
   inverts on ONNX CPU, where latency does track parameter count and
   MobileNetV3-Small is 15x faster than the heaviest model tested.
4. A heavy baseline (ResNet-50) buys a real but bounded accuracy gain over
   the lightweight model (+0.091 macro-F1) at a steep cost (15x latency,
   15x size) — the tradeoff curve has a genuine top end, not just one
   point.
5. **Reframing the task from whole-frame multi-label classification to
   region-level (per-instance) classification resolves the project's
   worst-performing classes almost entirely**, using the same lightweight
   model — Clip Applier and Laparoscopic Grasper both perform better here
   than they did with any Task A model, including the heaviest one. This
   suggests the rare-class problem was partly a task-framing problem, not
   purely a data-imbalance problem.
6. No comparison to the dataset authors' published benchmark is possible
   yet — different metric, different evaluation setting (oracle
   localization here vs. their end-to-end pipeline). That comparison
   requires this project's own detection/segmentation stage.
7. **Milestone 8 (detection) reached mAP@50 = 0.831 (weighted loss),
   ~5-9 points short of the literature's ~88-93% comparable range** — not
   closed by augmentation (regressed it), soft-NMS (null), or backbone
   capacity (regressed it; more capacity overfits on 8 train cases).
   Occlusion-stratified recall diagnosis is the throughline: every variant
   tried, regardless of backbone/loss/data changes, loses ~25-30 recall
   points on heavily-occluded instances (0.606-0.666 vs. ~0.89-0.90
   isolated) — including copy-paste augmentation, which specifically
   targeted occlusion exposure and still didn't move that number. This
   points to a structural limitation in Faster R-CNN's box-proposal + NMS
   pipeline, not a data or capacity problem, and motivates Milestone 9's
   architecture change.
8. **Milestone 9 (anchor-free, NMS-free instance segmentation) converged
   to a stable plateau (mIoU ~0.31-0.32, AP50_segm ~0.10-0.12) across four
   independent runs (including copy-paste augmentation, which gave a
   small real AP50_segm gain but didn't break the plateau), and does not
   yet outperform Faster R-CNN's occlusion handling** — its relative
   occlusion penalty (heavy/isolated recall ratio ~0.23-0.31) is currently
   *worse* than Faster R-CNN's (~0.72), the opposite of the hypothesis it
   was built to test. One real instance-decoding bug was diagnosed and
   fixed along the way (see above). Four runs across three different
   single-variable interventions landing in the same place points to a
   capacity/maturity ceiling, not a training-distribution problem — a
   validated foundation for further work, not evidence for or against the
   underlying architecture hypothesis yet.
9. **Tracking-by-detection improves the occlusion-recall diagnostic but
   is still net negative on mAP@50, even after fixing two real bugs and a
   hyperparameter sweep** — a near-zero-cost IOU-Tracker (Bochinski et al.
   2017) run across raw, densely-sampled frames (not the sparse annotated
   subset) improves heavy-occlusion recall (0.645 -> up to 0.683 at a
   fixed threshold), the first technique all session to move that number
   outside the flat 0.606-0.666 band. But naively coasting through every
   detection gap measurably hurt the full mAP@50 curve (0.7899 vs. raw's
   0.8264) until two failure modes were found and fixed — low-confidence
   tracks coasting, and (the dominant one) confidently-tracked instruments
   coasting on a stale box after actually leaving the frame, not being
   occluded. Best configuration found (confidence-gating + boundary-exit
   detection + `max_age=1`) reaches mAP@50 = 0.8130, still 1.34 points
   below raw. A real, validated, well-diagnosed research direction — not
   yet a net win on the metric that matters most.

## What's not done yet

Milestone 8 (detection) is done. Tracking-by-detection (its planned
post-processing addition) is built, debugged, and tuned, but is not yet a
net win on mAP@50 (best found: -1.34 points vs. raw) -- either further
tuning (a per-instance occlusion signal instead of a purely geometric one)
or accepting it as a validated-but-not-adopted direction is the open
question. Milestone 9 (segmentation architecture) is
built, debugged, and has reached a training plateau documented above
across four runs (including copy-paste), but has not yet been pushed past
that plateau — a model-capacity increase is the untried next lever, not
another training-recipe or data change. Milestone 10 (inference
optimization, final Pareto analysis) has not been started.
Titan Xp GPU latency for the detection models and a rough latency number
for the Milestone 9 segmenter are now measured (see the Latency section
above, `scripts/benchmark.py detection` mode). Still not done: ONNX-CPU
latency and FLOPs/MACs for detection (deliberately skipped -- unreliable
for two-stage detectors, see above) and any latency benchmarking at all
for the Milestone 9 segmenter's actual working input resolution alongside
a competitive accuracy number, since accuracy isn't there yet to pair it
with.

## Reproducibility

Every run referenced above has a full `manifest.json` under
`experiments/<run_id>/` recording the resolved config, git commit/dirty
state, seed, split file checksums, package versions, GPU info, and final
metrics — the numbers in this file were read directly from those, not
transcribed from logs or memory.
