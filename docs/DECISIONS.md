# Decisions log

Append-dated, newest last. Read this before reversing or re-litigating a past
choice.

## 2026-08-31 — Category 7 name: "Laparoscopic Grasper", not "Laparoscopic Instrument"

CLAUDE.md's dataset facts list instrument class 7 as "Laparoscopic
Instrument". The annotation JSON (`categories` field in
`grasp_short-term_*.json`) names it `Laparoscopic Grasper`. Verified across
train and test files during Milestone 0 dataset inspection
(`scripts/inspect_dataset.py`, see `docs/dataset_report.md`).

Decision: code, configs, and docs going forward use the JSON name,
`Laparoscopic Grasper`, since it's the string that must match the data at
training time. CLAUDE.md's class list should be corrected to match on next
edit rather than left to drift; not changed in this pass to avoid silently
rewriting a rules file the user hasn't reviewed.

**Update, same day:** user asked to fix the mismatch. `CLAUDE.md` and
`GraSp/CLAUDE.md` now say "Laparoscopic Grasper" (`data/instrument classes`
bullet), and the reference dict in `scripts/inspect_dataset.py` was updated
to match, so the report's discrepancy section no longer fires. Confirmed by
re-running `scripts/inspect_dataset.py` — `docs/dataset_report.md` no longer
has a "Discrepancy vs CLAUDE.md" section.

## 2026-08-31 — Segmentation directory layout differs by split

`annotations/segmentations/` holds train-fold case directories
(`CASE001`, ...) directly at the top level, but test-fold cases sit one
level deeper under `annotations/segmentations/test/`. Not documented in
CLAUDE.md or the dataset's own README. Handled in
`src/surgical_ai/data/statistics.py::mask_path`, which takes the split name
and resolves the path accordingly. Any future code touching segmentation
masks needs the same split-aware resolution, not a flat glob.

## 2026-08-31 — Dataset root path/layout differs between dev laptop and workstation

The Titan Xp workstation already had a full copy of GraSP (14 GB, all 13
cases, annotation JSONs verified byte-identical in image/annotation counts
to the dev laptop's copy) at
`~/Desktop/Classification Surgurical Tools/GraSP` — different casing
(`GraSP` vs. the dev laptop's `GraSp`) and a flatter frame layout
(`GraSP/frames/CASE0xx/`, not `GraSp/frames-001/frames/CASE0xx/`), likely
from an earlier student's setup.

Rather than change the dataset layout convention the code already assumes
(`<root>/frames-001/frames/`), added a compatibility symlink on the
workstation only: `GraSP/frames-001/frames -> ../frames`. No code changes.
`GRASP_DATA_ROOT` must be set per-machine regardless (`./GraSp` on the
laptop, `./GraSP` on the workstation) — this was already required since
paths are never hardcoded, but the casing difference makes it easy to typo.
If the workstation's `GraSP` directory is ever rebuilt from scratch, redo
the symlink (`mkdir -p frames-001 && ln -s ../frames frames-001/frames`)
rather than restructuring the actual frame directory.

## 2026-08-31 — Resized frame cache: short side 256px, layout mirrors source root

Milestone 1 loader benchmarking on the workstation showed the dataloader
capped at 317 img/s (4 CPU threads, all saturated) against a model that can
consume 5258 img/s on the GPU — a >10x gap, meeting CLAUDE.md's stated
criterion for building the resized cache of annotated frames only.

`scripts/build_frame_cache.py` resizes each of the 3449 annotated frames so
its shorter side is 256px (source is 1280x800, aspect preserved, not forced
square) and re-encodes as JPEG quality 90. 256 was chosen over the training
`image_size` (224) so `RandomResizedCrop(224, scale=(0.8, 1.0))` still has
real spatial room to crop from, rather than crop-then-upsample a
already-224px image, which would make the "crop" augmentation close to a
no-op zoom. This does mean train-time crops now come from a 256px-short-side
image instead of the native 800px one — a second resize step versus the
uncached path — but at 93.3 MB for the full cache this was judged an
acceptable tradeoff for the 2.6x loader speedup (317 -> 822 img/s at 4
workers), not something expected to measurably change reported metrics.

The cache directory mirrors the source root's layout exactly
(`frames-001/frames/<file>`, `annotations/` symlinked to the source, not
copied) so any script takes it as a drop-in `--data-root` with zero changes
to `dataset.py`/`splits.py`/`transforms.py`. Built at
`~/Desktop/Classification Surgurical Tools/GraSP_cache` on the workstation
only — the uncached `GraSP` directory remains the source of truth and this
script never modifies it. Not synced to the dev laptop (would need the full
dataset there first, which per CLAUDE.md's environment split isn't the
point of that machine).

## 2026-08-31 — Weighted sampler uses max, not average, class weight per sample

`training/samplers.py::compute_sample_weights` assigns each multi-label
sample the *maximum* inverse-class-frequency among its active classes, not
the mean. GraSP frames routinely have 2-3 co-occurring instruments (per
CLAUDE.md's task-framing note), so a frame containing both a common
instrument (e.g. Bipolar Forceps) and a rare one (Clip Applier) is common.
Averaging the two classes' weights would dilute the rare-class boost roughly
in proportion to how often it co-occurs with common ones, undermining the
entire point of weighted sampling for exactly the frames where the rare
class is most likely to appear. Max avoids that: any frame with a rare class
present gets the full boost regardless of what else is in frame. Verified
with `tests/test_samplers.py::test_sample_with_rare_and_common_class_takes_max_not_average`.
Milestone 5's ablation results are consistent with this working as intended
(weighted sampler and weighted loss deliver similar Clip Applier gains).

## 2026-08-31 — ResNet-50 as the Milestone 6 "deliberately heavy baseline"

CLAUDE.md and PROJECT_SPEC.md Sec.10 both require a heavy baseline for the
top of the tradeoff curve but don't name one. TAPIS itself (the paper's own
model) is not it for this milestone — it needs spatio-temporal video
features and a segmentation stage this project hasn't built yet
(Milestone 9), and `docs/imbalance_notes.md`'s note about TAPIS being "the
correct heavy baseline" was written about Task B (region classification,
Milestone 7), not this frame-classification sweep. A ViT was considered and
rejected: it needs more data/longer schedules to be a *fair* heavy baseline
rather than just a badly-tuned one, which would misrepresent the tradeoff
curve's top end.

Picked ResNet-50 (25.6M params vs. 1.5-11M for everything else in the
sweep): standard, well-understood, in torchvision already, and heavy purely
from parameter count/depth rather than needing task-specific tuning to be
representative. Registered as `resnet50` in
`models/classifiers/resnet.py`, sharing the same builder as `resnet18`
(both use torchvision's `.fc` head, `model.head = model.fc`).

Also fixed while adding ResNet: `scripts/train.py::build_optimizer`'s
discriminative-LR split previously matched `"classifier"` in parameter
names to find the head, which would have silently misclassified every
ResNet param as backbone (`.fc`, not `.classifier`) — not an error, just a
config that quietly stopped doing what its comment said. Replaced with a
registry-wide convention: every builder in `models/classifiers/` now sets
`model.head` to the actual head submodule, and the optimizer/freezing code
reads that instead of guessing from a name substring.

## 2026-08-31 — Task B (region classification) cannot use the resized frame cache

`GraspRegionDataset` crops instances by `bbox`/`segmentation` coordinates,
which are recorded in each frame's *native* resolution (800x1280).
Milestone 1's resized frame cache (`GraSP_cache`, short side 256px, built
for Task A's dataloader throughput) shrinks frames without touching
annotations, so a bbox computed for the 1280-wide original can point past
the edge of a ~410px-wide cached frame. First symptom was a `ZeroDivisionError`
deep inside `RandomResizedCrop` (crop width came out 0) when Task B's
baseline run was accidentally pointed at `--data-root ./GraSP_cache` on the
workstation — the actual bug was one level up: the clipped bbox silently
produced a degenerate region instead of erroring near the cause.

Fixed two ways: (1) `GraspRegionDataset.__getitem__` now compares the
decoded frame's shape against `segmentation["size"]` and raises `ValueError`
immediately if they don't match, so pointing Task B at the cache fails
loudly with a clear message instead of a cryptic downstream crash; (2) Task
B training must use the uncached `./GraSP`/`./GraSp` root going forward,
documented in `region_dataset.py`'s module docstring. This means Task B
doesn't get Milestone 1's loader speedup — if it turns out to be
loader-bound (plausible: it decodes the same ~2324 frames repeatedly, once
per instance, so more redundant I/O than Task A's one-decode-per-frame),
the right fix is a Task-B-specific cache of pre-cropped, pre-masked instance
thumbnails (already at final resolution, sidestepping the coordinate
mismatch entirely) rather than trying to reuse Task A's frame-level cache.
Not built yet -- measure first, per the same rule that justified Task A's
cache in the first place.

## 2026-08-31 — Dev laptop has a working GPU; "no local GPU" in the docs was wrong

User asked why the dev laptop would be slower for training, question
implied it shouldn't be — turned out `docs/environment.md` and CLAUDE.md's
hardware section both stated this machine has no local GPU. `nvidia-smi`
shows a real, working NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, driver
610.74). The "no GPU" claim was never true about the hardware; only a
CPU-only torch build (`2.13.0+cpu`) had been installed, so
`torch.cuda.is_available()` returned `False` for a reason unrelated to
whether a GPU existed. Nobody had run `nvidia-smi` to check the premise
before writing it down — a reminder that "is_available() is False" and "no
GPU" are different claims, and the milestone -1 standard (verify with a
real tensor op, on a machine you've actually checked) should have been
applied here too, not just to the workstation.

With the user's approval, installed `torch==2.13.0+cu126` /
`torchvision==0.28.0+cu126` here (Ada Lovelace, compute capability 8.9, no
Pascal-style sm-support constraint — any recent CUDA build works, cu126 was
picked for consistency with the workstation, not necessity) and verified
with the same matmul+conv2d standard as Milestone -1. Measured ~4x speedup
over the CPU build (74.4 -> 292.6 img/s, MobileNetV3-Small fine-tuning,
uncached). Full numbers and caveats in `docs/environment.md`; `CLAUDE.md`'s
"never assume local GPU access" line was amended in place (not removed) to
clarify it's about machines you haven't checked, not a standing fact about
this one.

This machine could become a second real training node (parallel runs
alongside the workstation, not just prototyping), but that's a bigger
scope decision than fixing the doc — nothing in this project's milestone
plan currently assumes it, and doing so properly would need its own frame
cache and its own honest throughput comparison rather than reusing the
workstation's numbers. Not decided; flagging for whenever it comes up.

## 2026-08-31 — Milestone 8 detection baseline: Faster R-CNN, Adam, no augmentation yet

Registered `fasterrcnn_mobilenet_v3` (torchvision's
`fasterrcnn_mobilenet_v3_large_fpn`) as the detector, consistent with the
project's lightweight-backbone thesis and no-novel-architecture rule.
Reports mAP@50/mAP@50:95/per-class AP@50 via `pycocotools.COCOeval`
directly (not a hand-rolled mAP calculation) — this is deliberately the
same metric family the literature check in `docs/findings.md` found every
third-party GraSP paper uses (TAPIS, LACOSTE's AP50_box), so this number
will be directly citable against theirs once it exists, unlike the Task
A/B classification metrics.

Two simplifications for this first baseline, not permanent choices:
- **Optimizer is Adam**, not the SGD+momentum+StepLR that standard
  torchvision detection recipes use. Kept for consistency with every other
  training script in this project rather than introducing a second
  optimizer convention for one task. If convergence is poor, SGD is the
  first thing to try before anything more exotic.
- **No augmentation** (not even the horizontal flip used elsewhere) —
  flipping requires flipping box coordinates too, which none of the
  existing transform code does. Left as a real gap, not a decision to
  leave unaugmented permanently; add box-aware flip if the baseline needs
  it.

Detection samples are one full frame with all its instances as a single
target set (unlike Task B's one-sample-per-instance), so it does not
inherit Task B's resized-frame-cache restriction — box coordinates are
still in the frame's native resolution here since no cache is used, but
nothing about this task assumes the cache one way or the other. Batch
size is deliberately much smaller (4) than the classification tasks (32):
Faster R-CNN with an FPN holds multiple feature-pyramid levels and RPN
proposals per image in memory at full 1280x800 resolution, unlike a single
224x224 classifier forward pass.

## 2026-08-31 — Milestone 6's `model.head` alias was silently corrupting checkpoints

Found while trying to benchmark a Milestone 3 checkpoint on the dev
laptop's RTX 4060: `load_state_dict(strict=True)` failed with missing
`head.*` keys. Root cause: `model.head = model.classifier` (added in
Milestone 6 for the discriminative-LR optimizer split) goes through
`nn.Module.__setattr__`, which registers any `nn.Module`-valued attribute
into `_modules` — since `classifier`/`fc` was already registered under its
real name, this silently duplicated every one of its parameters under a
second `head.*` key in `state_dict()`. Every classifier built after the
Milestone 6 refactor was saving checkpoints with redundant tensors, and —
the actual breakage — those checkpoints became incompatible with the
pre-refactor model code, and pre-refactor checkpoints (all of Milestones
3, 4, 5) became unloadable with strict mode against the current code,
since the checkpoint lacks the `head.*` keys the current model now
expects.

Fixed with `models/classifiers/common.py::set_head`, which writes directly
into `model.__dict__` instead of going through `nn.Module.__setattr__` --
`model.head` still resolves to the same submodule object, but is no longer
a second registered module. Verified: a freshly built model has zero
`head.*` state_dict keys, and the Milestone 3 `baseline_frozen` checkpoint
now loads under `strict=True` again. All checkpoints saved between the
Milestone 6 refactor and this fix (the four sweep runs' `best.pt` files)
have the harmless duplicate keys baked in; they still load fine into the
old (buggy) code that produced them, they just can't be loaded into the
fixed code without `strict=False`. Not worth re-training those four runs
over a checkpoint-portability issue that doesn't affect any reported
metric — flagging here in case anyone tries to reload one of those four
specific checkpoints later and hits a mismatch.

## 2026-08-31 — Detection: added augmentation, early stopping; corrected a wrong claim about small objects

User asked whether detection avoids guessing on barely-visible instruments,
worried the flat 0.000 AP@small-area from Milestone 8's baseline was
evidence of exactly that failure. Checked the actual data before acting on
it: only 2 train / 1 test instance in the whole dataset falls under COCO's
"small" threshold (bbox area < 32x32px), and even the 1st percentile of
*true visible pixel area* (the segmentation mask's pixel count, not just
the bbox) is ~2400-2700px^2 -- well above that threshold. GraSP essentially
has no genuinely tiny/sliver instances; AP@small=0.000 was a near-empty
metric bucket, not a measured failure. This corrects something said earlier
in the session (that number was cited as direct evidence) -- worth
recording so it isn't re-cited as a real finding later.

What the user's concern actually maps to on this dataset is occlusion
between co-occurring instruments (already documented: ~30% of pairs have
overlapping bboxes, ~10% with one >50% covered by another), not object
size. Added `evaluation/detection.py::compute_occlusion_fractions` and
`evaluate_occlusion_stratified_recall` -- for each ground-truth instance,
the max fraction of its own bbox area covered by any other co-occurring
instrument's bbox in the same frame (a cheap proxy needing no new
annotation), bucketed into isolated / light (<=50% covered) / heavy (>50%)
overlap, with recall at a fixed score/IoU threshold reported per bucket.
This is a real measurement of the actual concern, run automatically at the
end of every detection training run from now on.

Also added, since training time is no longer a constrained resource for
this project (user: unlimited compute available, only inference-time
efficiency matters for the lightweight thesis) while the Milestone 8
baseline used a flat 20 epochs picked as "a reasonable number," not from
hitting a real plateau:
- `fit_detection` gained an optional `patience` (early stopping on val
  mAP@50, `training.patience` in config) -- `configs/detection_augmented.yaml`
  sets `epochs: 100` as a ceiling with `patience: 10`, training to actual
  convergence instead of an arbitrary fixed count.
- `data/detection_dataset.py::build_detection_transforms` adds
  torchvision.transforms.v2-based box-aware augmentation (RandomIoUCrop,
  RandomZoomOut, horizontal flip, color jitter -- torchvision's own
  reference detection recipe, not novel), since the Milestone 8 baseline
  used zero augmentation, a real known gap at the time. Boxes are carried
  as `tv_tensors.BoundingBoxes` so crops/flips move them correctly instead
  of needing hand-rolled coordinate math (a likely source of subtle bugs
  if done manually).

## 2026-08-31 — Detection: the "strong" augmentation recipe regressed both mAPs, reverted to a lighter default

`configs/detection_augmented.yaml` (Adam) ran to completion with the full
recipe above (RandomZoomOut + RandomIoUCrop + flip + color jitter),
early-stopped at epoch 30 (best epoch 20, patience 10). Result: mAP@50
0.816, mAP@50:95 0.580 -- both *worse* than the unaugmented Milestone 8
baseline (0.829 / 0.616), and 78 minutes instead of 43. The mAP@50:95 gap
(-3.6 pts, versus -1.3 pts on mAP@50) points at RandomZoomOut/RandomIoUCrop
specifically hurting box localization precision, not just adding noise --
plausible on this dataset: those transforms are tuned for large, visually
diverse datasets like COCO, and GraSP is small (2324 train frames) and
visually homogeneous (one surgical setup, similar framing throughout), so
the aggressive scale/crop jitter likely added more distribution mismatch
than useful diversity.

`build_detection_transforms` now takes `augmentation: "default" | "strong"`
(matching the classification tasks' naming in `transforms.py`).
`"default"` is now just horizontal flip + color jitter -- the subset that
actually helped the classification tasks -- and is the new default for
`data.augmentation` when unset. `"strong"` keeps the full recipe, retained
for reference/comparison, not recommended given the above. The one SGD run
already launched with the old code (`detection_augmented_sgd.yaml`,
running on GPU1 at time of writing) used the strong recipe unconditionally
since it started before this change landed -- its result is still useful
as a second data point on whether the regression is optimizer-independent,
just don't read its config as reflecting the new default.

## 2026-08-31 — Literature check before more trial-and-error; soft-NMS tried, another null result

User asked for a literature review on surgical tool detection before
continuing to guess at fixes for the occlusion problem. Notes in
`docs/detection_literature_notes.md`. Two takeaways acted on immediately:

1. **Soft-NMS** (Bodla et al. 2017; general CV, not validated for surgical
   instruments specifically, but directly targets hard-NMS suppressing one
   of two real overlapping objects -- our diagnosed failure mode) was cheap
   to test: rebuilt the *already-trained* original-baseline checkpoint
   (`detection_baseline_20260831-190820/best.pt`) with `box_nms_thresh=0.99`
   and `box_detections_per_img=1000` (hard NMS effectively disabled, room
   for the raw overlapping candidates to survive), re-scored those raw
   candidates with `evaluation/detection.py::apply_soft_nms_to_predictions`,
   and re-evaluated -- no retraining needed, since NMS is inference-only
   postprocessing. Result: mAP@50 0.826 vs. 0.829 original, mAP@50:95 0.621
   vs. 0.616, heavy-occlusion recall 0.617 vs. 0.620 -- all within noise,
   essentially no change either direction.
2. This is itself informative: if hard NMS were the actual bottleneck
   (discarding a second, well-placed box for an occluded instrument), soft-
   NMS keeping that box at a reduced score should have recovered some
   recall. It didn't move at all, which points toward the model not
   producing a good candidate box for the occluded instrument in the first
   place -- a representation/localization quality problem at the proposal
   stage, not a postprocessing one. This is consistent with (and now
   evidence for) `detection_literature_notes.md`'s recommendation to build
   Milestone 9's segmenter around Kurmann et al.'s centroid/offset approach
   (richer per-pixel supervision) rather than trying further NMS variants
   or a standard Mask R-CNN.

Net effect on Milestone 8: augmentation (both variants) and soft-NMS have
now all been tried and all returned null-or-negative results. The original
unaugmented baseline (mAP@50 0.829, mAP@50:95 0.616) remains this project's
best detection result. Not spending further Milestone-8-scoped effort
chasing detector-level fixes -- the literature and this project's own
negative results both point at Milestone 9's architecture choice as the
real next lever, not more tuning of the current detector.

## 2026-09-01 — Two more detection levers, both well-precedented in this project already

Before committing fully to Milestone 9, tried two more things that are
directly justified by this project's own prior results rather than guesses:

1. **Class-weighted box-classification loss.** torchvision's
   `RoIHeads`/`fastrcnn_loss` computes the box classifier's cross-entropy
   with no `weight` argument -- confirmed by reading the source
   (`torchvision.models.detection.roi_heads.fastrcnn_loss`). This is the
   exact same imbalance problem Milestone 5 fixed for Task A/B, just never
   applied to the detector. torchvision doesn't expose a constructor hook
   for this, so `models/detectors/weighted_loss.py::apply_class_weighted_detection_loss`
   monkeypatches the module-level `fastrcnn_loss` function (confirmed via
   `inspect` that `RoIHeads.forward` calls it as a bare name resolved at
   call time, so this works without subclassing the much larger
   `RoIHeads.forward`). Documented as a real fragility tied to torchvision's
   internal call structure, not a public API. Config: `configs/detection_weighted_loss.yaml`.
2. **ResNet-50-FPN backbone**, registered as `fasterrcnn_resnet50`
   (41.3M total params vs. the lightweight detector's 18.96M) -- the same
   "does a heavier backbone buy a real accuracy gain" question Milestone 6
   already answered yes to for classification, now asked for detection.
   Config: `configs/detection_resnet50_heavy.yaml`. Chose ResNet-50
   specifically for consistency with Task A's own heavy-baseline choice.

Both tested as single-variable changes against the original best baseline's
exact recipe (`data.augmentation: none`, Adam, early stopping) -- not
stacked with each other or with anything else -- so their effects can be
read cleanly. Both running in parallel (one per GPU).

**Also fixed while adding this**: `build_detection_transforms`'s
`augmentation` parameter previously defaulted to `"default"` (light flip +
jitter), which would have silently changed what re-running
`detection_baseline.yaml` (or the two `_augmented*` configs, which predate
`"none"/"default"/"strong"` being distinct options and used the hardcoded
strong recipe unconditionally) actually produces. Default is now `"none"`,
matching every affected config's real historical behavior; the two
`_augmented*` configs now pin `data.augmentation: strong` explicitly so
they stay reproducible if ever rerun.

## 2026-09-01 — Class-weighted loss result: small aggregate move, real per-class win

`detection_weighted_loss_20260901-001115`: mAP@50 0.831 (vs. baseline
0.829 -- essentially flat), mAP@50:95 0.596 (vs. 0.616 -- down slightly).
But per-class: **Laparoscopic Grasper AP50 0.682 -> 0.737 (+5.5pts)** --
the single weakest class across every run in this entire project,
finally moving, at the cost of small dips (0.5-1.3pts) on the common
classes. This is the expected, textbook signature of class-weighted loss
(redistributes attention from common to rare classes) working as intended
-- the aggregate barely moving hides a real, targeted improvement.
Occlusion-stratified recall unchanged (heavy: 0.617 vs. 0.620) -- confirms
again that occlusion is a structural/localization problem, not a class-
imbalance one, consistent with every other result this session.

Against the user's stated decision rule (get within ~3% of the published
literature's comparable range, currently 88-93% box AP50, or pivot toward
more capacity): 0.831 is still ~5-9 points short. Launched two follow-ups
in parallel given unlimited compute: `detection_resnet50_heavy.yaml`
(backbone alone, already running, GPU1) and a new
`detection_resnet50_weighted_loss.yaml` combining both levers (GPU0) --
mirrors Milestone 5's finding that weighted loss and a second, different
kind of intervention can compound (there: weighted loss + augmentation;
here: weighted loss + backbone capacity). If the combined run doesn't
close the gap either, per the user's explicit instruction the next move is
leaning further into capacity (less lightweight) rather than continuing to
protect the efficiency thesis at the cost of hitting the target.

## 2026-09-01 — Milestone 9 plan refined: lightweight tracking-by-detection, not a video transformer

User asked why this project uses per-frame detection instead of the
temporal/video-context approach TAPIS itself uses, and separately observed
that our detector's latency (~22.5ms/frame) still leaves headroom under a
real-time budget (33ms at 30fps) for something cheap on top. Both points
are fair and already partly covered by `docs/detection_literature_notes.md`'s
Technique 2 (González et al.'s temporal identity-preservation module) --
refined that section rather than adding something new: the actual goal
(carry an instrument's identity through a momentary occlusion using
neighboring frames) doesn't require a video transformer to get most of the
benefit. Classical tracking-by-detection (SORT/DeepSORT-class frame-to-
frame association, near-zero added compute) can do the same job as a
post-processing/tracking layer on top of the existing per-frame
detector/segmenter, without touching its architecture or per-frame
latency -- a better fit for this project's efficiency thesis than TAPIS's
video-level architecture. Added as Milestone 9 scope (not deferred to
Milestone 10+ like the full video-transformer version still is): run the
per-frame segmenter as-is, add lightweight tracking across a case's
consecutive frames, and re-check occlusion-stratified recall (the same
metric already built for Milestone 8) for tracked vs. untracked instances.

## 2026-09-01 — resnet50_heavy result: capacity-scaling hypothesis fails

`detection_resnet50_heavy.yaml` finished: **mAP@50 = 0.808, mAP@50:95 =
0.576** (41.1M params, wall clock 8473.5s / ~2.35h). This is *below* the
MobileNetV3-Large-FPN baseline (0.829-0.831), not an improvement -- a
heavier backbone overfits rather than helps on 8 train cases. Per-class
AP@50 is uneven (Monopolar Curved Scissors 0.967, Prograsp Forceps 0.651)
but the pattern matches the lighter backbone's, not a different failure
mode. Occlusion-stratified recall is essentially unchanged from every
other variant tried (isolated 0.894, light 0.923, **heavy 0.666**),
confirming again that occlusion is not fixed by adding parameters.

This closes out the "trade efficiency for accuracy via bigger backbone"
branch of the user's decision rule: it does not work on this dataset size,
so it is not the lever to keep pulling. `detection_resnet50_weighted_loss.yaml`
(combined backbone+loss, GPU0) is still running as of this entry (best
0.7936 at epoch 8, epoch 13 currently) but tracking the same losing
trajectory and is expected to land in the same place.

Per the user's own framing ("if we're not within 3%, look at reducing
efficiency to gain accuracy") -- the capacity side of that instruction has
now been tried and failed, so the next lever is the one actually backed by
a diagnosis: the occlusion recall gap, not model size. Two things kicked
off in response, in parallel with letting the combined run finish
naturally:

1. **Copy-paste augmentation** (Ghiasi et al. 2021) is no longer a
   contingency -- built and launched (`configs/detection_copy_paste.yaml`,
   `src/surgical_ai/data/copy_paste.py`, GPU1, MobileNetV3-Large-FPN
   backbone since ResNet-50 is now a ruled-out variable). Pastes real
   Clip Applier instance crops (GraSP's own per-instance RLE masks, no
   synthetic data) onto other training frames, biased 70% of the time to
   land on top of a box already in the frame -- directly targeting the
   occlusion regime where every detector variant loses ~25-30 recall
   points. Verified end-to-end against real data before launch (correct
   class labels, correct box placement, visually clean composites --
   the only artifacts are a few source-mask edge pixels landing on the
   destination frame, inherited from imprecise GT mask boundaries in the
   source annotation, not a bug in the paste logic).
2. **Milestone 9 architecture work started** (see below) -- the occlusion
   gap is a Faster R-CNN/NMS structural problem per the existing diagnosis,
   and the real fix is the anchor-free, NMS-free centroid/offset segmenter
   already scoped for Milestone 9, not another Milestone 8 ablation.

## 2026-09-01 — Milestone 9 scaffolding: centroid/offset segmenter

Started building Milestone 9 ahead of schedule (normally follows Milestone
9's tracking-by-detection work once Task B/detection are "done" per
CLAUDE.md's milestone ordering) because it is now the primary lever for
closing the box-AP50 gap, not a later-stage nice-to-have -- occlusion is
the diagnosed bottleneck and this architecture is the one built to remove
NMS-based suppression of overlapping instruments entirely.

Architecture is a practical combination of two ideas, not a reproduction
of one paper (worth being precise about, given CLAUDE.md's no-novelty-
claims rule): a semantic segmentation head (per-pixel class) plus a
centroid-heatmap + offset-regression head for instance separation.
The heatmap+offset half follows Kurmann et al. 2021's "mask then classify"
framing; the two-head combination is structurally closest to Cheng et al.
2020's Panoptic-DeepLab. Both heads are anchor-free and NMS-free.

Backbone reuses the same MobileNetV3-Large-FPN torchvision utility the
Milestone 8 baseline detector uses (`torchvision.models.detection.
backbone_utils.mobilenet_backbone`), with `returned_layers=[1, 2]` instead
of the torchvision detection default -- the default picks the two coarsest
stages (both stride 32, useless for per-pixel prediction); `[1, 2]` gives
a finest FPN output at stride 4 (96x96 for a 384 input), which is what the
heatmap/offset/semantic heads are built against. Chosen empirically by
inspecting MobileNetV3-Large's actual stage indices/strides
(`[0, 2, 4, 7, 13, 16]` -> strides `[2, 4, 8, 16, 32, 32]`), not assumed.

Heatmap peak rendering uses the standard CornerNet/CenterNet Gaussian-
radius formula (Law & Deng 2018; Zhou et al. 2019) -- cited, not derived
here. Files: `src/surgical_ai/data/segmentation_targets.py` (pure target-
generation functions), `src/surgical_ai/data/segmentation_dataset.py`
(`GraspSegmentationDataset`), `src/surgical_ai/models/segmenters/` (model
+ registry, mirroring `models/detectors/`'s pattern),
`src/surgical_ai/training/segmentation_losses.py` (focal + L1 + CE),
`src/surgical_ai/evaluation/segmentation.py` (IoU/Dice/mIoU/per-class IoU
per PROJECT_SPEC, AP50_segm/mcIoU for literature comparability, and the
existing occlusion-stratified-recall metric adapted from box-IoU to
mask-IoU), `fit_segmentation`/`train_one_epoch_segmentation`/
`run_segmentation_eval` in `training/trainer.py`, and
`run_segmentation_training` in `scripts/train.py` (`task: segmentation`).
Model is 2.1M params -- lighter than either detection variant (18.9M
MobileNetV3 Faster R-CNN, 41.1M ResNet-50 Faster R-CNN), consistent with
the efficiency thesis.

Caught one real bug before it reached a training run: the heatmap head's
final conv layer was bias-initialized to -2.19 (CenterNet's
sigmoid(-2.19)~=0.1 convention) but not weight-zeroed, so the *weights*
still dominated the output at init -- random feature activations pushed
some pixels' logits to saturation, and the focal loss's `log(1-pred)` term
exploded (heatmap loss ~1691 on a real batch instead of the expected
low-tens). Fixed with RetinaNet/CenterNet's standard zero-weight-init on
that one layer (bias-only controls the initial output); heatmap loss
dropped to ~25 on the same batch, matching a hand-computed expected value
for this loss formula's known "small positive count, many negatives"
math -- not further reduced, since that number is inherent to the formula
at this heatmap resolution/instance-count ratio, not a remaining bug.

Verified with a real (not synthetic) end-to-end run before calling this
scaffolded: 16 real GraSP frames, 2 epochs, dev laptop RTX 4060 -- loss
dropped every epoch (heatmap 31.8->17.9, val mIoU 0.015->0.054), and the
full manifest/checkpoint/metrics-table path that Milestone 8's runs use
worked unmodified. All 52 tests pass (`pytest tests/`), including 6 new
test files covering target generation, losses, dataset, and evaluation.

Coarse-resolution caveat, stated once here rather than only buried in
module docstrings: all instance-level segmentation metrics (AP50_segm,
mask-IoU occlusion recall) currently compare masks at the model's output
stride (96x96 for a 384 input), not upsampled to input resolution. This is
consistent internally (predictions and downsampled ground truth are
compared at the same resolution) but understates true mask quality
somewhat -- upsampling predictions for a final, reportable number is
follow-up work, flagged here so it isn't mistaken for a finished number
later.

`configs/segmentation_baseline.yaml` is ready to launch (MobileNetV3-
Large-FPN-based centroid/offset model, image_size 384, output_stride 4,
batch_size 8, patience 10) -- queued to launch on whichever GPU frees up
first among the two still-running Milestone 8 follow-ups.

## 2026-09-01 — copy-paste augmentation interim result: closest variant to baseline yet

As of epoch 18 (still running, GPU1): best val mAP@50 = **0.8264** at epoch
11 (mAP@50:95 = 0.5897 same epoch), only ~0.3-0.5 points below the
0.829-0.831 baseline and clearly the best-performing Milestone 8 follow-up
tried so far -- better than ResNet-50 alone (0.808, finished) and better
than the combined ResNet-50+weighted-loss run (0.7998 at epoch 18, still
running, still improving intermittently so not final). Consistent with the
diagnosis this was built against: unlike a backbone swap, copy-paste
directly increases occlusion exposure and Clip Applier representation
during training rather than just adding parameters.

Not yet conclusive -- still needs to finish (patience 10 from its epoch-11
best, so due to stop within a few more epochs unless it improves again)
and needs its own occlusion-stratified recall table checked once done, to
confirm the mechanism (closing the heavy-occlusion recall gap
specifically) rather than just a generic aggregate wobble. Will log the
final number and occlusion breakdown once it stops.

## 2026-09-01 — copy-paste final result: closes the mAP gap, does NOT touch occlusion

Finished (early-stopped at epoch 21, patience 10 from its epoch-11 best).
**mAP@50 = 0.826, mAP@50:95 = 0.590**, wall clock 2957.9s (~49min, 18.9M
params, same MobileNetV3-Large-FPN as the original baseline). This
essentially matches the 0.829-0.831 baseline (within noise) and is by far
the best Milestone 8 follow-up tried -- confirms augmentation targeted at
the actual diagnosed problems (imbalance + occlusion exposure) works where
generic augmentation (`strong`/`default`, both regressed mAP) and generic
capacity (ResNet-50, regressed mAP) did not.

Per-class AP@50: Bipolar Forceps 0.940, Prograsp Forceps 0.714, Large
Needle Driver 0.947, Monopolar Curved Scissors 0.968, Suction Instrument
0.741, **Clip Applier 0.780**, Laparoscopic Grasper 0.696 -- Clip Applier
(the pasted rare class) lands respectably mid-pack, not the weakest class
anymore.

**But the occlusion-stratified recall table is the real finding here**:
isolated 0.899, light 0.920, **heavy 0.645** -- statistically
indistinguishable from every other variant this session (0.617-0.666
heavy-overlap recall across baseline, augmented, weighted-loss, and both
ResNet-50 runs). Despite `occlusion_bias=0.7` deliberately pasting
instances on top of existing boxes, heavy occlusion recall did not move.

This is an important negative result on *mechanism*, not just another
data point: it says the occlusion gap is not a training-distribution
problem (not enough occluded examples to learn from) but a
representation/localization problem in Faster R-CNN's box-proposal +
NMS pipeline itself -- pasting more occluded training examples doesn't
teach the model to separate two overlapping boxes better, because the
architecture's proposal and suppression mechanism is what's failing, not
its exposure to occlusion during training. This is exactly the
architectural failure mode Milestone 9's anchor-free, NMS-free
centroid/offset segmenter was built to avoid (see above), and this result
is the strongest evidence yet that Milestone 9 -- not another Milestone 8
data/loss ablation -- is the right lever left to pull. GPU1 freed;
launching `configs/segmentation_baseline.yaml` there now.

## 2026-09-01 — Milestone 9 first run: good pixel-level result, broken instance decoding (diagnosed and fixed)

`segmentation_baseline.yaml` finished (early-stopped at epoch 49, patience
10 from its epoch-39 best). Semantic head result is a genuinely reasonable
first number: **mIoU = 0.308**, mean Dice = 0.448, 32 min wall clock,
2.1M params. Per-class IoU tracks the same difficulty ordering detection
sees (Monopolar Curved Scissors 0.651 IoU, easiest; Prograsp Forceps 0.140,
hardest) -- the model is learning real, sensible per-pixel structure.

But the instance-level numbers were close to broken: **AP50_segm = 0.0017**,
occlusion-stratified recall 0.087/0.062/0.094 (isolated/light/heavy) --
uniformly near-zero and not even ordered by occlusion level, so this
number said nothing about whether the architecture fixes occlusion. Did
not accept this at face value -- diagnosed before writing it off or
reporting it as a real result.

**Diagnosis** (on the actual saved checkpoint, real val images): heatmap
peak-finding at the old defaults (`score_threshold=0.1`, `nms_kernel=3`,
both copied from typical CenterNet/COCO settings) found **14-41 candidate
peaks per image against 1-4 real ground-truth instances**. A 3x3 max-pool
NMS kernel only correctly isolates one peak per object once the heatmap
has trained down to a single sharp unimodal bump at that location; at 49
epochs the heatmap head hadn't gotten there yet (still visibly improving:
heatmap loss 0.42 -> 0.23 in the run's last 10 epochs, i.e. still
converging when semantic mIoU-based early stopping fired), so each
instrument's still-somewhat-diffuse probability mass registered as many
separate small local maxima -- one real instance fragmented into dozens of
spurious tiny "detections," none with enough mask-IoU overlap with the
true instance to count as a match. Confirmed this was a decode-time
tuning problem, not a broken representation, by sweeping
(score_threshold, nms_kernel) on 200 real val images against the *same*
checkpoint: `(0.1, 3)` -> map50=0.0027, avg 21.6 predicted instances/image
vs. 2.6 real; `(0.3, 7)` -> map50=0.1036 (~38x higher), avg 2.7
predicted/image vs. 2.6 real -- predicted instance count nearly matches
ground truth once the kernel/threshold are sane for this checkpoint's
actual heatmap sharpness.

**Fixed**: `decode_instances`'s defaults changed to `score_threshold=0.3,
nms_kernel=7` (`evaluation/segmentation.py`), tuned empirically against
real predictions, not assumed from COCO-scale CenterNet conventions --
this project's output grid (96x96 for a 384 input) is much coarser than
COCO-scale CenterNet's typical 128x128+, so a proportionally smaller
kernel makes sense even before considering the undertrained-heatmap
effect. Also launched `configs/segmentation_extended_patience.yaml`
(patience 30, epochs ceiling 150) so the heatmap head gets room to keep
sharpening past the point semantic mIoU alone looks converged, rather than
stopping on a checkpoint-selection metric that doesn't reflect what the
instance-decoding heads actually need. AP50_segm=0.10 on the old
checkpoint (with fixed decode defaults) is a real if modest number, not
production-quality -- expecting the extended-patience run to do
meaningfully better once the heatmap head actually converges, and that is
the number to trust for Milestone 9's first real comparison point, not
this one.

## 2026-09-01 — extended-patience result: real progress, occlusion hypothesis not yet testable

Finished (early-stopped at epoch 125, patience 30 from its epoch-95 best).
**mIoU = 0.319**, mean Dice = 0.460 (up slightly from 0.308/0.448), wall
clock 4422.5s (~74min). Heatmap loss converged much further than the
first run: 0.016-0.11 by the end vs. 0.42 where the first run's patience-10
stopping cut it off -- confirms the extended patience was the right fix
for letting the heatmap head actually sharpen.

**AP50_segm = 0.1105** (vs. ~0.10 on the old checkpoint with corrected
decode settings -- a small further improvement, consistent with the
sharper heatmap). Occlusion-stratified recall is now genuinely
differentiated rather than flat noise (a real result, unlike the first
run's broken 0.087/0.062/0.094): **isolated 0.334, light 0.399, heavy
0.105**.

**Honest reading of this, not spun either way**: this is real, working
progress -- loss is behaving correctly, mIoU is meaningful, occlusion
recall is no longer a broken flat line. But it does **not yet answer the
question this architecture was built to test**. Two problems with treating
this as a result on the occlusion hypothesis:

1. Absolute recall (isolated 0.334) is far below Faster R-CNN's
   equivalent (isolated ~0.88-0.90 across every Milestone 8 variant) --
   this model is still a long way from comparable overall maturity, on
   only 125 epochs of a 2.1M-param model trained from scratch overnight.
2. The *relative* occlusion penalty is actually worse than Faster R-CNN's
   right now: heavy/isolated ratio = 0.105/0.334 = 0.31 here vs.
   0.645/0.899 = 0.72 for copy-paste's Faster R-CNN (i.e. Milestone 9
   currently loses a *larger* fraction of its recall to occlusion, not a
   smaller one) -- the opposite of the hypothesis. This is very unlikely
   to be a fundamental property of the architecture at this stage; an
   undertrained centroid/offset model plausibly struggles most exactly
   where two predicted centroids sit close together and pixel-to-centroid
   assignment is most ambiguous, which is also the heavy-occlusion regime
   -- but this cannot be told apart from "just needs more training" using
   one 125-epoch run.

**Conclusion, not yet a verdict**: Milestone 9 needs meaningfully more
training/tuning before the occlusion hypothesis can be fairly tested.
Launching one more follow-up (single new variable: cosine LR decay,
`configs/segmentation_cosine_lr.yaml`) to address a real, visible symptom
in the training curve -- periodic heatmap-loss spikes (e.g. epoch 65, 73,
120 all jump ~0.5 then recover within 1-2 epochs) consistent with a fixed
LR=0.0001 occasionally destabilizing Adam once the loss is already small
-- with a much higher patience/epoch ceiling given epochs are cheap
(~35-40s each) and compute is not the constraint here.

## 2026-09-01 — copy-paste ported to segmentation

Per user direction after reviewing the overnight results ("work on seeing
what we can do to get closer to the literature numbers"): ported copy-paste
augmentation to the Milestone 9 pipeline (`data/segmentation_copy_paste.py`),
since it was the single most effective Milestone 8 lever and none of the
three segmentation runs so far (baseline, extended-patience, cosine-LR)
ever increased occlusion exposure or Clip Applier representation during
training -- the plateau documented below may simply reflect that gap, not
a hard capacity ceiling.

Refactored `data/copy_paste.py`'s bank-building/patch-loading/placement/
compositing helpers into module-level functions (previously
`CopyPasteDetectionDataset` methods) so both the detection and
segmentation copy-paste datasets share them -- two concrete users now
justifies the extraction. Unlike detection, pasting here must also update
the dense per-pixel targets (heatmap, offset, semantic), not just a
box+label list; compositing happens at native frame resolution before the
resize-to-`image_size` step, so the pasted patch is resized along with
everything else exactly like a real instance. Verified end-to-end
(bank contents, a real composited visual check, and a full
dataset->model->loss->backward dry run) before launching.

`configs/segmentation_copy_paste.yaml` combines copy-paste with the
already-best-converged training recipe (cosine LR, patience 30) rather
than testing copy-paste in isolation against the plain baseline --
justified here (deviating from strict one-variable-per-run) because the
cosine-LR recipe is now the established best foundation, not an
open question. Running on GPU0.

## 2026-09-01 — segmentation copy-paste result: small real gain, same plateau

Finished (early-stopped at epoch 102, patience 30 from its epoch-72 best).
**mIoU = 0.309**, mean Dice = 0.445 -- essentially flat, slightly *below*
the plain cosine-LR run's 0.321 (within run-to-run noise for this
architecture). **AP50_segm = 0.1216 -- the best of all four segmentation
runs so far** (0.0017 broken-decode, 0.1105 extended-patience, 0.0982
cosine-LR, now 0.1216). Occlusion-stratified recall: isolated 0.339, light
0.411, **heavy 0.087** -- a modest absolute improvement across every
bucket vs. the plain cosine-LR run (0.323/0.396/0.073), but the
heavy/isolated ratio (0.087/0.339 = 0.26) still sits in the same 0.23-0.31
band every segmentation run has landed in. Clip Applier IoU ticked up
slightly (0.205 vs. 0.195-0.208 in prior runs).

**Reading this honestly**: copy-paste gave a small, real improvement (best
AP50_segm yet, better absolute recall everywhere) but did not break the
plateau or meaningfully close the relative-occlusion-penalty gap to
Faster R-CNN. This is a different outcome than detection's copy-paste
result, where it was decisively the best lever tried. The likely reason:
detection's bottleneck was diagnosed as proposal/NMS-pipeline-specific
(soft-NMS null result, capacity-scaling null result both pointed away
from "needs more data/capacity" and toward "the architecture mishandles
occlusion structurally") -- copy-paste worked there because it's a data
intervention for a data-shaped problem underneath an otherwise-mature
model. Segmentation's bottleneck looks different: four independent runs
(default patience, extended patience, cosine LR, copy-paste) spanning
three different single-variable interventions have now converged to
essentially the same mIoU/occlusion-ratio plateau, which is much more
consistent with a **capacity or training-maturity ceiling** (a 2.1M-param
model, first-ever training runs for this from-scratch architecture) than
a training-distribution problem copy-paste-style data intervention can fix.

**Recommendation for where to look next, given the user's actual goal
("get closer to the literature numbers")**: detection is the more mature,
more literature-comparable path (0.831 mAP@50, 5-9 points short) and has
had every conventional lever exhausted; segmentation has real, working
infrastructure but needs a capacity increase (not another training-recipe
tweak) to fairly test its core hypothesis, which is a bigger, riskier next
step than what's been tried so far. Flagging this choice to the user
rather than unilaterally picking one, since it's a real fork in scope, not
a single-variable ablation. User chose tracking-by-detection.

## 2026-09-01 — tracking-by-detection: real positive result, with a causality caveat

Checked frame spacing before building anything: annotated frames within a
case are ~35 frames apart (median), far too sparse for frame-to-frame
tracking directly. But the raw frame directories contain every
intermediate frame at native (~1fps) sampling (e.g. CASE001 has 10,972
sequential files, not just its 329 annotated ones) -- so real tracking is
possible by running the detector across a window of *raw* frames around
each annotated frame.

Built an IOU-Tracker (Bochinski et al. 2017 -- greedy box-IoU association,
no motion model, no appearance features; the only addition on top of the
original paper is track survival through up to `max_age` consecutive
missed frames, borrowed from SORT/DeepSORT's own convention, needed for
the track to actually "survive" momentary occlusion rather than ending
the instant one frame is missed) in `inference/tracking.py`, and a
windowed-inference pipeline (`inference/pipeline.py`,
`scripts/evaluate_tracking.py`) that runs the copy-paste detector across
each annotated val frame's raw-frame window, tracks through it, and
compares occlusion-stratified recall between the raw per-frame detections
and the tracker's state at that frame. Verified with real trained-model
smoke tests before the full run.

**First result, window_radius=5, max_age=3, 1125 val frames**: recall
improved in every bucket -- isolated 0.899->0.913, light 0.920->0.930,
**heavy 0.645->0.683**. This is the first technique all session to move
heavy-occlusion recall outside the 0.606-0.666 band every other
Milestone 8 variant landed in, and the largest absolute gain is in the
heavy bucket specifically.

**Checked before reporting it as a clean win, not just assumed**: the
window this run used looked both backward and forward in time around the
annotated frame, which would be invalid for a real-time claim (a live
system can't see future frames). Added a `causal` (backward-only) window
mode and reran with `window_radius=10, causal=True` -- same look-back
budget, no look-ahead -- expecting a possibly-weaker but honest number.

**Result: identical numbers** (isolated 0.912, light 0.930, heavy 0.683
-- all within rounding of the first run). Traced this to the
implementation, not luck: `run_window_and_track` reads out the tracker's
state the instant it processes the target frame in the loop, *before* any
later frames are fed to the tracker -- so the original window's forward-
looking frames were computed but never actually influenced the result,
just wasted GPU time running them. There was no look-ahead leak to begin
with; the check was still worth doing empirically rather than trusting
that reading of the code. Fixed `evaluate_tracking.py`'s default to a
causal-only window (no reason to compute the wasted forward frames), so
**this result is real, causal, and stands as a legitimate real-time-valid
finding**: the first thing all session to move heavy-occlusion recall
outside the flat band, via a near-zero-added-compute post-processing step
on the existing, already-best detector.

## 2026-09-01 — tracking-by-detection, corrected: it hurt mAP@50, root-caused, fixed

User asked to push tracking-by-detection further and get the actual
mAP@50 effect, not just occlusion-stratified recall. Added full COCO-style
mAP computation to `evaluate_tracking.py` (needs a low score threshold for
a real precision-recall curve, unlike the fixed-0.5-threshold recall
table) and ran it plus a small tracker hyperparameter sweep
(`max_age` in {3, 6}, `iou_threshold` in {0.3, 0.2}, `window_radius=10`)
on both GPUs.

**Result, both configs**: mAP@50 got *worse* with tracking, not better --
0.7899-0.7900 tracked vs. 0.8264 raw, a ~3.7 point drop -- while occlusion
recall barely moved this time (heavy 0.645->0.652-0.655, far short of the
0.683 first reported). This was a real, important correction to make
before reporting the earlier result as a clean win.

**Root cause**: the low score threshold (0.05, needed for a real AP
curve) let the tracker treat far more low-confidence detections as track
material, and every unmatched-but-still-alive track (regardless of how it
was born) was allowed to coast forward with its last known box. Two
failure modes followed: (1) tracks born from low-confidence noise
detections coasting forward as confident-looking false positives, and
(2) tracks from real objects that *actually left the frame or moved far*
(not momentarily occluded) still coasting on a now-stale, wrong box --
both add false positives without adding real recall, which is exactly
what tanks precision (and therefore AP) even while a fixed-threshold
recall table might look flat or slightly improved.

**Fix**: added `min_confidence_to_coast` to `IOUTracker`
(`inference/tracking.py`) -- a track can only be *reported* while
coasting if its last real detection scored at least this threshold
(default 0.5); it still stays alive internally and can be re-matched
later, it just isn't trusted to extrapolate a guess into the output while
unconfirmed. This is a precision/recall tradeoff to tune, not a free
win -- flagged as such in the code, not oversold. Verified with new unit
tests (a low-confidence track's coast is suppressed from output but the
track itself survives to re-match later) before rerunning at scale.
Rerunning the full evaluation with the fix now; result pending as of this
entry.

This is exactly the kind of correction worth recording plainly: the first
version of this result was reported to the user as a clean, verified win
(after checking causality) -- and it still needed a second, more
demanding check (the actual literature-comparable metric, not just the
diagnostic one) to catch that it was net negative. Both checks were
worth doing; neither alone would have caught the full picture.

**`min_confidence_to_coast=0.5` alone barely moved it**: mAP@50 = 0.7914
(vs. 0.7899-0.7900 uncorrected) -- confirms low-confidence-noise tracks
coasting were not the dominant failure mode.

## 2026-09-01 — boundary-exit detection (user's proposal), combined with the confidence fix

User proposed the other half of the fix directly: track a coarse motion
vector (box-center delta between real matches) and, when a track goes
unmatched, check whether it was near a frame edge *and* moving further
toward that edge -- if so, treat the gap as the instrument genuinely
leaving the frame and drop the track immediately instead of coasting,
rather than treating every gap as possible occlusion. This targets
exactly the failure mode the confidence fix didn't: a real,
confidently-tracked instrument that actually exits frame still produces a
stale, wrong, confident-looking coasted box if nothing distinguishes
"gone" from "occluded".

Implemented in `inference/tracking.py`: `Track` now carries a `velocity`
(box-center delta from the last real match, `(0,0)` until a track has
matched twice), and a new `is_exiting_frame(box, velocity, frame_width,
frame_height, margin_frac)` helper checks proximity to each of the four
edges against the corresponding velocity component. `IOUTracker` takes
`frame_width`/`frame_height` (read from the first frame in each window,
`pipeline.py`) and `boundary_margin_frac` (default 0.05); an exiting track
is dropped immediately (removed from `self.tracks`, not just unreported)
rather than going through the normal `max_age` coasting path. No learned
parameters, no detector changes -- pure post-processing, same efficiency
profile as the rest of the tracker. 5 new unit tests added (exit-detection
logic in isolation, plus end-to-end tracker behavior: an edge-approaching,
outward-moving track is dropped on its very first gap; a track missing
away from any edge still coasts normally) -- all pass, and all prior
tracking tests still pass unmodified (they only ever have zero or one
prior match before testing gap behavior, so velocity stays at its
default `(0,0)` and never trips the new check, confirmed by inspection
before running, not just by the tests happening to pass).

Running with both fixes combined (`min_confidence_to_coast=0.5`,
`boundary_margin_frac=0.05`) against the same copy-paste checkpoint and
window settings as every prior tracking result, for a clean comparison.

**Result: mAP@50 = 0.8034** (vs. 0.7899 uncorrected, 0.7914 confidence-fix
alone) -- the boundary-exit check did almost all of the recovery (0.012
of the 0.0135 total gain over confidence-gating alone), confirming the
user's diagnosis: the dominant failure mode was confidently-tracked
instruments going stale on real frame-exit, not low-confidence noise.
Mechanistically clean too: occlusion recall barely moved (heavy 0.648 vs.
raw's 0.645), meaning the fix worked by removing false positives
(correctly dropping exiting tracks), not by adding new true positives --
exactly what a precision-focused fix should do. Still 2.3 points below
raw (0.8264), so tracking remains net negative on the headline metric at
this point.

**Follow-up sweep on `max_age`** (1, 2, 3, everything else held fixed) to
test whether shortening the coasting window reduces the remaining
staleness (an occluded instrument can drift during the gap without ever
nearing a frame edge, which the boundary-exit check can't catch):

| max_age | mAP@50 | gap to raw (0.8264) | heavy-occlusion recall |
|---|---|---|---|
| 3 | 0.8034 | -2.30 | 0.648 |
| 2 | 0.8075 | -1.89 | 0.648 |
| **1** | **0.8130** | **-1.34** | 0.652 |

Clean monotonic trend: shorter coasting window -> less staleness -> less
precision loss, closing more of the gap at every step. `max_age=1` is
close to the practical floor (anything lower disables occlusion tolerance
entirely, which defeats the point of tracking at all) and is the best
result found in this sweep.

## 2026-09-01 — occlusion-corridor mechanism (user's second proposal)

User proposed the positive counterpart to boundary-exit's negative
signal: instead of only detecting "this track is definitely leaving the
frame," detect "this track is plausibly occluded right now." On the
first missed frame of a gap, extrapolate the track's box one step forward
by its velocity and check whether that predicted position overlaps
another instrument actually detected this frame -- if so, treat the gap
as corroborated occlusion and grant a longer `occluded_max_age` instead
of the normal (now short, `max_age=1`) lifetime for the rest of the gap.
A track with no such evidence keeps the normal short lifetime.
Deliberately asymmetric: extended trust requires positive geometric
evidence of an occluder, not just the absence of a boundary-exit signal.

Implemented in `inference/tracking.py`: `extrapolate_box` (one-step
linear shift by velocity, no acceleration term -- simplest possible
motion model, no learned parameters) and `has_plausible_occluder` (IoU
between the predicted box and any of this frame's real detections,
regardless of class -- an occluder is whatever is physically in the way).
`Track` gained `likely_occluded: bool`, decided once per gap (at the
first miss) and reset on the next real match. `IOUTracker` gained
`occlusion_corridor_iou_threshold` (0 disables, matching the
`boundary_margin_frac=0` convention) and `occluded_max_age`. 5 new unit
tests (extrapolation math, occluder-detection in isolation, and two
end-to-end tracker scenarios: a gap with a corroborating occluder present
survives past what the short `max_age` alone would allow; a gap with
nothing nearby gets dropped at the normal short lifetime) -- all pass,
plus all 15 prior tracking tests still pass unmodified (the feature
defaults to disabled, `occlusion_corridor_iou_threshold=0`).

Running against the best configuration found so far
(`max_age=1, min_confidence_to_coast=0.5, boundary_margin_frac=0.05`)
plus `occlusion_corridor_iou_threshold=0.1, occluded_max_age=5` -- testing
whether corroborated occlusion can safely earn back some of the recall
that the very short `max_age=1` leaves on the table, without reintroducing
the staleness cost that made longer max_age net-negative on its own.

**Result: mAP@50 = 0.8048** -- worse than plain `max_age=1` alone (0.8130),
and occlusion recall barely moved (light 0.925 vs. 0.923, heavy 0.648 vs.
0.652 -- essentially flat or marginally worse). One-shot evidence at gap
start, checked only once, is not enough to justify 5 frames of pure
extrapolation: confirming an occluder was nearby *when the gap started*
says nothing about whether the missing instrument's real position stays
close to the stale coasted box for the following several seconds, and by
the time it reappears the coast is often just as wrong as an unconditional
one, only with an extra layer of justification that doesn't hold up.

**Two follow-ups launched to test that specific diagnosis**: (1)
`occluded_max_age=2` (much smaller, more cautious extension, same one-shot
evidence check) and (2) `require_continuous_occlusion_evidence=True` with
`occluded_max_age=5` -- re-checks the corridor condition on *every* missed
frame rather than only the first, so extended trust requires evidence to
keep holding up, not just have held once. Implemented in
`inference/tracking.py` (`Track.likely_occluded` is now re-decided each
frame when this flag is set, using that frame's own detections; matches
the same pattern `boundary_margin_frac=0`/`occlusion_corridor_iou_
threshold=0` use for opt-in/out). 2 new unit tests (evidence
disappearing mid-gap causes an immediate drop once the normal short
`max_age` is already exceeded; evidence persisting every frame keeps the
track alive) -- all pass, plus all 20 prior tracking tests unmodified.
Both variants running in parallel.

**Results**: `occluded_max_age=2` (one-shot) -> mAP@50 = 0.8089.
`require_continuous_occlusion_evidence=True, occluded_max_age=5` -> mAP@50
= 0.8056. **Both still underperform plain `max_age=1` with no corridor
mechanism at all (0.8130)** -- alongside the original `occluded_max_age=5`
one-shot result (0.8048), that's three corridor variants tried (one-shot
short extension, one-shot long extension, continuous-evidence long
extension), all consistently worse than the simple baseline.

**Conclusion on occlusion-corridor specifically**: not a noisy result --
consistent across extension length and evidence-checking strategy. Most
likely explanation: GraSP frames routinely contain 2-3 co-occurring
instruments (94.7% of frames per the dataset's own documented
distribution), so "another instrument's box overlaps the predicted
position" is true almost constantly in this dataset, occlusion or not --
the geometric proximity signal is plausible in principle but too weakly
discriminating here to earn back the staleness cost of any tested
extension. **Plain `max_age=1` (confidence-gating + boundary-exit, no
corridor) remains the best tracking configuration found, at mAP@50 =
0.8130, still 1.34 points below the raw detector (0.8264).**

This closes out the tracking-by-detection hyperparameter/mechanism
exploration for now: two real bugs found and fixed (low-confidence
coasting, boundary-exit staleness -- both genuine improvements over the
naive baseline), one mechanism tried and found not to help in this
specific dataset (occlusion-corridor, three variants). The honest overall
verdict is unchanged from before this round of testing: tracking provides
a real, validated recall benefit at a fixed diagnostic threshold, but has
not yet been shown to improve the literature-comparable headline metric.

## 2026-09-01 — Milestone 9 real capacity bug found and fixed: deep-backbone variant

User pushed to scope what a serious segmentation capacity increase would
look like, given four prior runs (default, extended patience, cosine LR,
copy-paste) all converged to the same ~0.31-0.32 mIoU plateau. Inspecting
the actual architecture (`models/segmenters/centroid_offset.py`) found a
real, previously undiagnosed bug rather than just "needs to be bigger":
`returned_layers=[1, 2]` was chosen to get a fine stride-4 FPN output
cheaply, but `mobilenet_backbone`/`IntermediateLayerGetter` **prunes the
network to stop at the deepest requested stage** -- requesting only
shallow stages (stride 4 and 8, 24 and 40 channels) meant MobileNetV3-
Large's entire deep half, up to its 960-channel final stage where most of
its actual semantic capacity lives, was **never computed at all**, not
frozen, not present in the model. Every one of the four plateaued runs
was training a model that literally could not see deep features. This
fully explains the plateau being a capacity ceiling, not a training
problem -- confirmed by gradient inspection (`model.backbone`'s deepest
layers had zero graph presence under the old `returned_layers`).

**Fix**: added `centroid_offset_mobilenet_v3_deep`
(`returned_layers=[1, 3, 5]`, spanning stride 4/16/32 at 24/80/960
channels) as a second registered variant, keeping the original
`centroid_offset_mobilenet_v3` registered unchanged so its four existing
runs stay reproducible. This is how FPN is actually meant to be used --
shallow-through-deep inputs, with the top-down pathway injecting real
semantic depth into the fine-resolution output, instead of the fine
output only ever seeing shallow, low-capacity features. Backbone grows
from ~1.9M to ~5.0M params, full model from 2.1M to 5.88M -- still far
lighter than the 18.9M-param detector, efficiency thesis intact. Verified
end-to-end (shape checks, a real forward+backward pass confirming
gradients now reach the deep backbone layers, and a full dataset->model->
train->eval->manifest dry run) before launching.

**Early result, deep-backbone-alone
(`configs/segmentation_deep_backbone.yaml`, same recipe as
`segmentation_cosine_lr.yaml` -- cosine LR, patience 40 -- only the
backbone changed, one variable at a time)**: **val mIoU = 0.5163 by epoch
5**, already 60%+ above the shallow variant's entire plateau (~0.32
reached only after 70-100+ epochs across four separate runs). This is a
dramatic, unambiguous confirmation that the shallow-features bug was the
real ceiling, not training recipe or data. Still running -- will continue
to its natural patience-based stopping point.

Also launched, in parallel, `configs/segmentation_deep_backbone_copy_paste.yaml`
(deep backbone + copy-paste augmentation combined) to test whether the
two independently-validated fixes compound, matching Milestone 5's
established pattern of combining two orthogonal interventions. Both
runs' final numbers, plus full instance-level metrics (AP50_segm,
occlusion-stratified recall) once each finishes, to follow.

**Honest conclusion after two real bug fixes and a hyperparameter sweep**:
tracking-by-detection, even at its best-tuned configuration found so far
(max_age=1, min_confidence_to_coast=0.5, boundary_margin_frac=0.05), is
**still net negative on mAP@50** -- 1.34 points below the raw detector.
It does provide a real, validated recall benefit (heavy-occlusion recall
0.652 vs. 0.645), and the two failure modes diagnosed and fixed along the
way (low-confidence-noise coasting; confident-track staleness on real
frame-exit) were both real, not imagined. But it has not yet cleared the
bar of improving the actual literature-comparable headline number. Not
reported as a finished win -- this is the honest state as of this entry,
pending a decision on whether to keep iterating (e.g. a per-instance
occlusion signal rather than a purely geometric one) or treat this as a
validated-but-not-yet-net-positive research direction and move on.

## 2026-09-01 — cosine-LR result: stabilized training, did not move the plateau -- stopping the hyperparameter chase here

Finished (early-stopped at epoch 112, patience 40 from its epoch-72 best).
**mIoU = 0.321**, mean Dice = 0.460 -- statistically flat vs. the
extended-patience run's 0.319/0.460, and the periodic heatmap-loss spikes
are gone (max bump in the whole tail is 0.06, vs. ~0.5 before), confirming
the cosine schedule did fix the specific instability it targeted. But
**AP50_segm = 0.0982** (vs. 0.1105) and occlusion-stratified recall
**(isolated 0.323, light 0.396, heavy 0.073)** did not improve --
heavy/isolated ratio is 0.073/0.323 = 0.23, if anything slightly worse
than the previous run's 0.31.

**Three independent runs (patience 10 -> 30 -> 40, plus a stabilized LR
schedule) have now converged to the same plateau**: mIoU ~0.31-0.32,
AP50_segm ~0.10-0.11, heavy-occlusion recall ~0.07-0.11. This is no longer
readable as "just needs more training" -- training is stable and
converged (loss curves flat, no more spikes) and further tuning of
training hygiene (patience, LR schedule) gave no further movement. The
ceiling here is more likely capacity or architecture design (2.1M params,
a single stride-4 feature map, no multi-scale fusion beyond the FPN's two
levels) than an optimization problem reachable by more epochs or a better
schedule.

**Stopping the Milestone 9 hyperparameter chase here for tonight.**
Diminishing returns from further one-variable-at-a-time tuning, and the
honest, current state is worth reporting plainly rather than chasing
another small variant: Milestone 9 is a real, working, anchor-free/
NMS-free architecture, correctly implemented and validated end-to-end
(dataset, targets, model, loss, decoding, evaluation all tested and one
real bug already caught and fixed), but at this capacity/training budget
it does **not yet** outperform Faster R-CNN's occlusion handling --
its absolute recall is lower across every occlusion bucket, and its
*relative* occlusion penalty is currently larger (23-31%) than
copy-paste's Faster R-CNN (72%), the opposite of the hypothesis it was
built to test. The architecture direction (removing NMS from the failure
path) remains well-motivated by the literature and by Milestone 8's
mechanism diagnosis, but confirming it actually helps here needs either
more model capacity, more training data exposure (the copy-paste
augmentation idea could be ported to this dataset pipeline too, not yet
tried here), or a longer training budget than one overnight session
allows for a from-scratch architecture. This is a legitimate foundation
for continued work, not a finished result -- see `docs/findings.md`'s
Milestone 9 section and "What's not done yet" for the precise, current
state to report to the user.

## 2026-09-01 — resnet50_weighted_loss final result: closes out the capacity-scaling branch completely

Finished: **mAP@50 = 0.800, mAP@50:95 = 0.587**, wall clock 13233.2s
(~3.67h, 41.1M params). Below the baseline (0.829-0.831), below
copy-paste (0.826), and even slightly below ResNet-50 alone (0.808) --
combining weighted loss with a heavier backbone did not help beyond the
backbone alone. Occlusion-stratified recall (heavy: 0.634) sits in the
same 0.62-0.67 band every single variant this session has landed in
(baseline, augmented, weighted-loss, ResNet-50 alone, ResNet-50+weighted,
copy-paste), regardless of backbone size, loss weighting, or training
data composition.

This is the last data point needed to close the capacity-scaling branch
of the user's decision rule completely: across every combination of
{MobileNetV3, ResNet-50} x {plain, weighted loss}, more capacity never
beat the lightweight baseline, and occlusion recall never moved outside
that narrow band regardless of what changed on the Faster R-CNN side.
The only thing that has moved the occlusion needle at all in this
session's diagnostic testing is decode-time tuning on the architecturally
different Milestone 9 segmenter (still in progress, extended-patience run
launched on the now-free GPU1). Both GPUs are free of Milestone 8 work as
of this entry.

## 2026-09-01 — Every run to date validated on official test, not a held-out set

Asked directly whether metrics were measured on test or validation data.
Checked `splits.py`: `_TRAIN_VAL_SPLITS["official"] = ("train", "test")`.
Checked every config in `configs/` (26 files) -- all use `split: official`,
zero exceptions. This means every "val_loss"/"val_mIoU" logged during
training, every early-stopping and checkpoint-selection decision, and
every headline metric reported so far (Task A/B, detection, tracking,
segmentation) was measured directly on the official 5-case test split.
There is no held-out validation set distinct from test anywhere in this
project's history to date.

This is not label leakage (test labels never enter the loss), but it is
model-selection leakage: early stopping picks the checkpoint that scores
best on test, and every comparative call this session made (copy-paste
vs. plain augmentation, the backbone-depth fix, the tracking `max_age`
sweep) was a judgment made by watching test-set performance move. That
inflates every reported number's credibility as a true generalization
estimate, by an unquantified amount. Never flagged anywhere before this
entry -- not in this file, `PROJECT_SPEC.md`, or `findings.md`.

Fix costs nothing to build: `docs/dataset_report.md` already confirms
fold1 (CASE001, CASE004, CASE014, CASE015) and fold2 (CASE002, CASE003,
CASE007, CASE021) exactly partition the 8 official train cases, fully
disjoint from the 5 official test cases. `splits.py` already resolves
`fold1` to `("fold2", "fold1")` -- train on fold2's 4 cases, validate on
fold1's 4 cases, official test never touched. This has existed since
Milestone 2 and was simply never used; every config defaulted to
`official` instead.

Decision, going forward only (all completed results stand as reported,
now with this caveat documented rather than silently retracted or
re-run): new development work (model selection, hyperparameter
comparison, early stopping) uses `split: fold1` or `fold2`, not
`official`. Once a configuration is locked in, one final confirmatory
run with `split: official` reports the number that goes in the report,
untouched by any prior decision. Not applied retroactively -- rerunning
this session's ~20 completed ablations under a fold split would cost
GPU time disproportionate to the value, since the qualitative
conclusions (occlusion-recall ceiling, capacity-scaling result,
tracking's net-negative mAP) are unlikely to flip on a 4-case-smaller
training set. The two Milestone 9 segmentation runs in progress at the
time of this entry (`segmentation_deep_backbone`,
`segmentation_deep_backbone_copy_paste`) were left running on `official`
rather than restarted, for the same reason -- both are far along and
restarting loses real GPU time for a number that will get the same
disclosure either way.

## 2026-09-01 -- Milestone 9 follow-up: Mask R-CNN, and a second, worse leakage bug caught before it produced a real number

Both finished segmentation runs (`segmentation_deep_backbone`,
`segmentation_deep_backbone_copy_paste`) landed at mIoU 0.637/0.654 --
nowhere near the 3-point-of-benchmark bar the user actually needs
(AP50_segm vs. TAPIS's mAP@0.5IoU_segm 89.85, our number 0.380 -- a
52-point gap). Tried attaching Milestone 8's `IOUTracker` to the
centroid/offset segmenter's decoded instances first (windowed raw-frame
tracking, same mechanism as `evaluate_tracking.py`, new
`run_window_and_track_segm` in `inference/pipeline.py` +
`scripts/evaluate_tracking_segm.py`): heavy-occlusion recall improved
0.185 -> 0.199 (~+7.6% relative) at the cost of AP50_segm dropping
0.380 -> 0.351 (coasted/stale masks becoming false positives) -- same
net-negative-on-the-headline-metric pattern as Milestone 8's tracking
result. Confirmed but not pursued further: tracking only ever changes
which instances get reported, it cannot touch the semantic head's
per-pixel mIoU, so it was never going to move the number that matters
here regardless of tuning.

**Real fix attempted**: the centroid/offset architecture is structurally
weak at AP50_segm specifically (coarse 96x96 instance separation from
heatmap peaks, not proposal-based) -- every literature entry with a high
AP50_segm/mAP@IoU_segm number (TAPIS included) uses proposal-based
instance segmentation (Mask R-CNN family). Milestone 8's box detector is
already close on the comparable metric (AP50_box 0.831, 5-9pts off
literature), so the plan: build `maskrcnn_mobilenet_v3`
(`models/detectors/mask_rcnn.py`), warm-start its backbone/RPN/box-head
from the trained `fasterrcnn_mobilenet_v3` checkpoint
(`detection_weighted_loss`, strict=False; verified directly --
only the 12 new mask-head tensors come back missing, zero unexpected/
mismatched keys), and train only the new mask head on the instance masks
already available. `GraspDetectionDataset` gained an `include_masks` flag
(native-resolution masks, `tv_tensors.Mask`-wrapped for transform
compatibility) rather than a parallel dataset class. `fit_detection`/
`train_one_epoch_detection` needed zero changes -- Mask R-CNN's train-mode
loss dict just adds `loss_mask` to the same `sum(loss_dict.values())`
already used there.

**First launch, on `fold1`, produced an impossible number**: val_mAP50 =
0.9946 after epoch 1 (vs. the 0.831 warm-start baseline -- fine-tuning
should perturb that number, not send it to near-ceiling in one epoch).
Root cause, caught before treating it as a real result: `fold1`'s val
split is 4 of the *same 8 official-train cases* the warm-start source
checkpoint's backbone/RPN/box-head were fit to via actual gradient
descent (that checkpoint was trained under `split: official`, i.e. on
fold1+fold2 combined). Evaluating on fold1 after warm-starting from it
measures memorization of frames the box components have literally
already seen weight updates from -- a strictly worse leakage than the
test-as-validation issue above (that one was *selection* bias from
early-stopping/checkpoint-choice; this one is direct training-set
reuse). Not something `data.split: fold1`'s existing definition was ever
meant to protect against -- it assumes whatever it's evaluating hasn't
already been fit to fold1 by some *other*, earlier training run, an
assumption a warm-start breaks.

**Fix**: retrained the box detector from scratch scoped to the same
fold boundary the Mask R-CNN will use (`configs/
detection_weighted_loss_fold1.yaml`, `split: fold1` -- i.e. trained on
fold2 only, same recipe as `detection_weighted_loss.yaml` otherwise,
one variable changed). Warm-starting the Mask R-CNN from *that*
checkpoint instead makes fold1 genuinely held out for every component,
box and mask alike. Killed the contaminated run immediately (single
epoch, ~2.5 min GPU time lost) rather than let it finish and report a
number that would need this same caveat attached anyway. In progress as
of this entry -- final AP50_segm/occlusion-recall numbers to follow once
both the fold-scoped detector and the re-warm-started Mask R-CNN finish.

**Expectation set going in, not walked back after the fact**: even a
successful Mask R-CNN result is very unlikely to land within 3 points of
TAPIS's 89.85 -- that figure comes from the dataset authors' own heavy,
purpose-built transformer architecture. Mask R-CNN is the right
*architecture family* for this metric (matches how the literature
actually gets these numbers), and warm-starting is the right way to
spend the GPU budget on it, but it is not a guarantee of closing a
49-52 point gap.
