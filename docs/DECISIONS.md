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
