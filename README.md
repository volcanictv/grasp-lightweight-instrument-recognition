# Surgical instrument recognition on GraSP

Research codebase evaluating whether a lightweight recognition model can
reach competitive instrument recognition on the GraSP dataset while cutting
parameters, latency, memory, and model size versus heavier architectures
(the paper's own TAPIS model). See `CLAUDE.md` and `PROJECT_SPEC.md` for
full project rules and reasoning.

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # dev laptop, has a local RTX 4060
```

Both machines now train. The Titan Xp workstation runs `torch==2.8.0+cu126`
in a Miniconda env (`~/miniconda3/envs/surgical`), verified with a real
tensor op on both GPUs; the dev laptop runs `torch==2.13.0+cu126` with its
own RTX 4060, also verified with a real tensor op (matmul + conv2d, not just
`is_available()`). See `docs/environment.md` for both machines' exact
versions and why the workstation's CUDA build is pinned more carefully
(Pascal sm_61 support) than the laptop's (Ada Lovelace, no such
constraint). Dataset path is read from `GRASP_DATA_ROOT` (defaults to
`./GraSp` if unset); it's never hardcoded in scripts.

## Status

- [x] Milestone -1 — Titan Xp workstation CUDA/PyTorch fixed and verified
      (`docs/environment.md`)
- [x] Milestone 0 — dataset inspection (`docs/dataset_report.md`)
- [x] Milestone 1 — loader throughput benchmark (below, both dev-laptop and
      workstation numbers; workstation is loader-bound, cache not yet built)
- [x] Milestone 2 — PyTorch Dataset/DataLoader against official splits
- [x] Milestone 3 — MobileNetV3-Small frozen-backbone baseline + runtime
      benchmark (`experiments/baseline_frozen_20260831-160550/`, workstation)
- [x] Milestone 4 — same model, fine-tuned, compared against Milestone 3
      (`experiments/baseline_finetuned_20260831-171454/`, workstation)
- [x] Milestone 5 — imbalance ablations: weighted loss, weighted sampler,
      augmentation, combined (below, workstation)
- [x] Milestone 6 — backbone sweep: MobileNetV3-Large, EfficientNet-B0,
      ResNet-18, ResNet-50 heavy baseline (below, workstation)
- [x] Milestone 7 — region classification (Task B) baseline (below,
      workstation)

## Dataloader throughput

Measured with `python scripts/benchmark.py loader`, dev laptop (Windows,
16 CPU cores, no GPU) — **not** the target Titan Xp + i5 workstation.
These numbers say the pipeline is CPU-bound on decode+augment, not disk
I/O; the actual img/sec on the workstation will differ and is what decides
whether the resized-frame cache from PROJECT_SPEC.md §5 is worth building.

Full augmentation pipeline (RandomResizedCrop, flip, color jitter, blur,
normalize, noise), train split, batch size 32, image size 224:

| num_workers | images/sec |
|---|---|
| 0 | 58.2 |
| 2 | 99.7 |
| 4 | 165.6 |
| 8 | 388.8 |

Single-threaded breakdown, isolating where the time goes:

| stage | images/sec |
|---|---|
| raw decode only (open + convert RGB) | 159.3 |
| decode + resize to 224 + to-tensor | 80.8 |
| full augmentation pipeline | 58.2 |

Decode itself isn't the dominant cost — resize and augmentation roughly
double it. That points at a resized on-disk cache of the ~3449 annotated
frames (a few thousand images, a few GB, well within the storage budget in
PROJECT_SPEC.md §5) as the likely fix if the workstation turns out to be
loader-bound, since a pre-resized JPEG skips the expensive native-resolution
(1280x800) decode+resize step every epoch.

**Workstation numbers** (`python scripts/benchmark.py loader --data-root ./GraSP`,
Titan Xp box: i5-7500, 4 threads, NVMe), same augmentation pipeline, train
split (2324 samples), batch size 32, image size 224:

| num_workers | images/sec |
|---|---|
| 0 | 94.6 |
| 1 | 92.8 |
| 2 | 174.4 |
| 3 | 228.7 |
| 4 | 317.3 |

Tops out at 317 img/s with all 4 threads saturated — higher than the
dev-laptop's 165.6 img/s at 4 workers (workstation has NVMe storage and no
other load), but the ceiling is real: there's no thread 5 to add. Model-side
benchmarking (`experiments/baseline_frozen_20260831-160550/benchmark.json`)
showed the frozen-backbone model can push 5258 img/s forward-pass throughput
on the GPU, so at 317 img/s the dataloader is the binding constraint by more
than 10x, not the GPU. Per CLAUDE.md's stated criterion this justified
building the resized on-disk cache of the ~3449 annotated frames.

**Cache built** (`scripts/build_frame_cache.py ./GraSP ./GraSP_cache`,
short side resized to 256px, JPEG quality 90): confirmed exactly 3449
annotated frames across train/test/fold1/fold2 (matches CLAUDE.md), written
in 34.2s, 93.3 MB total. The cache mirrors the source data root's layout
(`frames-001/frames/<file>`, `annotations/` symlinked back to the source) so
it's a drop-in `--data-root ./GraSP_cache` for `train.py` and
`benchmark.py` — no data-pipeline code changes needed.

Before/after loader throughput, same benchmark, same train split:

| num_workers | uncached img/sec | cached img/sec |
|---|---|---|
| 0 | 94.6 | 264.3 |
| 1 | 92.8 | 231.9 |
| 2 | 174.4 | 413.3 |
| 3 | 228.7 | 597.7 |
| 4 | 317.3 | 822.3 |

2.6x at the 4-worker ceiling. Still short of the model's 5258 img/s GPU
capacity, but the CPU is now the only remaining lever (already maxed at 4
threads) — this is close to the ceiling this hardware can give without a
CPU upgrade. Training configs should point `--data-root` at `./GraSP_cache`
going forward; `./GraSP` (uncached) remains the source of truth and is
untouched by this script.

## Milestone 4 — frozen vs. fine-tuned (PROJECT_SPEC.md Sec.6)

Same MobileNetV3-Small, same official split, same 20 epochs, run against
`./GraSP_cache`. Only `model.freeze_backbone` and the optimizer LR setup
differ: frozen trains only the head at lr=1e-3; fine-tuned unfreezes
everything and uses discriminative LR (`training.backbone_lr: 0.0001` on
backbone params, `training.lr: 0.001` on the head, per Sec.6's
recommendation — see `build_optimizer` in `scripts/train.py`).

| | frozen (`configs/baseline_frozen.yaml`) | fine-tuned (`configs/baseline_finetuned.yaml`) |
|---|---|---|
| trainable / total params | 598,023 / 1,525,031 (39.2%) | 1,525,031 / 1,525,031 (100%) |
| wall clock (20 epochs) | 399.6s | 169.4s |
| mean AP | 0.620 | 0.711 |
| macro F1 | 0.506 | 0.645 |

(Fine-tuned trained faster despite more trainable params — training here ran
after the frame cache was built, so it isn't dataloader-bound the way the
frozen run was; the params/MACs/latency numbers from Milestone 3's benchmark
still apply unchanged, since unfreezing doesn't alter the architecture.)

Per-class F1, frozen -> fine-tuned:

| class | frozen F1 | fine-tuned F1 |
|---|---|---|
| Bipolar Forceps | 0.861 | 0.885 |
| Prograsp Forceps | 0.525 | 0.580 |
| Large Needle Driver | 0.575 | 0.814 |
| Monopolar Curved Scissors | 0.895 | 0.934 |
| Suction Instrument | 0.437 | 0.429 |
| Clip Applier | 0.222 | 0.471 |
| Laparoscopic Grasper | 0.024 | 0.403 |

Fine-tuning clearly wins, most dramatically on the classes the frozen
backbone couldn't learn at all — Laparoscopic Grasper's recall was 0.012
frozen (effectively never predicted) and comes up to a usable, if still
weak, 0.403 F1 fine-tuned. ImageNet features alone don't carry enough
surgical-instrument-specific structure, matching Sec.6's expectation that
the frozen result would be weak.

The fine-tuned run also overfits hard by the numbers: train macro-F1 reaches
0.995 by epoch 20 while val macro-F1 plateaus around 0.60-0.65 from epoch
~8 onward and val loss rises after epoch ~3 (`docs/train_finetuned.log`).
Checkpoint selection already takes the best val-F1 epoch (19, not 20), so
this doesn't corrupt the reported numbers, but it's a real signal for
Milestone 5: this dataset is small enough (2324 train frames) that
unconstrained full fine-tuning memorizes it quickly, and imbalance/
regularization interventions should expect to be fighting overfitting, not
just class skew.

## Milestone 5 — imbalance ablations (PROJECT_SPEC.md Sec.7)

One variable at a time against the Milestone 4 fine-tuned baseline. Same
model, same 20 epochs, same `./GraSP_cache` data root. See
`docs/imbalance_notes.md` for the literature behind each choice.

- **weighted loss** (`configs/imbalance_weighted_loss.yaml`): inverse-frequency
  `pos_weight` per class in `BCEWithLogitsLoss` (already implemented,
  `training/losses.py`).
- **weighted sampler** (`configs/imbalance_weighted_sampler.yaml`):
  `WeightedRandomSampler` where each sample's weight is the *max*
  inverse-class-frequency among its active classes, not an average — so a
  frame with both a common and a rare instrument still gets the rare-class
  boost instead of it being diluted (`training/samplers.py`).
- **augmentation** (`configs/imbalance_augmentation.yaml`): a new
  `data.augmentation: strong` variant (`data/transforms.py`) — wider
  `RandomResizedCrop` range (0.6-1.0 vs 0.8-1.0), stronger color jitter
  (0.4 vs 0.2), and blur/noise applied more often (p=0.5 vs p=0.2), still
  within CLAUDE.md's plausibility constraints (flip/jitter/blur/noise/crop
  only).
- **combined** (`configs/imbalance_weighted_loss_augmentation.yaml`):
  weighted loss + strong augmentation together.

| run | mean AP | macro F1 | Clip Applier F1 | Laparoscopic Grasper F1 |
|---|---|---|---|---|
| baseline (Milestone 4, fine-tuned) | 0.711 | 0.645 | 0.222 | 0.403 |
| + weighted loss | 0.727 | 0.673 | 0.545 | 0.429 |
| + weighted sampler | 0.714 | 0.667 | 0.545 | 0.407 |
| + augmentation (strong) | **0.756** | 0.672 | 0.500 | 0.437 |
| + weighted loss + augmentation | 0.745 | **0.708** | **0.735** | 0.425 |

All four interventions beat the baseline on both mean AP and macro-F1 —
consistent with `docs/imbalance_notes.md`'s prediction that the cheap
interventions (weighted loss, weighted sampler) would help before reaching
for anything exotic. Two findings worth flagging:

- **Augmentation alone got the best mean AP but not the best macro-F1.**
  Its val loss stayed flat around 0.29-0.37 across all 20 epochs (versus the
  baseline's climb to 0.58), confirming it's acting as a regularizer against
  the Milestone 4 overfitting, not just an imbalance fix — but it doesn't
  target the rare classes directly the way weighted loss does.
- **Weighted loss + augmentation together is the clear macro-F1 winner**,
  driven almost entirely by Clip Applier: F1 0.735, better than either
  intervention alone (0.545/0.500) or the 0.222 baseline. Weighted loss
  pushes the model to attend to rare classes; augmentation stops it from
  overfitting to the ~64 Clip Applier training instances while doing so. This
  matches `PROJECT_SPEC.md`'s framing that combining interventions can help,
  even though each row above isolates one variable to know why.
- Laparoscopic Grasper stayed weak (F1 0.40-0.44) across every run,
  including the combined one — this looks less like an imbalance problem
  the current techniques can fix and more like the visual-similarity /
  test-set-concentration issue already flagged in
  `docs/dataset_report.md` (56% of its test instances come from one case,
  CASE050) and `docs/imbalance_notes.md` Problem 3. Not something a loss or
  sampler change addresses — worth a per-case breakdown before concluding
  it's failing, per that doc's stated reporting obligation, rather than
  reaching for a fifth ablation row.

## Milestone 6 — backbone sweep (PROJECT_SPEC.md Sec.10)

Same recipe as Milestone 5's winning combo (fine-tuned, discriminative LR,
weighted loss, strong augmentation) — architecture is now the only
variable. New backbones registered in `models/classifiers/`:
`efficientnet_b0`, `resnet18`, and `resnet50` (see `docs/DECISIONS.md` for
why ResNet-50 was picked as the "deliberately heavy baseline"
PROJECT_SPEC.md requires). Adding ResNet also exposed a real bug in
`build_optimizer`'s discriminative-LR split — it matched `"classifier"` in
param names to find the head, which silently misclassifies every ResNet
param as backbone (`.fc`, not `.classifier`). Fixed with a registry-wide
convention: every builder now sets `model.head` to its actual head
submodule, and training code reads that instead of guessing from a name.
MobileNetV3-Small's row below reuses Milestone 5's `imbalance_weighted_loss_augmentation`
run and Milestone 3's benchmark (architecture/runtime numbers don't change
with different trained weights).

| Model | Total params | Size | GPU latency (median) | Peak VRAM | GPU batch throughput | ONNX CPU latency (median) | Mean AP | Macro F1 |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small (proposed) | 1.53M | 5.93 MB | 4.14 ms | 16.6 MB | 5258 img/s | 1.41 ms | 0.745 | 0.708 |
| MobileNetV3-Large | 4.21M | 16.25 MB | 4.85 ms | 31.9 MB | 2099 img/s | 3.24 ms | 0.809 | 0.754 |
| EfficientNet-B0 | 4.02M | 15.59 MB | 6.31 ms | 34.4 MB | 1252 img/s | 5.95 ms | 0.808 | 0.749 |
| ResNet-18 | 11.18M | 42.72 MB | 1.95 ms | 77.9 MB | 2374 img/s | 9.49 ms | 0.798 | 0.740 |
| ResNet-50 (heavy baseline) | 23.52M | 90.02 MB | 4.98 ms | 126.0 MB | 708 img/s | 21.84 ms | **0.862** | **0.799** |

Runs: `experiments/sweep_mobilenet_v3_large_20260831-174946/`,
`experiments/sweep_efficientnet_b0_20260831-175341/`,
`experiments/sweep_resnet18_20260831-175902/`,
`experiments/sweep_resnet50_heavy_20260831-180241/`.

**Accuracy ranks roughly by params**, as expected — ResNet-50 wins
everything, including finally getting real traction on Laparoscopic
Grasper (F1 0.622, versus 0.40-0.52 for every other model in this project so
far). It's the top of the tradeoff curve PROJECT_SPEC.md Sec.10 asks for.

**GPU latency does not rank by params, and this is the finding that
justifies the whole ONNX CPU column.** ResNet-18 has ~4-8x the MACs of the
MobileNet/EfficientNet models but the *lowest* Titan Xp latency (1.95ms) of
the entire sweep — faster than MobileNetV3-Small itself. This is Pascal's
lack of tensor cores plus depthwise-separable convolutions (used by
MobileNetV3 and EfficientNet, not ResNet) being FLOP-efficient but not
GPU-kernel-efficient: low arithmetic intensity per launch, poor memory
bandwidth utilization on this hardware. CLAUDE.md's standing warning not to
treat Titan Xp numbers as a deployment claim is proven correct by this
result, not just a defensive caveat. On ONNX CPU — the portable number —
the ranking flips back to matching params/MACs exactly, with MobileNetV3-Small
over 15x faster than ResNet-50 (1.41ms vs 21.84ms). That gap, not the GPU
column, is the actual case for a lightweight model on hardware this project
doesn't own.

**Where this leaves the tradeoff curve**: ResNet-50 buys +0.091 macro-F1 over
MobileNetV3-Small (0.799 vs 0.708) at 15x the ONNX CPU latency and 15x the
model size. MobileNetV3-Large and EfficientNet-B0 sit at a similar
accuracy tier to each other (macro-F1 0.754/0.749, both well above
MobileNetV3-Small's 0.708) for 2-4x MobileNetV3-Small's ONNX latency and
model size — the actual middle of the curve. Milestone 10's Pareto analysis
is where this gets formalized; for now the honest summary is that
MobileNetV3-Small remains the right choice only if the deployment target is
CPU-latency-constrained, not GPU-latency-constrained, which is exactly the
argument CLAUDE.md's benchmarking section is structured to make.

## Milestone 7 — region classification (Task B)

`configs/region_baseline.yaml`: MobileNetV3-Small, fine-tuned, discriminative
LR (same as Task A), `CrossEntropyLoss` with balanced class weights (Task
B's per-instance class counts are imbalanced too — Clip Applier is 64/6170
train instances, a 25.6x gap vs. the most common class, same ratio
`docs/imbalance_notes.md` flagged for Task A). New infrastructure this
needed: `GraspRegionDataset` (one sample per annotated instance, not per
frame), which crops each instance to its bbox *and* multiplies by its own
per-instance segmentation mask before classifying — zeroing out background
and any overlapping second instrument rather than a raw rectangular crop,
per `docs/imbalance_notes.md` Problem 4. Verified visually on a Clip Applier
instance before training (clean instrument silhouette, black elsewhere, no
bleed from an adjacent instrument). Task B must use the uncached `GraSP`
data root, not `GraSP_cache` — see `docs/DECISIONS.md` for why the resized
cache silently breaks bbox-coordinate cropping.

Trained on 6170 instances (train) / evaluated on 2861 (test), 20 epochs,
1168.6s wall clock (uncached, no loader speedup available here):

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

Run: `experiments/region_baseline_20260831-182451/`.

**This is a materially different result than every Task A run.** Clip
Applier — the class that was barely usable across the entire Milestone 5/6
Task A sweep (F1 0.222-0.735, best case) — is F1 0.861 here, among the
*best*-performing classes, not the worst. Laparoscopic Grasper goes from
stuck at 0.40-0.62 across every Task A intervention (including all four
backbones) to 0.713. The confusion matrix (`figures/confusion_matrix.png`)
backs this up directly: 34/38 Clip Applier test instances land on the
diagonal.

The likely explanation isn't that Task B "solved" the imbalance — it's that
region classification removes two problems Task A never could: once the
model is handed an isolated, masked crop of one instrument, there's no
whole-frame background/tissue context to get lost in, no ambiguity about
*which* of 2-3 co-occurring instruments a positive label refers to, and no
localization the model has to implicitly learn. That's consistent with
`imbalance_notes.md` Problem 3's read on Laparoscopic Grasper specifically
(56% test-case concentration) — a chunk of what looked like an unfixable
class-imbalance problem in Task A may really have been a framing problem
that region-level crops sidestep entirely.

**Caveat that matters for interpreting this against the paper**: this
number uses *ground-truth* instance masks to build the crops — it measures
"can the model name an instrument once perfectly told where it is," not an
end-to-end detect-then-classify pipeline. TAPIS's own published number
(mAP@0.5IoU_segm, 89.85%, aggregate, no per-class breakdown) is an
end-to-end segmentation metric that folds in their own localization error,
so it isn't a fair comparison point yet — that only becomes possible once
Milestones 8-9 (detection, segmentation) exist and this project has its own
end-to-end number in the same protocol.
