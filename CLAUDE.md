# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

Research codebase for lightweight surgical instrument recognition on the **GraSP**
dataset. The question we are answering:

> Can a lightweight recognition model reach competitive instrument recognition on
> GraSP while substantially cutting parameters, latency, memory, and model size
> versus heavier architectures?

The deliverable is a defensible accuracy/efficiency tradeoff curve, not a new
architecture. Benchmarking rigor and reproducibility beat novelty here.

## Documents in this repo

This file is the rules. Read the others when the situation calls for it.

| File | Read it when |
|---|---|
| `docs/PROJECT_SPEC.md` | Scoping a milestone, or you need the reasoning behind a rule here. Contains lab context, imbalance strategy, visualization requirements, metric lists, and the build checklist. |
| `docs/dataset_report.md` | Before any modeling. Generated in Milestone 0. |
| `docs/environment.md` | GPU or CUDA trouble. Records the working version combination. |
| `docs/DECISIONS.md` | Before reversing or re-litigating a past choice. Append-dated, newest last. |
| `docs/imbalance_notes.md` | Working on class imbalance. Literature and fit-for-budget notes. |
| `docs/findings.md` | Writing up or citing results. Consolidated, report-ready results and analysis across all milestones, numbers sourced from `experiments/*/manifest.json`. |

If a rule here and the spec disagree, this file wins and the spec gets fixed.

## Dataset facts (verified, do not re-guess these)

GraSP = **Holistic and Multi-Granular Surgical Scene Understanding of
Prostatectomies**, Ayobi et al., Universidad de los Andes / CinfonIA.
Paper: arXiv 2401.11174. Repo: github.com/BCV-Uniandes/GraSP.

- Procedure is **robot-assisted radical prostatectomy** (da Vinci), not
  gastrointestinal and not conventional laparoscopy. Do not write "laparoscopic
  surgery" in docs or comments. It collides with the class name below.
- 13 annotated sequences. Official split: 8 train / 5 test. Case directories are
  named `CASE001`, `CASE053`, etc. Numbering is not contiguous.
- Native frame resolution 800x1280.
- 7 instrument classes: Bipolar Forceps, Prograsp Forceps, Large Needle Driver,
  Monopolar Curved Scissors, Suction Instrument, Clip Applier, Laparoscopic
  Grasper. Clip Applier is the rare class per the paper's own analysis.
- Instrument segmentation masks are **sparse**, roughly 3.5k annotated frames
  across the whole dataset, not one per sampled frame. Most of the 60 GB of
  frames carry no instrument annotation. Verify the exact count from the JSON in
  Milestone 0 and record it.

### Annotation files

```
annotations/
  segmentations/CASE0xx/000000068.png        instance masks, annotated frames only
  grasp_short-term_train.json                instruments + atomic actions
  grasp_short-term_test.json
  grasp_short-term_fold1.json
  grasp_short-term_fold2.json
  grasp_long-term_*.json                     phases and steps, not our task
```

Short-term files are ours. Long-term (phase/step) is out of scope.

## Splits: use the official ones

The leakage problem is already solved by the dataset authors. Splits are defined
at case level in the JSON files above, and `fold1`/`fold2` give case-level
cross-validation.

- Read splits from the official JSONs. Do not write a custom splitter.
- Never split at frame level. Frames within a case are near-duplicates.
- If a config requests a split not present in the JSONs, fail loudly rather than
  falling back to random.

Using official splits also makes our numbers directly comparable to the paper.

## Task framing

The spec's original "one instrument per frame" framing does not hold. GraSP
frames routinely contain 2 to 3 instruments simultaneously. Two valid tasks:

**Task A: multi-label frame classification.** 7 sigmoid outputs, BCE loss,
per-class AP and macro-F1. This is the honest version of "what instruments are
in this frame." Use this as the first baseline.

**Task B: region classification.** Crop each annotated instrument instance
(from its mask bounding box), classify the crop into 7 classes. Single-label,
clean, and it is the task TAPIS actually solves with a heavy transformer head.
A MobileNetV3 that matches it at a fraction of the cost is the strongest result
this project can produce.

Build A first because it needs no cropping infrastructure. Build B second
because it is where the research contribution lives.

Never implement single-label whole-frame classification. It is ill-defined on
this dataset and any number it produces is meaningless.

## Hardware reality

Workstation: 2x Titan Xp (12 GB each), 32 GB RAM, i5, ~256 GB storage, Ubuntu.

- Titan Xp is Pascal, compute capability **6.1**. No tensor cores. AMP saves
  memory but gives close to zero speedup. Do not report AMP as a throughput win.
- Pin the CUDA/PyTorch combo that supports sm_61 and record exact versions in
  every experiment manifest.
- With an i5 and 800x1280 JPEGs, MobileNetV3 training will be **dataloader
  bound**, not GPU bound. Before tuning anything, measure pure loader throughput
  (images/sec with the model removed) and put the number in the README. If the
  loader is the bottleneck, a one-time resized cache of the annotated frames only
  (a few thousand images, a few GB) is justified. Say so explicitly in the docs.
- Single GPU first (`cuda:0`). Once the pipeline is stable, prefer running two
  independent experiments on the two GPUs over DDP on one experiment.
- Development happens on a laptop over SSH. Never assume local GPU access. All
  scripts must run headless with no display, and all plots save to disk.
- Correction, 2026-08-31: the dev laptop itself has a working RTX 4060
  (8 GB, Ada Lovelace, tensor cores) — it can train too, verified the same
  way as the workstation (see `docs/environment.md`). "Never assume local
  GPU access" still applies to any *other* machine you haven't checked; it
  no longer means this specific laptop has none. Its CPU/disk numbers
  (loader throughput, etc.) still aren't the workstation's and must not be
  substituted for them.

## Latency benchmarking

Titan Xp latency is not a deployment claim. Every benchmark must report:

- Hardware-independent: parameter count, trainable parameters, FLOPs/MACs, model
  file size on disk.
- Titan Xp: single-image latency (median and p95 over >=200 warm runs), batch
  throughput, peak VRAM.
- ONNX Runtime CPU latency. This is what makes the "practical deployment"
  argument portable to hardware we do not own.

Always warm up before timing and always `torch.cuda.synchronize()` around GPU
timing. Report median, not mean.

## Repository layout

```
configs/            YAML experiment configs
src/surgical_ai/
  data/             dataset.py, transforms.py, splits.py, statistics.py
  models/           classifiers/, detectors/, segmenters/
  training/         trainer.py, losses.py, samplers.py, callbacks.py
  evaluation/       classification.py, detection.py, segmentation.py, benchmarking.py
  inference/        pipeline.py
  utils/            logging.py, seeds.py, visualization.py
scripts/            inspect_dataset.py, train.py, evaluate.py, benchmark.py, inference.py
experiments/        one directory per run, git-ignored except manifests
docs/               notes, including literature notes
tests/
```

Adding a new backbone must not require touching `data/`, `training/`, or
`evaluation/`. Use a registry pattern for models, losses, and samplers.

## Configuration

Every experiment is fully described by one YAML file. No hyperparameters in
Python. No CLI flags that change results (paths and device only).

```yaml
task: multilabel_frame        # or region_classification
model:
  name: mobilenet_v3_small
  pretrained: true
  freeze_backbone: true
data:
  split: official             # official | fold1 | fold2
  image_size: 224
  sampling: none              # none | weighted | oversample
loss:
  type: bce
  class_weights: false
training:
  batch_size: 32
  epochs: 20
  lr: 0.001
  seed: 42
```

Each run writes `experiments/<run_id>/manifest.json` containing: full resolved
config, git commit hash, git dirty flag, seed, split file paths and their
checksums, package versions, GPU name and driver, wall-clock duration, best
checkpoint path, and final metrics. The repo must be able to answer "exactly how
did we produce this number" without reconstruction.

## Metrics

Accuracy is never the headline number. Report macro-F1 and per-class F1 always.
Confusion matrix for every classification run.

- Multi-label: per-class AP, mAP, macro-F1, per-class precision/recall.
- Region classification: accuracy, macro precision/recall/F1, per-class F1,
  confusion matrix.
- Detection (later): mAP@50, mAP@50:95, per-class AP.
- Segmentation (later): IoU, Dice, mIoU, per-class IoU.

## Milestones

Work in this order. Do not skip ahead.

-1. **Environment.** The workstation's CUDA install is currently broken and GPU
   detection fails. Fix and pin it. Verify with a real tensor op on `cuda:0`, not
   just `torch.cuda.is_available()`. Record the working versions, driver, and the
   verifying command in `docs/environment.md`. Milestone 0 can run CPU-only, but
   nothing past 2 can.
0. **Dataset inspection.** Parse short-term JSONs, count cases/frames/annotated
   frames/instances, class histogram overall and per case, resolution
   distribution, co-occurrence matrix, missing or corrupt entries. Dump sample
   frames and mask overlays. Write findings to `docs/dataset_report.md`.
   Nothing else starts until this is done and read.
1. Loader throughput benchmark. Record images/sec. Decide on caching.
2. PyTorch Dataset + DataLoader against official splits. Lazy loading, no
   duplicated image directories.
3. MobileNetV3-Small, frozen backbone, multi-label head. Full metrics and
   runtime benchmark. This is the reference baseline.
4. Same model, fine-tuned. Compare against 3.
5. Imbalance interventions as separate ablation runs: weighted loss, weighted
   sampler, augmentation, and combinations. One variable per run.
6. Backbone sweep: MobileNetV3-Large, EfficientNet-B0, ResNet-18, plus a
   deliberately heavy baseline for the top of the tradeoff curve.
7. Region classification (Task B).
8. Detection. Boxes come free from the instance masks, no new annotation needed.
9. Segmentation integration.
10. Inference optimization and final Pareto analysis.

## Augmentation constraints

Surgical imagery has real constraints. Horizontal flip is acceptable. Brightness,
contrast, saturation, mild color jitter, blur, noise, mild crops and scales are
acceptable. Do not use vertical flips, large rotations, or hue shifts that turn
tissue non-physiological. Anything that makes a frame surgically implausible is
out.

## Literature notes

Maintain `docs/imbalance_notes.md` covering how other groups handle long-tailed
surgical instrument recognition, with citations and a short note on why each
technique would or would not fit our latency budget. Be honest in it: imbalance
handling is well-trodden. If a technique is standard, say so rather than dressing
it up as novel.

## Code style

Commits:
- Author is the repo's configured git user. Never set `--author`, never add
  `Co-Authored-By`, `Generated with`, or any AI attribution to commits or PRs.
- Lowercase, present tense, under ~60 chars. `fix mask path lookup for
  non-contiguous case ids`, not `fix(data): resolve mask path resolution issue`.
- No conventional-commit prefixes, no scopes, no task or phase numbers.
- Body only when the change needs justifying. A three-line diff gets a
  one-line message, not a bulleted summary.

Comments:
- Explain why, not what. If the line is obvious, no comment.
- No section banners (`# ---- Data Loading ----`), no `# Step 1:`, no
  narration (`# Now we normalize the tensor`).
- Docstrings only where behavior is non-obvious. Never restate the signature
  or list every arg on a two-line function.
- No emoji anywhere. No "comprehensive", "robust", "seamless", "production-ready".
- Don't wrap things in try/except unless a specific failure is expected and
  handled differently.

General:
- Type hints on public functions.
- Do not add abstraction layers before there are two concrete users of them.
- Seed everything (`random`, `numpy`, `torch`, cudnn deterministic where it does
  not destroy throughput) and log the seed.

## Things not to do

- Do not build the final model first and benchmark after.
- Do not add every imbalance technique at once. One variable per run.
- Do not preprocess the full 60 GB into duplicated directories.
- Do not claim novelty in docstrings or the README. Claims come from results.
- Do not hardcode absolute dataset paths. Read them from config or environment.
