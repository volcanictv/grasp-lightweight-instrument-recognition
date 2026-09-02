# Project Spec

Full context for the GraSP lightweight instrument recognition project. CLAUDE.md
holds the rules. This file holds the reasoning. Read it when you need to know
*why* a rule exists, when scoping a new milestone, or when writing docs.

> **Superseded 2026-09-01**: the lightweight-first framing throughout this
> document (efficiency as the primary axis, "secondary: an efficient
> component") has been reordered per direct guidance from the PhD students
> running the upstream research. Current priority: accuracy first (ideally
> exceeding TAPIS's published benchmark), then generalizability on a second
> dataset, then efficiency last. CLAUDE.md's "What this project is" section
> holds the current rule; this file's reasoning below is kept as the
> historical record of why lightweight-first was the original choice, not as
> a currently-governing goal. See `docs/DECISIONS.md`, 2026-09-01.

---

## 1. Context

This is a biomedical engineering / medical imaging lab project, and the first
formal research project for the person running it. The lab has had PhD students
work on adjacent parts of this problem before. Their recurring failures:

1. Severe class imbalance, with rare instruments effectively ignored.
2. Pipelines growing too large and computationally expensive.
3. Slow inference.
4. Codebases too difficult to maintain against real surgical imagery.

Those four failures are the reason for every constraint in CLAUDE.md. The
efficiency focus is not an aesthetic preference, it is a direct response to what
went wrong last time. When a design choice trades maintainability or inference
cost for a small accuracy gain, the default answer is no.

## 2. Research question

> Can a lightweight surgical instrument recognition model reach competitive
> performance on GraSP while substantially reducing parameter count, inference
> latency, memory consumption, and model size compared with heavier
> architectures?

Secondary: can that lightweight recognizer serve as an efficient component of a
broader instance-level segmentation pipeline?

The end goal is a pipeline that produces a near-immediate segmentation map for
instruments in a frame. That deployment target is what makes latency a
first-class metric rather than a footnote.

The final claim must come from the experiments. Do not assert a lightweight model
wins before the numbers exist, and do not write novelty language into the README
or docstrings.

## 3. Target pipeline

```
frame -> preprocessing -> lightweight backbone -> classification
                                               -> detection
                                               -> segmentation -> mask
```

Architecture is not fixed. Backbones, heads, losses, samplers, and segmentation
components must be swappable via config without rewriting the framework. This
modularity is a hard requirement, not a nice-to-have.

## 4. Environment blocker

The workstation runs an inherited Ubuntu + NVIDIA/CUDA install that is hard to
maintain, and **GPU detection is currently broken**. Nothing trains until this is
fixed. Titan Xp is Pascal, compute capability 6.1, so the PyTorch/CUDA pair must
still ship sm_61 kernels. Verify with a real tensor op on `cuda:0`, not just
`torch.cuda.is_available()`.

Record the working combination in `docs/environment.md` with exact versions,
driver version, and the command that verified it. This will break again.

## 5. Storage

~256 GB total, already nearly full from prior projects. Competing for it:
dataset, annotations, checkpoints, logs, TensorBoard runs, cached pretrained
weights, experiment outputs.

- Lazy/indexed access to frames. No duplicated image trees.
- The one justified exception: a resized cache of the annotated frames only.
  That set is a few thousand images, a few GB, and it fixes the dataloader
  bottleneck. Document the measured before/after throughput when adding it.
- A 1 TB SSD upgrade is expected. Do not architect around the current limit, but
  do not ignore it either.

## 6. Frozen vs fine-tuned

This distinction should be explicit in the code and the results, because it is
the first real experiment and the person running it is new to practical
fine-tuning.

**Experiment A, frozen.** ImageNet-pretrained backbone with `requires_grad=False`
on all backbone parameters. Only the new head trains. This measures how much
useful visual structure ImageNet features already carry for surgical imagery.
Expect it to be weak. Endoscopic frames are far from ImageNet's distribution.
Weak is a valid, informative result, not a failure.

**Experiment B, fine-tuned.** Same backbone, some or all layers trainable,
typically a lower learning rate on backbone params than on the head. This is the
real baseline.

Config controls it through `model.freeze_backbone`. The trainer should log how
many parameters are trainable versus total at startup so the two runs are
distinguishable from the logs alone.

Report both. The gap between them is a result worth having.

## 7. Class imbalance

Rare instruments (Clip Applier especially) can be ignored by the model while
overall accuracy stays high. This is why accuracy is never the headline number.

Interventions to test, each as its own run:

- **A. Weighted loss.** Higher loss weight on underrepresented classes. Weighting
  strategy configurable (inverse frequency, effective number of samples, capped
  variants).
- **B. Weighted sampling.** A sampler preventing common classes from dominating
  batches.
- **C. Oversampling.** Increased exposure to rare-class frames.
- **D. Augmentation.** See CLAUDE.md for the surgical plausibility constraints.

Ablation grid:

```
baseline
baseline + weighted loss
baseline + balanced sampler
baseline + augmentation
baseline + weighted loss + augmentation
```

One variable per run. Stacking everything at once tells you nothing about which
intervention actually worked.

`docs/imbalance_notes.md` tracks how other groups handle long-tailed surgical
recognition, with citations and a note on whether each technique fits our latency
budget. Be honest in it. Most imbalance techniques are standard. Our contribution
is more likely the efficiency curve than a new sampling trick.

## 8. Detection

Answers "what instrument, and where." Output is box + class + confidence.

Boxes come free from the instance masks. No new annotation is needed. Derive
them in the dataset layer and cache the derivation.

Do not commit to a detector architecture early. Lightweight YOLO variants and
other efficient detectors are candidates, chosen after the classification
baselines establish what the backbone can do.

Metrics: mAP, mAP@50, mAP@50:95, precision, recall, per-class AP, plus the full
runtime set.

## 9. Segmentation

The eventual target is instance-level: a pixel mask per instrument instance, not
a semantic map. Architecture is open and should be chosen to integrate with the
recognition stage rather than bolted on.

Metrics: IoU, Dice, mIoU, per-class IoU, pixel accuracy where meaningful, plus
latency and FPS.

Keep the interfaces clean enough that a segmentation model drops into the
pipeline without touching data, training, or evaluation code.

## 10. Benchmarking philosophy

Benchmark before fine-tuning, then after. Do not build the final model and
benchmark retroactively.

Stages to compare:

1. Pretrained frozen
2. Fine-tuned
3. Fine-tuned + imbalance mitigation
4. Lightweight architecture variants
5. Optimized architecture

Every experiment records: config, model, split, seed, training params,
augmentations, loss, sampling strategy, metrics, runtime measurements,
checkpoint, results.

The target output is a table of this shape, built incrementally:

| Model | Params | Size | VRAM | Latency | FPS | Macro-F1 |
|---|---|---|---|---|---|---|
| Heavy baseline | | | | | | |
| Medium baseline | | | | | | |
| Lightweight baseline | | | | | | |
| Proposed | | | | | | |

Include a deliberately heavy baseline. Without a top end there is no tradeoff
curve, only a single point.

## 11. Visualization

Numerical metrics hide systematic failure modes, and this is medical imaging.
Every stage needs visual output saved to disk (headless, no display).

**Dataset:** class distribution, samples per case, sample frames, mask overlays.

**Classification:** confusion matrix, per-class precision/recall/F1 bars,
train/val loss curves, train/val metric curves.

**Detection:** box overlays on frames, confidence distributions, per-class AP.

**Segmentation:** ground truth mask, predicted mask, overlay, and an error
visualization showing false positive and false negative regions distinctly.

Also useful: a worst-N viewer that dumps the frames with the highest loss or
lowest per-instance confidence. That is usually where the systematic failure
lives.

## 12. Expected challenges

- **Class imbalance.** Rare instruments ignored.
- **Occlusion.** Instruments overlapping tissue, each other, and equipment.
- **Appearance variation.** Lighting, camera angle, tissue, blood, smoke, blur,
  specular reflections.
- **Small or partially visible instruments.** An instrument may occupy a small
  fraction of the frame, which interacts badly with aggressive downscaling.
  Check whether 224px input destroys rare-class recall before accepting it.
- **Multiple instruments per frame.** The reason the task is multi-label or
  region-based, never single-label whole-frame.
- **Temporal correlation.** Adjacent frames are near-duplicates.
- **Leakage.** Case-level splits only, from the official JSONs.
- **Compute limits.** Accurate but slow is a failed result here.
- **Legacy GPU.** Pascal, no tensor cores, fragile driver stack.

## 13. Development sequence

```
understand dataset -> verify annotations -> data loader -> baseline ->
benchmark frozen -> fine-tune -> address imbalance -> lightweight variants ->
detection -> segmentation -> optimize inference -> ablations -> final model
```

The first goal is a working baseline, not the research contribution. Do not
attempt a novel architecture early.

## 14. Initial build checklist

Milestones 0 through 4 in CLAUDE.md should collectively deliver:

- [ ] Dataset configuration
- [ ] Dataset inspection script
- [ ] Annotation parser
- [ ] Dataset statistics
- [ ] Visualization utilities
- [ ] PyTorch Dataset / DataLoader
- [ ] Split handling from official JSONs
- [ ] MobileNetV3 baseline classifier
- [ ] Frozen-backbone mode
- [ ] Fine-tuning mode
- [ ] Configurable class weighting
- [ ] Configurable balanced sampling
- [ ] Classification metrics
- [ ] Confusion matrix
- [ ] Checkpointing
- [ ] Experiment logging and manifests
- [ ] Inference script
- [ ] Runtime / FPS benchmark
- [ ] GPU memory benchmark
- [ ] Parameter / model-size benchmark
- [ ] README covering setup and how to reproduce each experiment

No complexity beyond this until the baseline runs end to end.
