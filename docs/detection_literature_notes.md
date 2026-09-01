# Detection / occlusion literature notes

Companion to `docs/imbalance_notes.md` (Task A/B imbalance) and
`docs/findings.md` (GraSP-specific benchmark literature). This doc covers
Milestone 8's own open problem specifically: occlusion between co-occurring
instruments measurably hurts detection recall (occlusion-stratified recall,
`evaluation/detection.py`, ~0.60-0.65 for heavily-occluded instances vs.
~0.88-0.91 otherwise, consistent across all three detector variants tried).
Two rounds of generic COCO-style augmentation (Milestone 8 follow-up) both
made the aggregate detector *worse*, not better -- this doc is the
literature check done in response to that, so the next attempt is informed
by what the field has already tried rather than more trial and error.
Compiled 2026-08-31 from a web search, not a systematic review -- same
caveat as `docs/findings.md`'s literature section.

## Occlusion is a standing, unsolved challenge, not a gap in our approach

Both the systematic review of 48 studies on DL for surgical instrument
recognition/segmentation (Deng et al., arXiv 2410.07269, *Artificial
Intelligence Review* 2024) and the broader survey of DL applied to surgical
data (arXiv 2209.01435) list occlusion -- by other instruments, by tissue,
by blood, by smoke -- as an open problem the field is still working on, not
something any published method claims to have fully solved. Worth stating
plainly for the report: our finding is consistent with the field's own
standing difficulty, not evidence of a mistake in this project's approach.

## Technique 1: anchor-free, centroid-based instance segmentation (most relevant)

Kurmann et al., "Mask then classify: multi-instance segmentation for
surgical instruments," *IJCARS* 2021 (also PMC8260538). Instead of anchor
boxes + NMS -- the exact mechanism that incorrectly suppresses one of two
real, overlapping objects, which is our diagnosed failure mode -- the model
predicts three pixel-wise outputs: an instrument segmentation mask, an
offset regression (pixel -> instance centroid), and a centroid heatmap
(unnormalized Gaussians centered on each instance's mask median; where two
Gaussians overlap, the max value is kept). Instances are then formed by
clustering pixels around detected centroids using the offsets. Two
overlapping instruments still have spatially distinct centroids even when
their masks/boxes overlap heavily by area, so overlap stops being something
NMS has to adjudicate at all -- it's a structural fix, not a training-data
or augmentation fix.

**Why this matters for this project specifically**: it's a segmentation
architecture, not a box detector, so it maps directly onto Milestone 9
(segmentation integration), which hasn't started yet. This is a real
candidate for how to build that milestone, rather than defaulting to a
generic Mask R-CNN and re-discovering the same NMS-suppression problem at
the mask level.

**Fit for this project's constraints**: no heavier than what we already
run (still a CNN with a few output heads, not a transformer); doesn't add
architecture-family novelty beyond what a segmentation milestone needs
anyway; genuinely targets the diagnosed problem rather than being a generic
best-practice applied hopefully.

## Technique 2: temporal / video context

González et al. 2020 (cited in arXiv 2209.01435) add a temporal
information module to Mask R-CNN specifically to preserve an instrument's
identity across frames, so a frame where it's briefly occluded can still be
resolved using neighboring frames. This is architecturally what TAPIS
itself does -- its own description is "a global video feature extractor"
combined with region proposals, not a per-frame detector.

**Why this matters for interpreting our results honestly**: TAPIS's
advantage over our Faster R-CNN may not be reducible to parameter count.
Our detector looks at one static frame; TAPIS explicitly doesn't. Some of
the mAP gap documented in `docs/findings.md` and the Milestone 8 report
section may not be closeable by a bigger single-frame model at all --
worth stating in the report as a limitation of the detection baseline's
architecture, not just its size, so the "heavier model closes the gap"
narrative (true for Task A's backbone sweep) isn't assumed to transfer
here without qualification.

**Fit for this project's constraints**: a full video-transformer-style
version (matching TAPIS's own architecture) is a real scope increase --
changes the dataset/dataloader from frame-level to clip-level, and is a
bigger lift than Technique 1. Not recommended as a first attempt.

**Cheaper version worth building into Milestone 9 directly**: González et
al.'s actual goal -- preserving an instrument's identity across frames so a
momentary occlusion can be resolved from neighboring frames -- doesn't
require a video transformer to get most of the benefit. Classical
tracking-by-detection (IoU-based frame-to-frame association, Kalman-filter-
style track continuity -- SORT/DeepSORT-class techniques, decades-old and
essentially free computationally) layered on top of the existing per-frame
detector/segmenter could carry a track through an occluded frame using the
track's own recent history, without changing the detector's architecture or
its per-frame latency at all. This fits the project's efficiency thesis
better than a video transformer would: it's a post-processing/tracking
layer, not a heavier model, so it doesn't trade away the latency argument
that's the whole point of staying lightweight. Concretely for Milestone 9:
run the (still per-frame) segmenter as-is, then add a lightweight tracker
across consecutive frames of the same case, and check whether occlusion-
stratified recall (the exact metric already built for Milestone 8) improves
for tracked instances vs. the untracked per-frame baseline. If that closes
a meaningful part of the occlusion gap, the full video-transformer version
(Milestone 10+) may not be needed at all.

## Why generic augmentation likely failed (explains, doesn't excuse, the Milestone 8 follow-up result)

The survey (arXiv 2209.01435) describes the augmentation that's actually
been validated in this domain as skewing toward domain-specific synthesis:
GAN-based image-to-image translation (CycleGAN-style domain adaptation,
Pfeiffer et al. 2019), surgical-simulator-rendered synthetic data (3D
Slicer, dV-trainer, AMBF -- though the survey notes a realism gap), and
real-tissue-background-plus-synthetic-tool compositing (Su et al. 2021) --
not generic geometric/photometric jitter borrowed from natural-image
detection benchmarks like COCO. This is a plausible, literature-backed
explanation for why `RandomZoomOut`/`RandomIoUCrop` (docs/DECISIONS.md,
2026-08-31 entry) regressed both mAP@50 and mAP@50:95 here: those
transforms were validated on large, visually diverse datasets, and GraSP is
small (2324 train frames) and visually homogeneous (one surgical setup,
consistent framing) -- exactly the profile where COCO-style scale/crop
jitter is more likely to add distribution mismatch than useful diversity.
Not pursuing GAN-based synthesis in this project -- real engineering cost,
out of scope for a benchmarking project per `CLAUDE.md`'s "no novelty"
framing -- but this closes the open question of *why* augmentation
regressed rather than leaving it unexplained.

## Backbone / lightweight-vs-heavy data points from the literature (context for the tradeoff curve)

The survey also documents lightweight surgical-instrument segmentation
work directly comparable in spirit to this project's own backbone sweep
(`docs/findings.md` Sec.4.3):
- MobileNetV3 + ghost modules, real-time instrument segmentation at 37 FPS
  (Sun et al. 2021).
- ToolNet: 15x fewer parameters than FCN-8s at 29 FPS, but "limited feature
  receptive fields" -- an explicit accuracy/efficiency tradeoff acknowledged
  by its own authors (Islam et al. 2019), the same shape of tradeoff this
  project's Task A sweep already demonstrates.
- ShuffleNet, GhostNet, LinkNet/ICNet/PSPNet listed as other lightweight
  options for real-time surgical segmentation.

This corroborates the project's core framing (lightweight models trade a
bounded amount of accuracy for real efficiency gains, per Task A's ResNet-50
comparison) as a live, validated pattern elsewhere in this exact
application area, not something specific to GraSP or to this project's
own choices.

## NMS variants: essentially undocumented in this literature -- a real gap, not a validated fix

The survey provides "minimal coverage" of NMS variants specifically; no
soft-NMS, weighted-NMS, or similar was found described for surgical
instrument detection. What multi-instance handling *is* documented (Kong et
al. 2021's IoU-threshold rules for Mask R-CNN region proposals; Kurmann et
al.'s centroid approach above) works at a different level than swapping the
NMS algorithm. **Soft-NMS (Bodla et al. 2017, general computer vision, not
surgical-specific) is being tried anyway** (see `docs/DECISIONS.md`) because
it directly targets the mechanism of our diagnosed failure -- hard NMS
suppressing one of two real overlapping objects as a "duplicate" -- and
costs nothing to test (pure inference-time post-processing on an already-
trained checkpoint, no retraining). But it should be reported as an
untested-in-this-domain general-CV technique, not a literature-validated
fix for surgical instrument occlusion specifically -- that honesty matters
if this ends up in the report.

## One dataset caveat worth remembering

A real-time DETR-based surgical tool detector (Loza et al. 2024, *Healthcare
Technology Letters*) reports mAP@50 = 0.945 -- much higher than this
project's 0.829 -- but evaluated on m2cai16-tool-location (laparoscopic
cholecystectomy, 2532 frames), not GraSP. Not a fair comparison to our
number; recorded here only as evidence that transformer-based detectors are
a live, credible direction in this application area generally, the same
caution `docs/findings.md` already applies to TAPIS/LACOSTE comparisons.

## Net recommendation

1. ~~Try soft-NMS now~~ **Done, 2026-08-31: null result.** Re-scored the
   original baseline checkpoint's raw (un-suppressed) candidates with
   Gaussian soft-NMS instead of hard NMS -- mAP@50 0.826 vs. 0.829, heavy-
   occlusion recall 0.617 vs. 0.620, both within noise. See
   `docs/DECISIONS.md`. This is informative, not wasted: it means hard NMS
   discarding a second, well-placed box isn't the actual bottleneck, which
   points *away* from postprocessing fixes and *toward* recommendation 2.
2. **Design Milestone 9's segmenter around Kurmann et al.'s centroid/offset
   approach**, not a default Mask R-CNN -- now the most evidence-backed
   remaining option, since it addresses proposal/representation quality for
   occluded instances rather than how candidates are filtered afterward.
3. **Do not chase more generic detection augmentation** -- the literature
   explains why it's unlikely to help here, and this project already spent
   two runs confirming that empirically.
4. **State the temporal-context gap as an architectural limitation** in the
   report, not just an accuracy gap TAPIS wins on -- our detector's
   per-frame design is not directly comparable to TAPIS's video-level one,
   independent of parameter count.
5. **Add lightweight tracking-by-detection in Milestone 9**, not a full
   video transformer -- classical frame-to-frame track association (SORT/
   DeepSORT-class, near-zero added latency) targets the same occlusion
   problem González et al.'s temporal module does, without the scope
   increase or the latency cost of TAPIS's video-level architecture.
   Evaluate with the existing occlusion-stratified recall metric (tracked
   vs. untracked). If it closes a meaningful part of the gap, the heavier
   video-transformer route (Milestone 10+) may be unnecessary.
6. **Contingency, not a default plan: copy-paste augmentation** (Ghiasi et
   al. 2021, "Simple Copy-Paste is a Strong Data Augmentation Method for
   Instance Segmentation") -- pasting real instrument crops (using masks
   already available from GraSP's annotations) onto other real frames,
   deliberately targeting both class imbalance (paste more Clip
   Applier/Laparoscopic Grasper instances) and occlusion exposure (paste
   instruments so they overlap others already in frame) at once. Not a GAN
   -- no generative model, no mode-collapse risk, standard/validated
   technique. Explicitly held in reserve per the user's direction (2026-09-01):
   only build this if the recommendations above don't get within ~3% of
   the published literature's comparable range. Rejected a real GAN
   (BAGAN/CycleGAN-style class-imbalance synthesis) for this project even
   as a contingency -- Clip Applier has only 64-102 total instances,
   too few to train a generative model from scratch without a high risk of
   mode collapse or learned artifacts that hurt more than they help.

## Sources

- Deng et al., "Deep Learning for Surgical Instrument Recognition and
  Segmentation in Robotic-Assisted Surgeries: A Systematic Review,"
  arXiv:2410.07269 / *Artificial Intelligence Review* 2024.
  https://arxiv.org/abs/2410.07269
- "A comprehensive survey on recent deep learning-based methods applied to
  surgical data," arXiv:2209.01435.
  https://arxiv.org/pdf/2209.01435
- Kurmann et al., "Mask then classify: multi-instance segmentation for
  surgical instruments," *IJCARS* 2021.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8260538/
- Loza et al., "Real-time surgical tool detection with multi-scale
  positional encoding and contrastive learning," *Healthcare Technology
  Letters* 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11022231/
- Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of
  Code," ICCV 2017 (general CV, not surgical-specific; cited for the
  soft-NMS algorithm itself). https://arxiv.org/abs/1704.04503
