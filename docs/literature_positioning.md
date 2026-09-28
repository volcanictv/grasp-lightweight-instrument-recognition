# Literature positioning: the whole classifier pipeline, not just the tracking gate

Prepared for the Stanford AI+HEALTH talk (Dec 8, 2026) and an eventual paper. Covers the
pipeline as one system: task framing, ensemble composition, imbalance handling, preprocessing,
evaluation discipline, and the uncertainty-gated correction step, in that order, since that is
the order in which the pipeline's own decisions were actually made. Numbers below are this
project's own, already established; nothing here is invented or re-rounded.

## 1. Landscape, by part of the system

### 1.1 Task framing: per-instance region classification on ground-truth crops

GraSP's own paper (Ayobi et al., TAPIS) poses instrument recognition as end-to-end instance
segmentation: mAP@0.5IoU_segm, no classification-only baseline, no per-class breakdown. TAPIS
itself is a two-stage design (a segmentation model proposes regions, a transformer classifies
them), so a detect-then-classify split is not new; what TAPIS never does is measure the
classification stage alone against ground-truth localization to isolate "can the model name an
instrument once its location is given" from "can the model find it." MATIS is architecturally
the closer analog: a two-stage transformer (mask proposal, then classification with a
temporal-consistency module), evaluated on EndoVis 2017/2018, again always end-to-end.

We found no paper that explicitly frames the crop-and-classify step as a distinct, separately
reported task on GraSP or EndoVis, the way this project's Task B does (2,861 official-test
instances, crops built from the ground-truth mask, classification only). That is a real
framing choice worth stating plainly as such, not a preprocessing detail: it isolates
classification accuracy from segmentation accuracy, at the explicit cost of not being an
end-to-end number (already disclosed in the report's Limitations). One related precedent
exists in older laparoscopic-tool work comparing classification given manual boxes against
classification given detected boxes, but we did not find a clean, citable canonical paper for
this exact "oracle-crop versus end-to-end" framing as a named methodology; treat it as an
implicit assumption of most two-stage pipelines rather than a claim this project invented the
idea, but be explicit that GraSP-specific work has not reported it this way before.

### 1.2 Ensemble composition: two ResNet-50 variants, two MobileNetV3-Small variants

The shipped ensemble is not a diversity-engineered design. Per `docs/findings.md`
(2026-09-02/03), it emerged from combining checkpoints that already existed for other reasons:
a MobileNetV3-Small baseline and a letterbox-crop MobileNetV3-Small variant from the crop-style
ablation, plus two ResNet-50 checkpoints (224px and 320px) from the backbone-capacity sweep.
Flat averaging of all four reached macro-F1 0.8929 / accuracy 0.9266, "no new training." A
later attempt to reweight toward resnet50_320 (0.40) looked like a further gain on official
test (macro-F1 0.9031) but did not replicate on a genuinely held-out fold (fold1 slightly
favored flat weighting) and was documented as reverted in `findings.md`. The currently shipped
ensemble configs (`region_ensemble_deepdropout.yaml` and siblings) nonetheless use
weight_resnet50_320=0.40 again; we could not find a `findings.md` entry re-justifying this
second adoption. That gap is worth closing before a reviewer notices the two documents
disagree (see Section 4).

On the ensemble-diversity question itself: the literature is more skeptical than the intuitive
"more architectures, more diversity, better ensemble" story. Abe et al. (arXiv 2302.00704,
"Pathologies of Predictive Diversity in Deep Ensembles") find that deliberately engineering
diversity in high-capacity deep ensembles, via bagging, negative correlation learning, or
architectural heterogeneity, often does not help and can hurt, relative to just investing
capacity in the individual models; larger-scale comparisons of homogeneous versus
heterogeneous deep ensembles find heterogeneous mixes give only slightly better performance.
This actually supports the honest framing here: the ensemble's heterogeneity was not chosen to
maximize diversity, it was a byproduct of combining the best individually-found configurations
from separate ablations, and the literature does not promise that engineered diversity would
have done better anyway.

### 1.3 Class imbalance handling

`docs/imbalance_notes.md` already states plainly that most of this is standard long-tail
material, not novel, and that discipline holds up under a literature check. What was tried:
weighted loss (inverse-frequency), a weighted sampler, and augmentation, alone and combined;
what was documented but not adopted: class-balanced loss by effective number of samples (Cui
et al., CVPR 2019), focal loss as applied to surgical scenes (Urrea, Garcia-Garcia, Kern,
Biomedicines 2024), and synthetic rare-instrument compositing (Zhao et al., Medical Image
Analysis 2025, built for class-incremental segmentation, judged overkill here). MixUp/CutMix
was explicitly rejected as visually implausible for this domain, consistent with the project's
own augmentation constraints. This is a textbook-correct, appropriately conservative subset of
the long-tail toolbox for the problem's scale (worst-case 25.6x imbalance, not the extreme
long tails those heavier techniques target); nothing here needs defending as novel, and
`imbalance_notes.md` already says so itself.

### 1.4 Letterbox-crop preprocessing and the backbone sweep

Letterbox padding (scale to fit, pad to a square canvas, preserve aspect ratio) is standard
object-detection preprocessing, most associated with the YOLO family and implemented directly
in Albumentations; it is not a novel technique. What is specific here is applying it to
per-instrument crops rather than full frames, motivated by a documented error-case finding
(`findings.md`, 2026-09-02: raw bbox crops compress long, thin instruments, an error-case
diagnosis, not a guess) and confirmed by ablation (macro-F1 0.827 to 0.870 when swapping
MobileNetV3-Small for ResNet-50 under letterbox cropping, every class improving). The backbone
sweep itself (MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, ResNet-50 as a heavy
baseline) is a standard accuracy/params/latency tradeoff-curve exercise; the honest, specific
finding is that ResNet-50 bought +0.091 macro-F1 over MobileNetV3-Small at 15x the ONNX CPU
latency and 15x the model size, with no accuracy/latency crossover reversal, i.e., capacity
paid off cleanly on this task rather than plateauing. None of this needs a novelty claim; it is
solid, reported engineering work, and should be presented as such rather than dressed up.

### 1.5 Case-level splits, not frame-level

GraSP frames within a case are near-duplicates; the project has held to case-level (never
frame-level) splits throughout, per its own CLAUDE.md rule and validated in
`imbalance_notes.md` against Bradshaw, Huemann, Hu, Rahmim (*A Guide to Cross-Validation for
Artificial Intelligence in Medical Imaging*, Radiology: AI, 2023), which documents accuracy
inflated by up to 41% when patient identity leaks across splits in small medical-imaging
cohorts. This is not a project-specific insight; leakage from image-level (rather than
patient- or case-level) splitting is a broadly documented problem in medical imaging generally,
including recent 2025-2026 audits finding it in whole-slide-image and small-dataset
benchmarks. The point worth stating plainly in the talk: this discipline is what makes the
project's numbers comparable to GraSP's own official-split protocol, and it is not guaranteed
that other GraSP-adjacent work controls for it as carefully, since near-duplicate frame leakage
is exactly the kind of thing that inflates reported numbers silently.

### 1.6 Uncertainty quantification in surgical video, and the calibration-versus-discrimination split

MC Dropout (Gal & Ghahramani, 2016) and deep ensembles are both long-established;
neither is a contribution on its own. What is specific to this project is the argument for
choosing between them (and between them and softmax, and between them and evidential deep
learning) on threshold stability under shift rather than on AUROC. That argument's
foundational citations, Guo et al. (2017) on calibration and Ovadia et al. (2019, NeurIPS,
"Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset
Shift") on calibration collapsing under shift even when discrimination looks fine, are both
still standard and still cited (Ovadia et al. has over 1,300 citations and continues to be
cited in 2025 uncertainty work). A much stronger, more specific citation exists now: Pham et
al. (arXiv 2608.16748, "Beyond Uncertainty: Generalizable Failure Monitoring for Surgical
Segmentation under Acquisition Degradation") makes almost exactly this project's argument, on
EndoVis 2017 specifically. They show standard confidence-based uncertainty (softmax, entropy)
stays sharp even as segmentation quality collapses under acquisition degradation, explicitly
separate discrimination (AUROC) from calibration/threshold behavior, and report a single global
threshold producing false alarms on up to 40.4% of correctly segmented frames at moderate
corruption severity. This is independent, recent, EndoVis-specific confirmation of the exact
mechanism this project found on its own EndoVis zero-shot test (softmax and evidential both
rank well in-domain, both drift badly under shift): use this citation, it is stronger than
Ovadia alone because it is in the same clinical domain and the same dataset family.

Evidential deep learning (Duan et al., WACV 2024) is itself contested in the broader
literature independent of this project's own finding: Bengs et al. and a NeurIPS 2024 paper
("Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?") argue
evidential epistemic uncertainty is asymptotically unreliable (does not vanish even with
infinite data) and behaves more like an energy-based OOD score than a true epistemic measure.
This project's own result, that evidential fails threshold stability under EndoVis shift worse
than MC Dropout does, is consistent with that broader critique rather than an isolated
surprise; citing it strengthens the claim that this was a principled test of a contested
method, not an arbitrary one.

Conformal prediction is the newest entrant in this specific space: a MICCAI 2025 paper
("Conformal forecasting for surgical instrument trajectory") is, by its own description, the
first application of conformal prediction to surgical guidance, for instrument-trajectory
forecasting rather than classification. This confirms the broader field is moving toward
distribution-free, coverage-guaranteed uncertainty for surgical instruments, and that
classification-specific work in this vein is still thin. It is a candidate future-work citation
("pick a coverage-guaranteed threshold instead of an empirically fixed one"), not a method this
project currently uses or should claim to have beaten.

### 1.7 Uncertainty-gated selective computation, and where SAM2/ISINet sit relative to it

ISINet's own temporal-consistency module (González et al., MICCAI 2020, already cited in this
project's flow-tracking baseline) matches instrument instances across an entire sequence and
aggregates class predictions for every instance, unconditionally. It is not gated by
uncertainty; it is applied to everything. That is a real, checkable point of contrast: this
project's SAM2-tracking step is invoked only for the ~29% of instances the model's own
uncertainty flags, not for all instances, which is the entire reason the report can quote a
GPU-hour cost lower than "track everything."

The pattern of gating an expensive correction by an uncertainty or energy score is established
in machine learning generally, just not in this specific surgical-video-classification form.
ESCAPE (arXiv 2407.14605, energy-gated test-time adaptation for out-of-distribution 3D human
pose estimation) applies a cheap forward pass to in-distribution inputs and reserves costly
adaptation for inputs an energy threshold flags as OOD, reporting a 7x inference-time speedup
while keeping most of the accuracy gain; similar uncertainty-triggered selective-compute ideas
appear in click-through-rate prediction and code generation. In video segmentation
specifically, several 2025-2026 SAM2 variants (UAMP, motion-uncertainty-aware mask fusion, an
SAM2 "existence score" gate) use uncertainty to decide how much to trust propagated masks
per-frame or per-object, mainly to fight error accumulation over long videos, not to decide
whether to invoke propagation for a still-frame classification decision at all. We did not
find a published method that uses an ensemble/MC-Dropout vote-disagreement score specifically
to gate whether a surgical-instrument classification gets a SAM2 (or SAM2-style) temporal
correction. That combination, uncertainty-gated invocation of a video foundation model as a
post-hoc classification correction, appears to be a genuine, if narrow, gap this project fills,
not a wholly new idea (the gating pattern itself is known) but a new application of it.

## 2. Novelty assessment

**Not novel, and should not be presented as if it were:** MC Dropout itself, deep/architecture
ensembling itself, SAM2 itself, weighted loss and weighted samplers for imbalance, letterbox
cropping, case-level splitting discipline (all individually standard, and `imbalance_notes.md`
already says so for the imbalance techniques specifically).

**Thin but real, worth a sentence each, not a headline:** the ResNet-50/MobileNetV3
architecture and crop-style ablations (solid, ordinary engineering, not a contribution); the
class-imbalance interventions (correctly scaled to the problem, not novel).

**The actual contribution, in order of strength:**
1. The empirical argument that an uncertainty gate should be chosen for threshold stability
   under distribution shift rather than for in-domain discrimination (AUROC), demonstrated
   with four separate uncertainty sources on the same pipeline (softmax, MC Dropout,
   between-member disagreement, evidential), all four validated the same honest way (AUROC
   compared, then shift-tested), three of the four shown to fail the property that matters and
   kept in the report as negative results rather than dropped. This is a real methodological
   argument, independently supported by very recent, domain-specific literature (Pham et al.,
   2026) rather than asserted from first principles.
2. Uncertainty-gated invocation of SAM2 (or a comparable video model) as a selective, cost-aware
   correction step for instrument classification specifically, contrasted directly against an
   ISINet-style always-on temporal-consistency baseline (this project's own flow-tracking
   negative result already makes half of that comparison; the other half, applying tracking to
   every instance rather than a gated subset, is already quantified in the report's GPU-hour
   accounting).
3. A fully honest, case-level-split, negative-results-included account of a classifier system
   built from ordinary parts (ensembling, imbalance handling, backbone sweep, preprocessing),
   which is less a novelty claim than an evaluation-discipline claim: the numbers are more
   trustworthy than most because of what was checked and rejected along the way (the
   ensemble-weight overfitting catch, the contrastive-fine-tuning reversal on fold1, the
   384px-resolution reversal on fold2), not because any single piece is new.

## 3. Proposed narrative and contribution claim

Lead with the system, not the mechanism: this is a per-instrument surgical classifier built
from ordinary, individually well-established parts (a heterogeneous CNN ensemble, standard
imbalance handling, a backbone/preprocessing sweep), evaluated with unusually high discipline
(case-level splits, held-out reversals caught and reported, five architecture/hyperparameter
decisions that looked good on one split and were killed on a second), that adds one genuinely
new piece: gating an expensive video-based correction by an uncertainty signal chosen for
threshold stability, not ranking quality, and testing that choice against three alternatives
that all fail it under real domain shift.

Proposed contribution claim (two sentences): "We present a fully-ensembled, imbalance-aware
surgical instrument classifier whose accuracy claims are validated by aggressive held-out
reversal-testing rather than single-split reporting, and we show that its one add-on
component, an uncertainty-gated temporal correction, should be chosen for threshold stability
under distribution shift rather than for in-domain error-ranking quality: we demonstrate this
by validating and rejecting three alternative uncertainty sources (softmax, evidential deep
learning, and an ISINet-style always-on temporal baseline) that each look competitive or
superior in-domain but fail exactly the property the gate depends on."

For the talk specifically: open with the system diagram and the headline number (0.9623 /
0.9351), spend real time on the reversals (ensemble weighting, 384px resolution, contrastive
fine-tuning) as evidence of process rigor before mentioning uncertainty at all, then present the
uncertainty-gate argument as the one place genuinely new work happened, using the Pham et al.
2026 EndoVis result as external validation that the underlying concern (discrimination is not
calibration under shift) is real and current, not a house argument. Close on the EndoVis
zero-shot weakness (0.528/0.551) stated as a limitation, immediately followed by the
architecture-transfers-when-retrained result (0.85-0.93), since that is the most decision-
relevant single fact for a room of people deciding whether to trust a surgical AI system
outside its training distribution.

## 4. New reviewer-pushback risks found

- **The ensemble-weight documentation gap.** `findings.md` documents weight_resnet50_320=0.40
  as tested, found not to replicate on fold1, and reverted (2026-09-02/03). The shipped configs
  (`region_ensemble_deepdropout.yaml` and siblings) use 0.40 again, without a visible
  `findings.md` entry explaining the re-adoption. A reviewer who reads both documents will spot
  the contradiction. This should be reconciled (either a missing log entry needs writing, or the
  current default should be revisited) before it is asked about.
- **Ensemble diversity is a byproduct, not a design.** If the write-up ever frames the four-model
  ensemble as "diverse architectures chosen for complementary errors," a reviewer who knows the
  Abe et al. (2023) diversity-pathologies result will ask whether that diversity was validated
  to help versus just adding capacity, since the literature is skeptical of the intuitive story.
  The honest framing (found opportunistically, kept because it worked, not engineered for
  diversity) is more defensible and should be the one used.
- **"Oracle crop" framing has no clean prior citation.** Presenting Task B's ground-truth-crop
  classification as a deliberate, named methodological choice is fine, but claiming it as an
  established framing would be overclaiming; we found no canonical paper that names or defends
  this framing on GraSP or EndoVis. State it as this project's own scoping decision, disclosed
  as non-end-to-end, not as adopting a recognized protocol.
- **Evidential deep learning's contested status cuts both ways.** Citing the NeurIPS 2024 mirage
  critique to explain why evidential failed here is honest and helpful, but it also invites the
  question of why it was tried at all given the critique predates this project's own evidential
  experiment (2026-09-21). Be ready to say plainly that it was tried because a supervisor asked
  for a genuine attempt, not because the literature suggested it would work; the pre-registered,
  negative-result-reported process is the actual defense, not the outcome.

## Sources

- [Pixel-Wise Recognition for Holistic Surgical Scene Understanding (GraSP/TAPIS), Ayobi et al., arXiv 2401.11174](https://arxiv.org/abs/2401.11174)
- [MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation, arXiv 2303.09514](https://arxiv.org/abs/2303.09514)
- [ISINet: An Instance-Based Approach for Surgical Instrument Segmentation, MICCAI 2020, arXiv 2007.05533](https://arxiv.org/abs/2007.05533)
- [LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation, arXiv 2409.09360](https://arxiv.org/pdf/2409.09360)
- [Pathologies of Predictive Diversity in Deep Ensembles, Abe et al., arXiv 2302.00704](https://arxiv.org/abs/2302.00704)
- [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- [Beyond Uncertainty: Generalizable Failure Monitoring for Surgical Segmentation under Acquisition Degradation, Pham et al., arXiv 2608.16748](https://arxiv.org/abs/2608.16748)
- [Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html)
- [Conformal forecasting for surgical instrument trajectory, MICCAI 2025](https://papers.miccai.org/miccai-2025/0168-Paper0260.html)
- [ESCAPE: Energy-based Selective Adaptive Correction for Out-of-distribution 3D Human Pose Estimation, arXiv 2407.14605](https://arxiv.org/abs/2407.14605)
- [SAM 2: Segment Anything in Images and Videos, Meta, arXiv 2408.00714](https://arxiv.org/abs/2408.00714)
- [UAMP: Consistent video object segmentation with uncertainty-aware memory propagation, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0353156)
- [A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging, Bradshaw et al., Radiology: AI, 2023 (cited via this project's own docs/imbalance_notes.md; not independently re-verified against the original text in this pass)](https://pubs.rsna.org/journal/ai)
