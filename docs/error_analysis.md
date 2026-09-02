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
