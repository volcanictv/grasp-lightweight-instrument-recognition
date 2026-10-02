# Training experiment 4: neighbour crops from the deployed YOLO tracker

Written 2026-10-02, before any model for this experiment is trained.

Background (docs/reports/training_experiments_preregistration.md, results in docs/reports/training_arms/): arm N, which swaps a
training crop for an EdgeTAM crop of the same instrument at a nearby frame, improved the three-member ensemble on fold1 and
fold2 (accuracy +0.6 and +0.85 points) and cut the number of correct predictions broken by EdgeTAM tracking on the official
test (38 vs 60 at 539 tracked). Through the YOLO26s tracker it gave no gain (0.9193 vs 0.9210 at 539 tracked). Hypothesis
tested here: the benefit is specific to the tracker whose masks the training crops came from, so crops made with the YOLO tracker
help the YOLO tracker.

## Arms (training split only; everything else identical to the baseline and arm N)

| arm | training crops |
|---|---|
| Y | with probability 0.5 swap in a YOLO-tracker crop of the same instrument, 1 to 5 frames away |
| NY | with probability 0.5 swap in a crop drawn uniformly from the union of the EdgeTAM and the YOLO crops |

YOLO crops are made by scripts/build_temporal_neighbors_yolo.py with the settings of tracker B (conf 0.1, matching min-iou 0.1,
coast 3, centre-frame fallback, imgsz 640) and the same area guard [1/3, 3] as arm N. The YOLO weights are cross-fitted: the
crops of the fold2 cases come from the YOLO trained on the fold1 cases, the crops of the fold1 cases from the YOLO trained on the
fold2 cases, so the masks have the quality YOLO has on cases it never trained on. Members, seeds 42, 43, 44, loss and schedule
are those of the baseline. The baseline and arm N runs already exist and are reused.

## Protocol and decision rule (unchanged from experiment 2 and 3)

1. fold1 development, 3 seeds, held-out fold1. Primary: three-member ensemble alone, accuracy and macro-F1. Secondary: accuracy
   on held-out EdgeTAM-style crops and on held-out YOLO-style crops of the fold1 instruments. An arm improves only if its mean
   accuracy beats the baseline mean by more than the larger of the two seed standard deviations.
2. fold2 confirmation, seed 42: same sign of the accuracy difference against the fold2 baseline.
3. The arm used for the final model is the confirmed arm (fold1 rule met and fold2 same sign) with the larger fold1 mean
   accuracy; a tie goes to NY. If neither is confirmed, nothing is trained for the official split and the result is reported as is.

## Final model

Same recipe as the first final model: three members, 8 official training cases, seed 42, 20 epochs, last-epoch weights,
validation on the fold1 cases (inside the training data), neighbour crops from the fold1 and fold2 generator outputs.
Evaluated once on the official test through both trackers with the locked settings (EdgeTAM: 20 causal frames, gate 2.85e-4;
YOLO26s: 15 causal frames, gate 5.75e-4). Because a less confident model sends more instruments to tracking at a fixed gate,
every tracked result is also rescored at the baseline members' tracked counts (373 for EdgeTAM, 313 for YOLO) and at 539, from
the same tracking outputs, so the comparison is compute-matched. All rows are reported, including negative ones.

## Not done, on purpose

No change to a gate, fusion rule, look-back or tracker setting; no selection on the official test.
