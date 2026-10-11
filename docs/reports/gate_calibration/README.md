# Development-calibrated gate thresholds (follow-up evaluation)

CPU analysis of stored predictions, no training and no model refitting. Script: `scripts/gate_calibration.py`. Files here:

- `gate_calibration.json`: the frozen configuration (design, thresholds with full precision, tie rule, case roles, SHA-256 of the inputs, coverage pre-check). Committed (75218b6) **before** `evaluate` was run.
- `gate_evaluation.csv`, `gate_evaluation.json`, `evaluate.log`: the one evaluation of those thresholds (per case, per fold, pooled).

## Task 2A: which checkpoint sets can calibrate and evaluate

A calibration case must be outside training and checkpoint selection of every learned component whose output feeds the gate (classifier members, SAM2/SAM3 segmenters). "Final recipe" = arm N evidential classifier members, SAM2 + SAM3 ensemble masks, fusion and ensemble weights, SAM2-large tracking.

Cases: official train = 001, 002, 003, 004, 007, 014, 015, 021 (fold1 = 001, 004, 014, 015; fold2 = 002, 003, 007, 021); official test = 041, 047, 050, 051, 053.

| Checkpoint set | Classifier training cases | Segmenter fine-tuning cases | Available development cases | Final recipe? | Cached predictions? |
|---|---|---|---|---|---|
| Final all-case pipeline, arm N, seeds 42/43/44 | 8 official train (fixed 20 epochs, last epoch, fold1 only monitored) | same 8 (SAM2 5 epochs, SAM3 3 epochs, last epoch) | none: every non-test case is in training | yes (defines it) | yes, all 2,861 test instruments, refine-all tracks, 3 seeds |
| Ladder rung gtft_ens_armN | 8 official train (same arm N members) | fold2 (002, 003, 007, 021); epoch and flip rule chosen on fold1 | none: fold1 is in classifier training and in segmenter selection | no (fold-trained, fold1-selected segmenters) | yes (official test) |
| Fold-trained arm N classifiers (armN_*_fold1_s42/43/44, armN_*_fold2_s42) | fold2 cases (evaluated on fold1) or fold1 cases (evaluated on fold2); 3 members, no resnet50_224 | SAM2 enc4 (fold2, epoch chosen on fold1), enc4_f1 (fold1, 5 fixed epochs); SAM3 fold2 only | none usable: best epoch chosen on the held-out fold (select_best default) | no (3 members, no fold1-trained SAM3) | classifier logits on the held-out fold only |
| Earlier evidential members behind tau = 1.7e-5 | fold2 cases (pre arm N) | n/a (earlier pipeline) | fold1 was used to choose tau, so not independent | no (no arm N) | not reused |
| CV fold cv5_f0 | 11 cases: 001, 002, 003, 004, 014, 015, 021, 041, 047, 051, 053 (monitor case 051 inside) | same 11 (SAM2 5, SAM3 3 epochs, last epoch) | held out 007, 050; official-train: **007** | yes | logits + masks (all held-out), tracks for the top 50% by S1 |
| CV fold cv5_f1 | 10: 001, 002, 003, 004, 007, 014, 015, 041, 050, 051 | same 10 | held out 021, 047, 053; official-train: **021** | yes | same |
| CV fold cv5_f2 | 10: 001, 002, 004, 007, 015, 021, 047, 050, 051, 053 | same 10 | held out 003, 014, 041; official-train: **003, 014** | yes | same |
| CV fold cv5_f3 | 10: 002, 003, 004, 007, 014, 021, 041, 047, 050, 053 (dev/monitor case 047 inside) | same 10 | held out 001, 015, 051; official-train: **001, 015** | yes | same |
| CV fold cv5_f4 | 11: 001, 003, 007, 014, 015, 021, 041, 047, 050, 051, 053 | same 11 | held out 002, 004; official-train: **002, 004** (no official-test case) | yes | same |

The CV helper's `_dev` case (051, or 047 for fold 3) is inside the training cases and only monitors a fixed schedule, so it is not a calibration case. Classifier seed: 42 only per CV fold (seeds 42/43/44 exist only for the final all-case set); fold 0 also has two extra SAM2 seeds.

**Eligible combination: yes. Folds cv5_f0 to cv5_f3 each hold out official-training cases (calibration) and official-test cases (evaluation) that no learned component of that checkpoint set trained on or was selected on, with cached predictions for all of them; the final all-case checkpoints have no eligible development case.**

## Task 2B: design (frozen before evaluation)

- Roles (my assumption, stated because the request left them open): calibration cases = held-out official-training cases of the fold; evaluation cases = held-out official-test cases. Fold 4 has no test case, so it is calibrated but not evaluated. Every official-test case appears in exactly one fold, so the pooled evaluation covers all five test cases, each scored by a model that did not train on it.
- Scores: S1 (primary) and 1 - largest belief (comparator), same four-member evidential ensemble. Target: 30% refined on the calibration cases, a chosen budget, not a discovered optimum. Cutoff = the distinct calibration score whose selected fraction (score >= cutoff) is closest to 30%, exact distance ties to the larger cutoff. Separate cutoffs per fold (each fold is an independently trained checkpoint set). Masks, preprocessing, tracking window (+-10), fusion unchanged. Cutoffs were never adjusted on evaluation cases.
- Coverage pre-check (scores only): the campaign tracked only the 50% highest-S1 instruments of each fold; every instrument the frozen cutoffs select in the evaluation cases has a stored track (0 untracked for both scores in all four folds). A selected instrument without a track would have stopped the run, never counted as unchanged.

## Results (one evaluation, no re-runs)

Calibration cutoffs and fractions (calibration cases / evaluation cases):

| Fold | Calibration cases (n) | S1 cutoff | S1 dev / eval fraction | Max-belief cutoff | Max-belief dev / eval fraction |
|---|---|---|---|---|---|
| 0 | 007 (567) | 3.54e-06 | 30.0% / 31.8% | 4.19e-03 | 30.0% / 31.8% |
| 1 | 021 (958) | 1.05e-05 | 30.0% / 33.4% | 6.71e-03 | 30.0% / 34.3% |
| 2 | 003, 014 (1,436) | 1.97e-05 | 30.0% / 34.5% | 1.06e-02 | 30.0% / 35.5% |
| 3 | 001, 015 (1,602) | 3.23e-05 | 30.0% / 30.9% | 1.46e-02 | 30.0% / 28.9% |
| 4 | 002, 004 (1,607) | 7.58e-05 | 30.0% / not evaluated | 2.29e-02 | 30.0% / not evaluated |

Evaluation cases (mIoU / IoU / mcIoU; instance accuracy in %). S1 and max-belief outputs are identical in every fold: the two selections differ by up to 14 instruments per fold (4 to 14 in one direction) and none of the differing instruments changes its label (checked directly).

| Scope | n | Single pass | S1 gate | Max-belief gate | Refined (S1 / MB) |
|---|---|---|---|---|---|
| fold 0 (case 050) | 1,055 | 84.84 / 82.85 / 68.46; 93.0 | 86.69 / 85.62 / 76.61; 95.7 | same | 31.8% / 31.8% |
| fold 1 (047, 053) | 1,024 | 86.83 / 85.53 / 71.28; 94.0 | 88.40 / 87.71 / 76.61; 96.3 | same | 33.4% / 34.3% |
| fold 2 (041) | 588 | 83.95 / 81.31 / 68.56; 90.3 | 87.60 / 86.01 / 75.42; 94.7 | same | 34.5% / 35.5% |
| fold 3 (051) | 194 | 88.97 / 88.14 / 81.47; 95.9 | 89.27 / 88.49 / 82.37; 96.9 | same | 30.9% / 28.9% |
| **pooled, all five test cases** | 2,861 | 85.67 / 83.88 / 71.78; 93.0 | 87.69 / 86.68 / 77.93; 95.8 | 87.69 / 86.68 / 77.93; 95.8 | 32.9% / 33.3% |

Per case (single to gated mIoU, S1 = max-belief): 041 83.95 to 87.60; 047 85.76 to 87.06; 050 84.84 to 86.69; 051 88.97 to 89.27; 053 87.60 to 89.35 (`gate_evaluation.csv` has IoU, mcIoU, accuracy).

Compute at the actual selected fraction (paper convention, `docs/reports/paper_gflops.json`: single pass 2,946 GFLOPs per instrument plus fraction x 25,584 per refined instrument): S1 11,360 GFLOPs per instrument, 60.2% below refining every instrument (28,530); max-belief 11,459, 59.8% below. A100 time per instrument on the same model: 0.48 s single pass, 2.48 s at 32.9% refined, 6.57 s refining everything.

Refine-all: not available for these checkpoint sets (only the 50% highest-S1 instruments of each fold are tracked), reported as null. It exists only for the final all-case set (`docs/reports/refinement_transitions`).

## Caveats

- Not the final checkpoints. Each fold's classifiers and segmenters are trained on 10 or 11 cases that include other official-test cases, so these numbers are not comparable with the official-split benchmark (single pass is 85.67 here against 83.47 for the final all-case pipeline). They answer only whether a calibrated threshold transfers.
- One classifier seed per fold (seed 42), four folds evaluated, five evaluation cases (194 to 1,055 instruments each). No intervals or significance tests were computed; fold 3 (194 instruments) and the per-case gains (+0.3 to +3.6 mIoU) show the spread. Thresholds differ ninefold across folds (S1 3.5e-06 to 7.6e-05; the old shared tau is 1.7e-05), because single-seed members differ in confidence; a threshold does not transfer across checkpoint sets, a fraction calibrated per set does.
- Evaluation fractions land 28.9 to 35.5% for a 30% target: the dev to evaluation shift in score distribution is visible.
- Label flips are only known inside the tracked top-50%-by-S1 set; instruments outside it were never tracked. S1 is not better than the maximum-belief comparator here (identical outputs).
- Recipe-level exposure remains: the ensemble weights were set on test cases, the segmenter schedule, ensemble and flip rule on fold1, and the 30% budget is motivated by earlier test-set work. This is a follow-up evaluation, not a prospective one.
