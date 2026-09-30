# Evidential pipeline switch: evaluated and not adopted

> **Superseded 2026-09-29.** The user later decided to ship the single-pass evidential
> pipeline as the default anyway (cost and defensibility, not accuracy). The measurements
> below stand: about 0.7 to 1.0 point below the vote pipeline at matched budgets
> (`docs/reports/evidential_default/`). The default is now defined in
> `configs/pipeline_default.yaml`.

Requested by the user: replace the shipped MC-Dropout vote pipeline (base prediction and
tracking gate) with the single-pass evidential ensemble outright, keeping domain-shift
transfer as a disclosed con. Design pre-registered in `docs/DECISIONS.md`, 2026-09-28,
before any result existed. This does not change the shipped pipeline; it is a candidate
evaluated on the official test set and rejected based on the result.

## Bottom line

The evidential pipeline is worse than shipped, not a wash and not a cost-for-accuracy
trade worth taking. It scores lower accuracy and macro-F1, and its domain-shift transfer
is also slightly worse than the already-weak shipped-pipeline numbers, so there is no
compensating benefit on the one dimension it was hoped to help. Not adopted.

## Fold1 calibration (held out, trained on fold2, official test never touched)

Tracked 1,132 fold1 instances (top 35% by epistemic uncertainty) with SAM2, swept the
threshold. Selection rule, fixed in advance (same rule used for the 2026-09-20 MC-gate
confirmation: maximize accuracy, then take the highest threshold within 0.0010 of that
maximum): the nominal maximum (0.9607) sat at a 31% flagged share; the rule instead
selected the cheaper 20% point, fold1 accuracy 0.9604.

## Official test, threshold applied unchanged

| | accuracy | macro-F1 |
|---|---|---|
| Evidential base (untracked, single pass) | 0.9207 | 0.8867 |
| **Evidential gated pipeline (final)** | **0.9507** | **0.9279** |
| Shipped CE MC-vote pipeline | 0.9623 | 0.9351 |

539 of 2,861 instances tracked (18.8%, against the shipped pipeline's 833 / 29.1%).
Fixed 121, broken 35, 141 errors left. Every class improves under tracking (for example
Laparoscopic Grasper F1 0.764 to 0.836, Large Needle Driver 0.931 to 0.981), but the
final numbers still land below the shipped pipeline on both metrics.

## Domain-transfer con (the property the user asked to keep visible)

Official-split evidential ensemble, zero-shot, no retraining, no tracking:

| dataset | evidential | shipped CE pipeline |
|---|---|---|
| EndoVis 2018 | 0.499 / 0.390 | 0.528 / 0.414 |
| EndoVis 2017 | 0.516 / 0.501 | 0.551 / 0.570 |

Worse on both datasets and both metrics than the pipeline it would replace.

## Correction to the pre-registered assumption

Switching to evidential does not make tracking itself meaningfully faster: measured
about 21s/instance against the shipped pipeline's 24s/instance. SAM2 mask propagation,
not the classification passes, dominates tracking cost either way, so the real saving
from evidential is the cheap single-pass path for the ~81% of instances that never get
tracked, and tracking fewer instances overall (18.8% vs 29.1%), not faster tracking per
instance.

## Deliverables

`scripts/evidential_gate_fold1_calibrate.py`, `evidential_gate_official_prep.py`,
`evidential_gate_final_eval.py`, `evidential_endovis_con.py`,
`tests/test_evidential_gate_combine.py` (2 tests, passing).
`docs/reports/evidential_pipeline_switch/{results.json, fold1_calibration.json,
endovis_con.json, official_base.json}`. Outcome logged in `docs/DECISIONS.md`,
2026-09-28. `docs/reports/grasp_report_2026-09-01.html` was not touched.
