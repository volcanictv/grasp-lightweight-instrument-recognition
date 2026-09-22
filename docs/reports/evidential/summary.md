# Evidential (Dirichlet, variance-based) uncertainty: validation result

Requested by the user's PI. Design pre-registered in `docs/DECISIONS.md`, 2026-09-21, before
any evidential result existed. Source: Duan, Caffo, Bai, Sair, Jones, "Evidential Uncertainty
Quantification: A Variance-Based Perspective", WACV 2024 (arXiv 2311.11367).

## Bottom line

Evidential variance-based uncertainty detects errors about as well as the MC-dropout vote gate,
at a fraction of the cost (1 forward pass instead of 80), and it catches more of the unanimous
errors the vote gate is blind to. That clears the pre-registered go rule for all four candidate
signals (S1-S4). But it fails the threshold-stability claim it was meant to help with: under
domain shift to EndoVis, its threshold drifts more than the vote gate's does, not less, and the
evidential-trained ensemble itself is less accurate zero-shot on EndoVis than the CE-trained one.
Recommend: worth a downstream tracking trial for the cost win, not as a shift-robustness fix.

## Lambda selection (pre-registered: resnet50_320 member, fold1 seed 42, held-out macro-F1 only)

| lambda | anneal epochs | held-out macro-F1 | held-out accuracy |
|---|---|---|---|
| 0.01 | 10 | **0.8853** | 0.9332 |
| 0.1 | 10 | 0.8828 | 0.9292 |
| 1.0 | 10 | 0.8645 | 0.9184 |
| 1/7 (0.1429) | 0 (no anneal) | 0.8699 | 0.9206 |

Chosen: lambda = 0.01, annealed linearly over 10 epochs. Applied unchanged to every other member,
seed and fold. Evidence clamp: logits clipped to [-10, 10] before exp (no softmax anywhere in the
evidential arm).

## Ensemble accuracy, E (evidential) vs C (CE deep-dropout, MC vote)

| target | acc C | acc E | macro-F1 C | macro-F1 E |
|---|---|---|---|---|
| fold1, seed 42 | 0.9388 | 0.9376 | 0.8989 | 0.8990 |
| fold1, seed 43 | 0.9385 | 0.9406 | 0.8978 | 0.9041 |
| fold1, seed 44 | 0.9385 | 0.9332 | 0.8975 | 0.8797 |
| fold2, seed 42 | 0.9284 | 0.9271 | 0.9028 | 0.9039 |

E is within about 1.5 points of C on every seed. No collapse; the two objectives train comparable
classifiers.

## Error-detection AUROC, held-out fold1 (mean +/- std over seeds 42/43/44, C also 3-seed)

| signal | AUROC | recall@10% | recall@20% | recall@30% | unanimous flagged@20% |
|---|---|---|---|---|---|
| S1 epistemic trace | 0.9271 +/- 0.0036 | 0.622 | 0.893 | 0.961 | **0.736** |
| S2 aleatoric trace | 0.9279 +/- 0.0056 | 0.622 | 0.880 | 0.954 | 0.627 |
| S3 total trace | 0.9279 +/- 0.0057 | 0.623 | 0.880 | 0.954 | 0.627 |
| S4 evidence-only (1/(a0+1)) | 0.9155 +/- 0.0042 | 0.594 | 0.857 | 0.943 | **0.780** |
| S5 member disagreement (no MC) | 0.8454 +/- 0.0164 | 0.607 | 0.776 | 0.826 | 0.000 |
| S6 rank-avg(S4, B1) | 0.9253 +/- 0.0022 | 0.635 | 0.882 | 0.958 | 0.672 |
| B1 MC vote disagreement (main method) | 0.9124 +/- 0.0104 | 0.625 | 0.835 | 0.925 | 0.000 |
| B2 softmax confidence (baseline) | 0.9274 +/- 0.0086 | 0.656 | 0.866 | 0.941 | 0.000 |
| B3 member disagreement, dropout off | 0.8504 +/- 0.0202 | 0.585 | 0.803 | 0.840 | 0.001 |

S1-S3 and S6 all beat B1's mean AUROC (0.9124), by 1.3 to 1.6 points, with lower seed-to-seed
variance. S4 is within 0.3 points below (inside the pre-registered -0.01 margin). Every evidential
signal catches more errors at the 20% and 30% budgets than the vote gate. The fold2 replication
(1 seed, no tuning) agrees: S1-S3 AUROC 0.930-0.932 against B1's 0.930.

**Unanimous errors** (the vote gate's stated blind spot) are defined per arm on its own agreement
rule: for C, all 80 votes name the wrong class (5 of 198 fold1-seed42 errors); for E, all four
members agree on the wrong class, a much weaker condition since it needs only 4, not 80, opinions
to align (46 of 202 errors). Because of that asymmetry the raw counts are not comparable, but the
question that matters, "does each arm's own uncertainty score flag its own unanimous errors", is
self-referential and fair: B1 cannot flag any of C's unanimous errors by construction (u=0 for all
of them), while S1 flags 74% and S4 flags 78% of E's unanimous errors at a 20% review budget.

Per-case AUROC (fold1 seed 42, 4 held-out cases) for S1: 0.918/0.940/0.947/0.938; for B1:
0.879/0.907/0.911/0.922. S1 is ahead on every case. Instance-bootstrap 95% CI for S1 is
[0.916, 0.945] and for B1 [0.880, 0.922]; these overlap, and with only 4 held-out cases the
bootstrap understates the real uncertainty (full per-seed CIs and per-case numbers are in
`results.json`).

## Threshold stability (fixed on fold1 seed 42 at a 25%-flagged threshold, applied unchanged)

| target | S1 share flagged | S1 error catch | S4 share flagged | S4 error catch | B1 share flagged | B1 error catch |
|---|---|---|---|---|---|---|
| fold1 seed 42 (reference) | 25.0% | 92.1% | 25.0% | 90.1% | 25.3% | 86.4% |
| fold2 (fold2-direction models) | 21.2% | 89.7% | 19.7% | 87.4% | 22.4% | 88.1% |
| EndoVis 2018 (zero-shot) | 84.5% | 89.6% | 84.1% | 89.7% | 77.5% | 81.6% |
| EndoVis 2017 (zero-shot) | 86.9% | 98.2% | 85.7% | 97.2% | 82.8% | 92.4% |

Within GraSP (fold2, a held-out case split from the same dataset), both signals hold close to
25%. Under real domain shift (EndoVis), every signal collapses toward flagging almost everything,
and the evidential signals drift further than the MC-vote gate (about 85% flagged against about
80%). This is the opposite of what the "threshold stability" hope needed: evidential does not
transfer its operating point better than the vote gate does. It also is not informative once
80%+ of instances are flagged, since a gate that flags almost everything is not selective.

Zero-shot ensemble accuracy also favours C on EndoVis: 2018 accuracy 0.533 (C) vs 0.457 (E);
2017 accuracy 0.594 (C) vs 0.501 (E). The evidential loss produces a somewhat weaker classifier
under this shift, independent of its uncertainty signal.

## Pre-registered go rule for stage B (downstream tracking)

Rule: a signal S1-S4 has mean AUROC >= B1's mean AUROC - 0.01 on fold1, AND (it flags more
unanimous errors than B1 at a 20% budget, OR its EndoVis flagged share stays within 10 points of
25% while B1's does not).

Result: S1, S2, S3 and S4 all meet the rule (auroc_condition true, unanimous_more_than_B1
true for all four; the EndoVis-stability clause is false for all four, since evidential is the
one that drifts more, not less). The rule passes on the AUROC-and-unanimous-errors branch alone;
the shift-stability branch is a clear negative result, not a pass.

## Caveats

- Fold-trained members see half the training cases; the held-out fold is 4 cases, so the
  effective sample size for AUROC and the stability test is close to 4, not thousands.
- fold2 and the EndoVis runs are single-seed (no tuning), by design, as a clean replication check.
- Stage B (does an evidential gate actually improve the tracked pipeline's accuracy) was not run;
  this is the AUROC/threshold-detection stage only.
- Unanimous-error counts for E and C use different agreement conditions (4 members vs 80 votes)
  and are not directly comparable in absolute count, only in each arm's own flag rate on its own
  unanimous set (see above).
- lambda was selected on one member's held-out macro-F1 only, not on any uncertainty metric, to
  avoid picking lambda for a favourable AUROC.

Full per-seed and per-case numbers, bootstrap CIs and stability tables: `results.json`.
