# Evidential trigger for the unanimous-error blind spot: does not work, and shows why

Requested as a narrower, additive justification for evidential uncertainty (docs/DECISIONS.md
2026-09-23, design pre-registered before any result existed): add an evidential-epistemic
trigger on top of the shipped 9% MC-vote gate, aimed only at the vote gate's structural blind
spot (unanimous errors, where all 80 MC-dropout votes agree on the wrong class, which no vote
signal can flag by construction). The shipped gate and SAM2 checkpoint are untouched; this is
an additive experiment.

## Bottom line

It does not work as a cheap patch, and the reason is informative: the unanimous-error instances
are not concentrated at high evidential uncertainty either. Catching all of them costs more than
half the test set, not a small addition. Stopped before running any SAM2 tracking, per the
pre-registered rule ("if this adds more than 5% of instances, report it, do not silently proceed")
and the parent's explicit stop-before-large-tracking-cost instruction. No pipeline change, no new
tracking spend beyond training the evidential members and single-pass extraction (no MC, no SAM2).

## What was built

Four evidential members (same four architectures/weights as the shipped ensemble, lambda=0.01,
anneal=10, unchanged from the 2026-09-21 validation), trained on data.split: official, the
shipped ensemble's own split (new: configs/evidential/edl_E_*_official_s42_lam0p01a10.yaml,
configs/evidential/ens_E_official_s42_lam0p01a10.yaml). Single deterministic pass per member on
official test (no MC dropout anywhere in this arm), weighted-mean Dirichlet alpha, epistemic trace
score S1 (surgical_ai/evaluation/evidential.py, unchanged).

## Fold1 calibration (before touching official test)

Held-out fold1 (3,235 instances, CE ensemble retrained on fold2, from the 2026-09-21 gate
confirmation): 198 ensemble errors, of which 5 are unanimous (all 80 votes agree, wrong) and
none are already caught by the 9%-equivalent gate (u >= 0.09 flags 862 instances but happens to
include none of the 5). Sweeping the evidential epistemic threshold down from strict to loose,
the smallest additional flag set that catches all 5 unanimous errors is 1,916 of 3,235
instances (59.2%), an estimated 12.8 extra GPU-hours of tracking on top of the existing budget.
The curve is not close to a reasonable budget at any point: at a 1% additional-flag budget none of
the five are caught, and full recall needs the majority of the dataset. Full curve:
docs/reports/hybrid_gate/tau_calibration.json.

This triggered the pre-registered stop condition (more than 5% added) and the parent's explicit
instruction (stop before tracking hundreds of newly-flagged instances). No SAM2 tracking was run
on fold1 or official test for this experiment.

## Official test, for the numbers themselves (no tracking spend needed for these)

Sanity check: the evidential-official-test analysis reproduces the shipped ensemble's own base
accuracy and macro-F1 exactly (0.9266 / 0.8898) from the same MC-vote cache used everywhere else,
confirming the comparison is apples to apples.

- 8 unanimous errors on official test (matches the report's "8 of the 210" limitation), none
  caught by the 9% gate.
- Evidential error-detection AUROC on official test: 0.9165, against the MC-vote gate's own
  0.8962 on this exact cache, replicates the fold1 pattern (0.927 vs 0.912) at a smaller margin.
  Evidential is a genuinely better error-ranker here too, consistent with the earlier validation.
- But the 8 unanimous errors are not where evidential uncertainty is high. Their epistemic-score
  percentile ranks among all 2,861 instances: 16.9, 20.1, 42.6, 48.5, 52.8, 73.9, 81.0, 85.7. Three
  of the eight sit in the bottom half of the uncertainty distribution, evidential is relatively
  confident on them too, just confidently wrong, the same failure mode as the vote gate. This is
  the direct explanation for why no cheap threshold can isolate them: they don't look uncertain to
  either method, only wrong in hindsight.
- Applying the fold1-calibrated tau unchanged to official test: flags 1,475 additional instances
  (51.6% of the test set, an estimated 9.8 extra GPU-hours) and still only catches 7 of the 8 known
  unanimous errors, not all of them, the fold1-fitted threshold does not even transfer perfectly.
- Illustrative smaller budgets (score-based projections only, not tracked, so their eventual
  accuracy effect on the caught instances is unverified): 1% additional (29 instances) catches 0 of
  8; 2% (57 instances) catches 1; 5% (143 instances) catches 2; 10% (286 instances, about a third of
  the existing 833-instance tracking budget) catches 3 of 8.

## Why this matches the report's own documented mechanism

The report's Limitations section already names the reason a still frame can be confidently wrong
under every signal: a genuine single-frame information ceiling (closed/featureless instrument
states that are visually identical across classes, e.g. Grasper vs. Suction, Bipolar vs. Needle
Driver, Monopolar vs. Suction). If an instrument's true identity simply is not visible in the
frame, no uncertainty signal computed from that frame, softmax, MC-dropout vote, or evidential
variance, has a reason to look uncertain. This experiment is consistent evidence for that
explanation rather than a coincidence: switching which uncertainty axis is used does not surface
these errors cheaply, because the ambiguity is not in the signal, it is in the input.

## Recommendation

Do not adopt an evidential trigger for this purpose. The honest framing for the report, if this is
mentioned at all, is a second confirmation of the single-frame information ceiling already
documented, not a new lever. If catching more of these 8 errors is still wanted, the real levers
are the ones already in Limitations and Next steps (more training data for the confused pairs,
which cannot fix a true information ceiling but can help the genuine-similarity share of the
confusion), not a cheaper uncertainty signal, since the finding here is that the failure mode is
shared across signals, not signal-specific.
