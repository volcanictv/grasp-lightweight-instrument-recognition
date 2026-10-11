# Refinement transitions (official test set, final pipeline, seeds 42, 43, 44)

Produced by `scripts/refinement_transitions.py` (CPU only, about 6 s on titanxp, no training, no new gate or threshold). Files:

- `refinement_transitions.csv`: 8,583 rows (2,861 instruments x 3 seeds). Columns: seed, instrument index, case, frame, file, box (xywh), ground truth, single-pass class and confidence, refine-all class and confidence, S1, maximum belief, S1 rank (0 = highest), `has_valid_track`, `transition` (wrong_to_correct, correct_to_wrong, correct_to_correct, wrong_to_wrong, or missing_track) and one `sel_<policy>` flag per policy.
- `refinement_summary.json`: per seed and policy, transitions of the selected and of the unselected instruments (the latter with their stored refine-all outcome, i.e. what refining them would have done), the six required quantities, the per-case four transition counts (all instruments and per policy, selected and unselected), the identity checks, and the mean over seeds.

## Inputs
`experiments/gtbox_sam/final` on titanxp (same inputs as `scripts/refine_all_analysis.py`): `logits_s<seed>.npz` (four-member ensemble logits, same fusion weights, S1 = evidential epistemic score) and the refine-all tracks `tracked` + `tracked_rest` (seed 42), `tracked_s<seed>` + `tracked_rest_s<seed>` (seeds 43, 44), fused with `fuse()` of `scripts/gtbox_sam_final_eval.py`. Instrument identity (file, box, label) comes from `GraspRegionDataset("test")`.

## Missing outputs
None. All three seeds have a valid track for all 2,861 instruments. If a track were missing the script stops (unless `--allow-missing`, which excludes such instruments from every count and lists them); missing tracks are never treated as unchanged predictions.

## Policies
`tau_1.7e-5` (S1 >= 1.7e-5), `s1_top833` (833 highest S1), `maxbelief_n_tau` and `maxbelief_n_top833` (same number selected, ranked by 1 - largest belief of the same ensemble), `refine_all`. Ranking ties are broken by lower instrument index. The two maxbelief policies are the matched-count comparators, not new gates.

## Checks (all passed, recorded in the summary)
- Stored labels equal the dataset label for every instrument, per seed (not just equal length); (file, box) is unique.
- Identity by content: the centre frame of each instrument's track reproduces that instrument's single-pass logits for every member (max gap 0 for seed 42, 3.4e-5 and 4.1e-5 for seeds 43, 44, float rounding), 2,861 of 2,861 per seed. No duplicate or out-of-range track indices.
- The separately stored `tracked_all_s<seed>` tracks equal the tracks used (2,861 of 2,861 per seed).
- For every policy, net corrections / N equals the change in instance accuracy (asserted exactly).
- Ratios with a zero denominator are `null`. Classification transitions are not mIoU changes; mIoU is not computed here.

## Reading the numbers (mean of three seeds, N = 2,861)
| policy | selected | corrections | errors introduced | net | recoverable captured | harmful avoided |
|---|---|---|---|---|---|---|
| tau 1.7e-5 | 44.9% | 171.0 | 30.3 | +140.7 | 98.8% | 18.7% |
| S1 top 833 | 29.1% | 163.7 | 27.0 | +136.7 | 94.5% | 27.7% |
| max-belief, n of tau | 44.9% | 171.3 | 30.3 | +141.0 | 99.0% | 18.7% |
| max-belief, n of top 833 | 29.1% | 165.7 | 26.3 | +139.3 | 95.7% | 29.6% |
| refine all | 100% | 173.0 | 37.3 | +135.7 | 100% | 0% |

Refine-all fixes 173 and breaks 37 instruments on average; selective refinement keeps almost all fixes but avoids at most about 30% of the breaks, so the gate selects mostly where tracking helps but barely avoids where it hurts. The 833 budget was chosen on this test set (see the paper caveats). At matched counts the maximum-belief ranking is equal to or slightly ahead of S1 on every net column.
