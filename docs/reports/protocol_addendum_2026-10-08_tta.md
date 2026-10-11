# Protocol addendum, 2026-10-08: decision rules for the compute-matched TTA control, fixed before the result is read

Extends docs/reports/gtbox_sam_protocol.md.

- Control. scripts/tta_control.py classifies 21 augmented views of each instrument's keyframe crop (3 mask variants x 7 transforms, all settings fixed in the script) with the final members of the 3 seeds, fuses them with the same confidence weighting as the 21 propagated frames, and scripts/gtbox_sam_final_eval.py scores the result at tau 1.7e-5 and at refine-all. The run started 2026-10-08 at 04:25 on titanxp; no result had been read when this addendum was written. View 0 reproduces the single-pass logits (max abs difference 1.6e-5 on 6 instruments).
- Rule. Let g = (TTA-on-all mIoU minus single-pass mIoU) / (refine-all mIoU minus single-pass mIoU), as the 3-seed mean on the test cases. The paired case-bootstrap interval of (refine-all minus TTA-on-all) is reported. mcIoU is checked in the same direction.
  - T1: g <= 0.33 and the paired interval excludes 0. The paper states that the gain comes from following the instrument through neighbouring frames.
  - T2: 0.33 < g < 0.67. About half of the gain is available from cheap views; tracking adds the rest.
  - T3: g >= 0.67, or the paired interval includes 0. The paper states that most of the second-look gain comes from more views of the same instrument, not from temporal context, and reports the gate for both operators.
- The name "tracking" stays for the propagation step in every outcome.
- Also registered here: the baselines on the test crops (softmax, MC dropout, deep ensemble) use cross-entropy members trained on the official training cases with the recipe of the evidential finals (scripts/make_ce_official_configs.py), the test cases are never read in training, and all signals listed in scripts/baselines_on_test.py are reported whatever the result.
