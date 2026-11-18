# LG failure investigation (2026-11-18), GraSP official test, ground-truth masks

What was looked at and what came out. Nothing here changed the paper's numbers.

- `rows.json`: for 28 inspected instruments (LG missed while TAPIS is right, LG correct, SI correct, LG called SI), the ensemble probabilities of the original crop and of silhouette / scrambled-texture / blur versions (scripts/inspect_members.py).
- `track_probe.json`: per-frame P(LG) along the +-10 s track of the same instruments (scripts/track_probe.py). About half of the misses have a jaw-visible frame (P(LG) 0.83 to 0.97) that the mean fusion averages away, the other half never do (max 0.13 to 0.46); a true SI instrument has frames up to 0.95.
- `cue_pooling.json`: class-specific top-k pooling for LG and CA (scripts/cue_pooling_eval.py). Rule fixed in advance (accuracy and macro-F1 must improve on the five CV folds); not met, not adopted. LG recall +5.8 but precision 93 to 81-85 on test.
- `geometry_evidence.json`: box-geometry evidence as one decode term (scripts/geometry_evidence_eval.py). lambda 0.25 chosen on the two development folds; test mIoU / IoU / mcIoU 96.83 / 95.81 / 90.32 against 96.72 / 95.69 / 90.24, not consistent across seeds, within noise, not adopted. (The fold selection used a scratch script, not saved here: logistic regression on the same features, folds fitted on the other cases.)
- `neurons.py`: decomposition of the ResNet-50 head's LG-minus-SI margin into channels (needs feats.npz from inspect_members.py). The difference is diffuse (top 20 channels carry 23 percent, top 100 carry 59 percent) and swapping those channels also flips 7 of 8 correct SI instruments, so they are the true discriminators and there is nothing to prune.
