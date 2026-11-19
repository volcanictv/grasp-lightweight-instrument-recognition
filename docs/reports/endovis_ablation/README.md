# EndoVis 2018 ablation, seed 42 (one seed, 2026-11-18/19)

ISINet split, annotation masks, members trained on EndoVis 2018 + 2017, revisit budget 941/2861 (32.9%) everywhere, tracks of all 1,441 validation instruments (SAM2, +-10 frames).
`endovis18_ablation_seed42.json` holds every table and the paired frame-bootstrap contrasts (produced by `scripts/endovis_ablation_eval.py`); `dup17.npz` are the 2017 duplicate-class counts the
soft frame decode needs; `perclass_single_pass.py` compares single-pass per-class recall and precision of the ensemble and the DINO models.

One seed, DINOv2/DINOv3 are single runs. Read the contrasts, not the point values. Headline contrasts (mean difference, 95% interval, argmax, mIoU / IoU / mcIoU):
revisit vs none +4.27 / +4.91 / +7.21; ensemble vs best single member with revisit +2.00 / +1.71 / +3.37; evidence gate vs random +2.75 / +3.14 / +4.91;
evidence gate vs softmax +0.12 / +0.07 / +0.36 and vs MC dropout +0.28 / +0.18 / +0.44 (intervals include zero); evidence gate vs revisit-all -0.04 / -0.11 / -0.16 (a tie at a third of the compute).
DINOv2 single pass (92.02 / 89.90 / 68.79) beats the ensemble's single pass and the full revisit pipeline on mcIoU (62.12); the pipeline leads only on mIoU (+2.20).
