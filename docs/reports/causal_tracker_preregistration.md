# Causal tracker A/B: settings locked before the official-test run

Written 2026-10-01, before any causal tracking was run on the official test set.

Pipeline: three-member evidential ensemble (resnet50_320, baseline, letterbox_crop; weights 0.40 / 0.30 / 0.30;
resnet50_224 dropped), single-pass epistemic score S1 as the gate, tracking of the gated instances over the
past k frames only (a live 1 Hz system has no future frames), frame-weighted evidential combine. Ungated
instances keep the single-frame prediction. The tracker is initialised from the annotated instance mask at the
current frame; detection is out of scope.

All choices below come from fold1 only (members trained on fold2; official test never read for selection).

| | A: EdgeTAM | B: YOLO26s-seg |
|---|---|---|
| Look-back k (past frames) | 20 | 15 |
| Gate threshold on S1 | 2.85e-4 | 5.75e-4 |
| Matching | mask propagation from the annotated mask | IoU 0.1, coast 3 frames, centre-frame fallback, conf 0.1, imgsz 640 |
| Weights | edgetam.pt (no training) | official-split run, fixed 60 epochs, final weights (last.pt), no per-epoch validation |
| Fold1 result (acc / macro-F1) | 0.9493 / 0.9083 | 0.9440 / 0.8921 |

Selection rule for k and threshold: maximise fold1 accuracy, take the highest threshold within 0.0010 of the
best, then the shortest k within 0.0010 of the best k (scripts/calibrate_tracker_causal.py, grid = S1
percentiles 98, 96, 94, 92, 90, 88, 85, 82).

YOLO26m-seg was trained as a capped improvement attempt and gave no gain on fold1 (0.9437 / 0.8948); it is not
carried forward.

Scored once on the official test (2861 instances, 5 cases): accuracy, macro-F1, per-class F1, fixed and broken
counts against the three-member base prediction. Both trackers are reported whatever the result. No setting
above is changed after the official numbers are seen.

Disclosure: earlier in this work an official-test number for a non-causal YOLO tracker (first matching config,
0.9217 accuracy at its fold1-calibrated threshold) was read. It informed the decision to run the fold1 matching
sweep, but every setting in the table was chosen on fold1.
