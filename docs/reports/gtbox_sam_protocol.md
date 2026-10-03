# GT-box -> SAM -> evidential classifier, scored with mIoU / IoU / mcIoU: protocol fixed before the results

Written 2026-10-02 16:05, while the tracking is running and before any score of this pipeline exists.

Pipeline (the shipped classifier is not retrained or changed): the ground-truth bounding box prompts a SAM-family segmentor; the
four-member evidential ensemble classifies the crop of the predicted mask in one pass; instruments the evidence gate flags are
tracked by SAM2-large (10 frames each way, started from the predicted mask) and relabelled by the evidence-weighted fusion; the
predicted masks with the final labels are painted into per-frame semantic maps and scored against the ground-truth maps with the
benchmark's three IoUs (scripts/gtbox_sam_final_eval.py, a port of MATIS compute_all_iou.py).

Fixed choices:
- Segmentor: SAM2.1-large with the GraSP-fine-tuned mask decoder ("our best SAM model"); zero-shot SAM2.1-large reported alongside
  because the decoder checkpoint was selected on validation loss when validation meant the test cases.
- Headline configuration: the evidence gate with the top 833 instruments by S1 tracked, the budget at which the shipped evidential
  pipeline averages 0.958 over three seeds. The fold1-chosen threshold (1.7e-5) and the 539 budget are reported as well.
- Headline seed: seed 43, the best of the three on the held-out fold1 cases (accuracy 0.9403 against 0.9357 and 0.9369, macro-F1
  0.8974 against 0.8896 and 0.8926). The mean and spread over the three seeds are reported next to it, because the 0.958 reference
  is a three-seed mean. No seed is chosen on a test score.
- Comparison: TAPIS, TAPIS-VST and SlowFast on the GraSP test set (paper, Table 4) and ISINet on the cross-validation set (Table 10, a
  different split). The pipeline is given ground-truth boxes, the compared methods find their own, so this is an oracle-box
  comparison and is labelled as one.

If the scores are poor, the classifier is not tuned against them: that would be a separate piece of work with its own plan.

## Addendum, 2026-10-03: segmentor fine-tuned on ground-truth boxes with a dev-fold checkpoint choice

Fixed before any test-set mask of the new segmentor was produced.

- Motivation: the decoder behind the first run was chosen on validation loss over the test cases, and the segmentor ceiling
  (oracle classes: mIoU 89.29, mcIoU 85.93) bounds every number. The new segmentor is trained with scripts/finetune_sam2_gtbox.py on
  fold2 and its epoch is chosen on fold1 by mean per-instrument mask IoU. The test cases play no part in training or selection.
- Variants run in parallel on fold1: decoder only (dec) and decoder plus the last four Hiera blocks and the FPN neck (enc4).
  Zero-shot fold1 mean mask IoU is 0.8545; the best epoch of each is its own best fold1 IoU.
- Rule: the variant with the higher best fold1 mean IoU is used, with its best-epoch weights (trained on fold2 only, no retrain on
  all cases). Horizontal-flip averaging of the mask logits is used if and only if it raises fold1 mean IoU for that checkpoint.
- Test use: one pass of the chosen configuration over the official test boxes, then the same unchanged pipeline as the first run
  (single-pass logits, three evidential seeds, SAM2-large tracking, gating, top 833 headline, seed 43 headline). The result is
  reported next to the first run whatever it is. Nothing about the classifier, gate or budgets changes.
