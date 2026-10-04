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

## Addendum, 2026-10-03 (second): mask-refinement network after the segmentor

Fixed before the refiner is trained or any test mask is refined.

- Network: scripts/train_mask_refiner.py, a U-Net with an ImageNet ResNet-34 encoder. Input is a square crop (1.25 x the longer box side,
  384 px) of the image, SAM's mask logits and the box; output is SAM's logits plus a learned correction, zero at initialisation,
  forced to background outside the ground-truth box. SAM is the enc4 + flip-average segmentor of the previous addendum.
- Cross-fitting, so the refiner trains on masks of the quality it will meet on unseen frames: training crops are fold2 frames
  segmented by a SAM trained on fold1 only (same recipe, 5 epochs, enc4_f1); dev crops are fold1 frames segmented by the existing
  SAM trained on fold2 only. The refiner's epoch is chosen on fold1 mean mask IoU in crop space. The test cases are used by nothing
  until the acceptance check below passes.
- Acceptance: the refiner is used on the test cases only if its full-resolution fold1 mean mask IoU (apply_mask_refiner.py, paste-back
  inside the box) is at least 0.005 above the SAM fold1 mean mask IoU of 0.9086 and also above the same paste-back applied to SAM's
  own logits. Otherwise it is dropped and the result stays the previous run. One refiner, one test pass; no second try on test.
- If accepted: refined masks replace SAM's masks and the unchanged pipeline reruns as variant gtft_ref (single-pass logits, three
  seeds, tracking, gating, top 833, seed 43 headline), reported next to the other two runs.

## Addendum, 2026-10-03 (third): fine-tuning SAM3 as the segmentor

Fixed before any SAM3 fine-tune result exists. Motivation: zero-shot SAM3 (Sam3TrackerModel, GT-box prompts) scores 0.862 mean mask IoU on a
157-instrument fold1 sample against 0.911 for the fine-tuned SAM2 on the same instruments, so it can only matter if fine-tuned the same way.

- Protocol: as for SAM2 (scripts/finetune_sam3_gtbox.py): train on fold2, choose the epoch on fold1 by mean mask IoU (each epoch on every
  third fold1 frame), then score the best checkpoint on all fold1 frames, plain and flip-averaged. 3 epochs. Test cases are not used.
- Variants: S3a, mask decoder and prompt encoder only; S3b, those plus the FPN neck and the last four ViT layers (the SAM2 analogue of the
  enc4 variant). Reference for both: SAM2 enc4 with flip averaging, fold1 mean mask IoU 0.9086 (plain 0.9059).
- Adoption: the variant with the higher flip-averaged fold1 IoU replaces SAM2 only if that IoU is at least 0.9136 (0.005 above SAM2).
  A SAM2 + SAM3 ensemble (mean of the two models' flip-averaged mask logits, best SAM3 variant) is scored on fold1 as well and adopted
  instead if it is at least 0.9136 and above both members.
- If something is adopted: one pass over the official test boxes, then the unchanged pipeline as variant gtft_sam3 (three seeds,
  tracking, gating, top 833, seed 43 headline), reported next to the other runs. If nothing is adopted, SAM2 stays and the SAM3 result is
  reported as tried.
- Not part of this: clipping masks to the given box (a separate, free post-process measured on fold1 at +0.004) is decided separately.

## Addendum, 2026-10-04: tracker-style crops (arm N) in the GT-box pipeline

Fixed before the arm-N members are trained or scored in this pipeline. Requested by the user; the classifier is otherwise unchanged.

- What changes: the evidential members are trained with the registered arm N (docs/reports/training_experiments_preregistration.md: with
  probability 0.5 the training crop is replaced by a stored EdgeTAM crop of the same instrument one to five frames away; confirmed on fold1
  +0.0063 and fold2 +0.0085 accuracy). All four members (resnet50_320, resnet50_224, baseline, letterbox_crop) for seeds 42, 43 and 44 are
  trained on the official split with the existing fixed 20-epoch schedule and last-epoch weights, validated on fold1 inside the training data
  (scripts/run_training_arms.py --members ..., scripts/make_armN_ensemble_configs.py). The three seed-42 members of the earlier official
  run are reused. Nothing else changes: same frames, weights 0.4/0.2/0.2/0.2, evidential fusion, gate budget top 833, seed 43 headline.
- Evaluation: one pass on the official test cases, in the same pipeline, with the same SAM masks and the same SAM2-large tracked masks as the
  baseline-member run it is compared with (gtft_ens, or gtft if the ensemble run is not used): single-pass logits from the arm-N members, tracks
  reclassified with them, gate by the arm-N ensemble's own S1 with the top-833 budget as the headline (the fold1-chosen threshold would shift with the
  members' confidence, so it is reported but not the headline). Instruments in the arm-N top 833 that the baseline run did not track are tracked in an additional pass before
  scoring, so the arm is not handicapped by a smaller tracked set.
- Reported next to the baseline-member run whatever it is; the two are not mixed per seed or per case.
