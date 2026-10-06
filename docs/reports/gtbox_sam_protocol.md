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

## Addendum, 2026-10-04 (second): box clipping and the ensemble in the scoring, with the oracle-box caveat

Written before the ensemble run (gtft_ens) is scored.

- Fold1 evidence (scripts/postproc_study.py, docs/reports/gtbox_sam/postproc_fold1.json, 3,235 instruments, four cases): SAM2 alone 0.9086; SAM2 + SAM3
  ensemble with equal weights 0.9131; plus clipping the mask to the given box 0.9163. Fill holes, largest component, polygon simplification, erosion, a
  threshold bias and logit smoothing do not help (or hurt) and are not used. Tuning on one half of the cases and scoring on the other gave +0.0074 and +0.0064.
- Scoring: scripts/gtbox_sam_final_eval.py now writes every configuration twice, the segmentor's masks as they are and clipped to the given box
  (rows suffixed "+box clip"). Both are always reported. The unclipped rows stay the registered headline.
- Caveat to state wherever the clipped numbers appear: the ground-truth boxes are the tight bounding boxes of the ground-truth masks, so clipping to them uses
  the true mask extent. With a detector's own boxes the gain would be smaller or negative. The clipped rows are therefore oracle-box results only.
- First observation, on the already scored SAM2-only run (seed 43, top 833): unclipped mIoU 86.54 / IoU 85.33 / mcIoU 77.78, clipped 86.77 / 85.56 / 78.01.

## Addendum, 2026-10-04 (third): the final configuration and the final test run

Fixed before any component of it is trained. The final test numbers come from this one configuration; nothing below is chosen on test scores.

- Segmentor: SAM2.1-large (decoder, prompt encoder, last four Hiera blocks and neck trained) and SAM3 (decoder, prompt encoder, neck and last four ViT layers trained),
  each trained on all eight official training cases (the "train" split) with the schedules the fold experiments fixed: SAM2 five epochs, SAM3 three epochs, the
  last epoch's weights, no checkpoint selection on any data (fold1 is inside the training data, so it is only monitored). Masks: equal-weight mean of the two
  models' flip-averaged mask logits, threshold 0. Test frames only at the final pass.
- Classifier: the arm N members of the four-member evidential ensemble for seeds 42, 43 and 44 (addendum of 2026-10-04), same weights, evidential fusion and
  gate (top 833 headline; the fold1-chosen threshold is reported but is not the headline because arm N shifts the members' confidence). SAM2-large tracking of every instrument
  in the union of the three seeds' gated sets, run for the final masks, so no flagged instrument lacks a track.
- Scoring: mIoU, IoU and mcIoU on the official test cases, every configuration twice, unclipped (the headline) and clipped to the given box (oracle-box variant only,
  see the previous addendum). Headline seed: the seed with the highest fold1 accuracy of the arm N three-member ensembles (docs/reports/training_arms/fold1.json), with
  the three-seed mean and spread beside it.
- Ablation ladder reported with it, all on the same test cases: (1) gtft: SAM2 masks, baseline members; (2) gtft_ens: SAM2 + SAM3 ensemble masks (fold-trained on fold2), baseline
  members; (3) gtft_ens_armN: the same masks and tracks with the arm N members; (4) final: all-case segmentors, arm N members. Rungs 1 to 3 already exist or run first; the final is
  the registered headline. If a rung is worse than the one below it, it is still reported.
- Comparison: TAPIS (Swin-L Mask2Former + MViT), TAPIS-VST and SlowFast on the GraSP test set. Oracle-box caveat stated with every table.

## Correction, 2026-10-05: the SAM3 "encoder" variant did not train the encoder

While packaging the weights, the SAM3 delta (the tensors that differ from the base model) contained only the mask decoder and prompt encoder: 106 tensors,
14.6 MB, none from the neck or the ViT layers. The cause is that `Sam3TrackerModel.get_image_embeddings` is decorated with `@torch.no_grad()` in transformers
5.18, so no gradient reached the encoder in scripts/finetune_sam3_gtbox.py, and `--unfreeze-layers 4` changed nothing in the weights. Consequences:

- The third addendum's variants S3a (decoder only) and S3b (decoder plus neck and last four ViT layers) were both decoder-only runs. They differ only in
  the training mode of the encoder (train mode for S3b, which has no effect under no_grad apart from dropout and drop-path if any) and in random state.
  The fold1 results (0.9073 and 0.9076 flip-averaged) are two replicates of the same recipe.
- The final SAM3 (all eight training cases, `--unfreeze-layers 4`) is also decoder-only. All reported numbers are what that model produced; only the
  description "plus the last four ViT layers" is wrong and is withdrawn everywhere it appears.
- The SAM2 fine-tune is not affected: its own forward pass has no such decorator and its delta holds the decoder, prompt encoder, neck and last four Hiera blocks
  (169 tensors).
- Not tested: whether really training the SAM3 encoder would help. The flag now prints a warning.

## Addendum, 2026-10-05 (second): can the GT-box pipeline run in real time

Fixed before any run of it. The PI direction parked the real-time goal on 2026-10-02; the user restarted it on 2026-10-05. Ground-truth boxes stay the input (a detector is out of scope).

- Definition. Online and causal: one keyframe per second, only frames up to the current one, mean end-to-end latency at most 1.0 s per keyframe on one Titan Xp
  (Pascal, no fast half precision; fp32 unless a stage is measured faster in another precision). Budget split: segmentation at most 0.5 s, classification and
  overhead at most 0.15 s, tracking amortised at most 0.35 s per keyframe. Latency is measured per stage on 60 test frames (scripts/benchmark_gtbox_stage_latency.py)
  and, for tracking, in a streaming replay. What a faster GPU would allow is not measured and is not claimed.
- Rung S (segmenter speed). SAM2.1 small and tiny fine-tuned with the large model's dev recipe (fold2 training cases, last four Hiera blocks and neck unfrozen,
  dev fold1, checkpoint chosen on fold1 mean mask IoU without flip). Reference: the large model's fold1 value for the same recipe (0.9059 without flip, 0.9086 with).
  Rule: take the fastest variant whose fold1 no-flip mean IoU is within 0.005 of the large model's; if none qualifies, keep the large model. Flip averaging is used
  only if the chosen segmenter with flip stays within the 0.5 s segmentation budget.
- Rung A (no tracking). The chosen segmenter trained on all eight training cases with the same fixed schedule as the final run (last epoch, no checkpoint choice;
  the all-case SAM2-large weights already exist), test masks, the arm N four-member evidential ensemble of seeds 42, 43 and 44 on those masks, no gate, no tracking.
- Rung B (causal tracking). Rung A plus EdgeTAM tracking of the gated instruments over the past 20 frames only, initialised from the segmenter's mask, the
  tracker, look-back and combine unchanged from the causal preregistration (docs/reports/causal_tracker_preregistration.md). Gate: the 539 most uncertain instruments
  (19%, the budget of the fold1-chosen threshold) and the same threshold 2.85e-4 of that preregistration, both reported; nothing is tuned on test. Streaming
  replay reports the amortised tracking time per keyframe.
- Rung C (association fusion, no tracker). Each instrument's belief is fused with the beliefs of earlier instruments matched by box overlap in the previous k
  keyframes, no ground-truth identity used. Settings (overlap threshold from {0.3, 0.5}, k from {3, 5, 10}) chosen on fold1 only with the fold2-trained members,
  by the rule of the causal preregistration (highest fold1 accuracy, then the cheapest within 0.001). If fold1 shows no gain, rung C is reported as negative and not run on test.
- Scoring. mIoU, IoU and mcIoU as in the final run (unclipped headline), three seeds, beside the non-causal final (87.37 / 86.22 / 78.33) and TAPIS. Every rung is
  reported whatever the result. Segmenters are trained once, so the seed spread covers the classifier only. The oracle-box caveat applies to every row.

## Addendum, 2026-10-06: YOLO26s as the causal tracker (rung B2), added after rung B

Rung B used EdgeTAM only; the first addendum named no other tracker and YOLO was left out without a registered reason. The user asked for it, so it is run now,
after the EdgeTAM scores were seen (disclosed: its settings are not chosen with that knowledge, they are the locked ones of the causal preregistration).

- Rung B2: the rung A tiny + flip masks (all-case SAM2.1 tiny) and the arm N four-member ensemble of seeds 42, 43 and 44, as in rung B, with the YOLO26s-seg
  tracker in place of EdgeTAM: official-split weights (experiments/yolo26_seg_official_20260930-131921/weights/last.pt), 15 past frames only, matching min-IoU 0.1,
  coast 3, centre-frame fallback, conf 0.1, imgsz 640, the track started from the segmenter's mask. Gate: S1 >= 5.75e-4 (the YOLO tracker's fold1-chosen
  threshold of the causal preregistration) and, separately, the 539 most uncertain. Nothing is tuned.
- Reported beside rung B, whatever the result. Latency: YOLO per frame from docs/reports/causal_realtime/latency/yolo.json (10.5 ms median), matching on the CPU
  measured in the run.

## Addendum, 2026-10-06 (second): causal SAM2-large propagation

Fixed before the run. The final pipeline tracks with SAM2-large over 10 frames on each side of the keyframe, which uses future frames. This run makes the propagation causal and measures what it costs.

- Tracker: SAM2-large (the same checkpoint and configuration as the final run), propagated backward only from the keyframe over the 20 past frames (21 frames with the keyframe; `--causal --window 20` of scripts/evaluate_temporal_track_ensemble.py),
  initialised from the segmenter's mask. The look-back of 20 matches rung B (EdgeTAM) so the two causal trackers are compared like for like. Classifier, evidential fusion and frame combine unchanged. No setting is tuned.
- Two runs. (a) The final pipeline's masks (all-case SAM2 + SAM3 ensemble, arm N members, seeds 42, 43, 44): only the tracker changes from non-causal to causal, scored beside the non-causal final (87.37 / 86.22 / 78.33). (b) The real-time pipeline's masks (SAM2-tiny with flip, rung A) with SAM2-large in place of EdgeTAM, scored beside rung B.
- Gates, both reported: the 539 most uncertain instruments, S1 >= 2.85e-4 (the causal preregistration's threshold), and for run (a) also the 833 most uncertain (the final run's headline budget). Tracks are made once for the union over the three seeds of the instruments any of these gates selects.
- Scoring as in the final run (mIoU, IoU, mcIoU, unclipped headline, three seeds). Reported whatever the result. SAM2-large's cost is unchanged (about 21 frames per tracked instrument), so this run answers accuracy, not speed.
