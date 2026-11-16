SlowFast baseline for GraSP instrument recognition, built on the TAPIS code base (which already contains the SlowFast model and a region head), following the TAPIR recipe
(Valderrama et al. 2022, Section 4.1): SlowFast video features of 16 consecutive frames centred on the keyframe (alpha 8, beta 1/8), the detector's region features fused in the region head,
SGD 0.0125 cosine, 30 epochs with 5 warm-up. Initialisation: Kinetics-400 SlowFast 4x16 R50 (https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl); the
CaDIS-pretrained weights of the original baseline are not available.

Config: configs/slowfast/SLOWFAST_INSTRUMENTS.yaml (copy to TAPIS/configs/GraSP/SLOWFAST/).
Two edits to the TAPIS checkout (backups kept as *.orig): patch_region_head.py (the SlowFast region head took AVA-style (K,5) boxes; it now consumes the loader's padded boxes plus boxes_mask, uses the
detector features without dropping a column, and its classifier takes the 2048 concatenated features) and patch_eval_loop.py (the evaluation loop moved only the first pathway to the GPU).
Smoke tests (one epoch on the test-case files, then given-box evaluation with ENDOVIS_DATASET.USE_PREDS False and INCLUDE_GT True): smoke_train.sh, smoke_test.sh. Training and evaluation both run; 0.26 s per batch of 4 on a Titan Xp.
