#!/bin/bash
# Released TAPIS two-fold models (INSTRUMENTS.pyth) on their own region proposals and features, one fold per GPU, on our 1 Hz-style frames (30 fps names linked to the nearest image).
set -uo pipefail
D=$HOME/tapis_folds/data/GraSP
cd ~/baselines/GraSP/TAPIS_gtbox
export PYTHONPATH=$PWD/tapis:${PYTHONPATH:-}
T0=$(date +%s)
for f in 1 2; do
  mkdir -p $HOME/tapis_folds/outputs/swap_fold$f
  CUDA_VISIBLE_DEVICES=$((f-1)) $HOME/tapis_venv/bin/python -B tools/run_net.py --cfg configs/GraSP/TAPIS/TAPIS_INSTRUMENTS.yaml NUM_GPUS 1 \
    TRAIN.CHECKPOINT_FILE_PATH $D/pretrained_models/fold$((3-f))/INSTRUMENTS.pyth TRAIN.CHECKPOINT_EPOCH_RESET True TEST.ENABLE True TRAIN.ENABLE False \
    ENDOVIS_DATASET.FRAME_DIR $D/frames ENDOVIS_DATASET.FRAME_LIST_DIR $D/frame_lists ENDOVIS_DATASET.TRAIN_LISTS fold$f.csv ENDOVIS_DATASET.TEST_LISTS fold$f.csv \
    ENDOVIS_DATASET.ANNOTATION_DIR $D/annotations/ ENDOVIS_DATASET.TEST_COCO_ANNS $D/annotations/grasp_short-term_fold$f.json \
    ENDOVIS_DATASET.TRAIN_GT_BOX_JSON grasp_short-term_fold$f.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON fold${f}_val_preds.json \
    ENDOVIS_DATASET.TEST_GT_BOX_JSON grasp_short-term_fold$f.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON fold${f}_val_preds.json \
    FEATURES.TRAIN_FEATURES_PATH $D/features/fold${f}_val_region_features.pth FEATURES.TEST_FEATURES_PATH $D/features/fold${f}_val_region_features.pth \
    TRAIN.BATCH_SIZE 6 TEST.BATCH_SIZE 6 OUTPUT_DIR $HOME/tapis_folds/outputs/swap_fold$f FEATURES.USE_RPN False > $HOME/tapis_folds/outputs/swap_fold$f/run.log 2>&1 &
done
wait
echo "elapsed $(( $(date +%s)-T0 )) s, exit $?"
