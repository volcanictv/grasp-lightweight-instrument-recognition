#!/bin/bash
# Evaluation-path smoke test: the one-epoch smoke checkpoint, given-box mode (USE_PREDS False) on the test files.
set -uo pipefail
D=$HOME/slowfast_build/data/GraSP
cd ~/baselines/GraSP/TAPIS_gtbox
export PYTHONPATH=$PWD/tapis:${PYTHONPATH:-}
CK=$(ls $HOME/slowfast_build/smoke/checkpoints/*.pyth | tail -1); echo "checkpoint $CK"
mkdir -p $HOME/slowfast_build/smoke_test
CUDA_VISIBLE_DEVICES=0 $HOME/tapis_venv/bin/python -B tools/run_net.py --cfg configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml NUM_GPUS 1 \
  TRAIN.CHECKPOINT_FILE_PATH $CK TRAIN.CHECKPOINT_TYPE pytorch TRAIN.CHECKPOINT_EPOCH_RESET True TRAIN.ENABLE False TEST.ENABLE True TRAIN.BATCH_SIZE 4 TEST.BATCH_SIZE 4 \
  ENDOVIS_DATASET.USE_PREDS False ENDOVIS_DATASET.INCLUDE_GT True \
  ENDOVIS_DATASET.FRAME_DIR $HOME/tapis_gt/frames ENDOVIS_DATASET.FRAME_LIST_DIR $D/frame_lists ENDOVIS_DATASET.TRAIN_LISTS test.csv ENDOVIS_DATASET.TEST_LISTS test.csv \
  ENDOVIS_DATASET.ANNOTATION_DIR $D/annotations/ ENDOVIS_DATASET.TEST_COCO_ANNS $D/annotations/grasp_short-term_test.json \
  ENDOVIS_DATASET.TRAIN_GT_BOX_JSON grasp_short-term_test.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON test_val_preds.json \
  ENDOVIS_DATASET.TEST_GT_BOX_JSON grasp_short-term_test.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON test_val_preds.json \
  FEATURES.TRAIN_FEATURES_PATH $D/features/test_val_region_features.pth FEATURES.TEST_FEATURES_PATH $D/features/test_val_region_features.pth \
  OUTPUT_DIR $HOME/slowfast_build/smoke_test FEATURES.USE_RPN False > $HOME/slowfast_build/smoke_test/run.log 2>&1
echo "elapsed, exit $?"
