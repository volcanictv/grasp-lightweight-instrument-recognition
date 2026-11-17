#!/bin/bash
# Smoke test of the SlowFast instruments config: one epoch of training (and its evaluation) on the test-case files already prepared, only to check config, weights and the region head.
set -uo pipefail
D=$HOME/slowfast_build/data/GraSP
cd ~/baselines/GraSP/TAPIS_gtbox
export PYTHONPATH=$PWD/tapis:${PYTHONPATH:-}
mkdir -p $HOME/slowfast_build/smoke
CUDA_VISIBLE_DEVICES=0 $HOME/tapis_venv/bin/python -B tools/run_net.py --cfg configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml NUM_GPUS 1 \
  TRAIN.CHECKPOINT_FILE_PATH $HOME/slowfast_build/weights/SLOWFAST_4x16_R50.pkl TRAIN.CHECKPOINT_TYPE caffe2 TRAIN.CHECKPOINT_EPOCH_RESET True TRAIN.ENABLE True TEST.ENABLE False \
  SOLVER.MAX_EPOCH 1 SOLVER.WARMUP_EPOCHS 0.0 TRAIN.BATCH_SIZE 4 TEST.BATCH_SIZE 4 \
  ENDOVIS_DATASET.FRAME_DIR $HOME/tapis_gt/frames ENDOVIS_DATASET.FRAME_LIST_DIR $D/frame_lists ENDOVIS_DATASET.TRAIN_LISTS test.csv ENDOVIS_DATASET.TEST_LISTS test.csv \
  ENDOVIS_DATASET.ANNOTATION_DIR $D/annotations/ ENDOVIS_DATASET.TEST_COCO_ANNS $D/annotations/grasp_short-term_test.json \
  ENDOVIS_DATASET.TRAIN_GT_BOX_JSON grasp_short-term_test.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON test_val_preds.json \
  ENDOVIS_DATASET.TEST_GT_BOX_JSON grasp_short-term_test.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON test_val_preds.json \
  FEATURES.TRAIN_FEATURES_PATH $D/features/test_val_region_features.pth FEATURES.TEST_FEATURES_PATH $D/features/test_val_region_features.pth \
  OUTPUT_DIR $HOME/slowfast_build/smoke FEATURES.USE_RPN False > $HOME/slowfast_build/smoke/run.log 2>&1
echo "elapsed, exit $?"
