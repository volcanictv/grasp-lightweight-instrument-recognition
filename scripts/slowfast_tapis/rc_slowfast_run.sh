#!/bin/bash
# One SlowFast run on the 30 fps windows: train, test on the held-out part (own proposals and given boxes), score with the annotated masks.
# usage: rc_slowfast_run.sh official|fold1|fold2|smoke     knobs by environment: EPOCHS WARMUP BS RATE WORKERS CK_EVERY FRAMES, and W PY CODE for another machine (see run_titanxp.sh)
# Resumable: every stage leaves a .done marker, training resumes from its newest checkpoint, so the same command continues after any stop. Checkpoints are never selected on the held-out part: the last epoch is tested.
set -uo pipefail
RUN=${1:?run name (official|fold1|fold2|smoke)}
W=${W:-$HOME/grasp_work/slowfast}; D=$W/data/run_$RUN; FR=${FRAMES:-/shared/rc/robot-surg/grasp30fps/windows}
CODE=${CODE:-$W/code/TAPIS_gtbox}; PY=${PY:-$HOME/grasp_work/tapis/venv/bin/python}; OUT=$W/runs/$RUN
EPOCHS=${EPOCHS:-30}; WARMUP=${WARMUP:-5.0}; BS=${BS:-6}; RATE=${RATE:-15}; WORKERS=${WORKERS:-6}; CK_EVERY=${CK_EVERY:-5}
say() { echo "[$(date +%F\ %T)] $RUN: $*"; }
for p in "$D/annotations/gt_train.json" "$D/features/train.pth" "$D/frame_lists/train.csv" "$W/weights/SLOWFAST_4x16_R50.pkl" "$CODE/tools/run_net.py" "$PY"; do [ -e "$p" ] || { say "missing $p"; exit 2; }; done
mkdir -p "$OUT"
cp "$W/scripts/SLOWFAST_INSTRUMENTS.yaml" "$CODE/configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml"
cd "$CODE"; export PYTHONPATH=$CODE/tapis:${PYTHONPATH:-}
DATA=(ENDOVIS_DATASET.FRAME_DIR "$FR" ENDOVIS_DATASET.FRAME_LIST_DIR "$D/frame_lists" ENDOVIS_DATASET.TRAIN_LISTS train.csv ENDOVIS_DATASET.TEST_LISTS test.csv
  ENDOVIS_DATASET.ANNOTATION_DIR "$D/annotations/" ENDOVIS_DATASET.TEST_COCO_ANNS "$D/annotations/gt_test.json"
  ENDOVIS_DATASET.TRAIN_GT_BOX_JSON gt_train.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON preds_train.json
  ENDOVIS_DATASET.TEST_GT_BOX_JSON gt_test.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON preds_test.json
  FEATURES.TRAIN_FEATURES_PATH "$D/features/train.pth" FEATURES.TEST_FEATURES_PATH "$D/features/test.pth" FEATURES.USE_RPN False
  DATA.NUM_FRAMES 16 DATA.SAMPLING_RATE "$RATE" DATA_LOADER.NUM_WORKERS "$WORKERS" NUM_GPUS 1)
CFG=configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml

if [ ! -f "$OUT/train.done" ]; then
  say "training: $EPOCHS epochs, batch $BS, sampling rate $RATE, frames from $FR"
  "$PY" -B tools/run_net.py --cfg $CFG "${DATA[@]}" TRAIN.CHECKPOINT_FILE_PATH "$W/weights/SLOWFAST_4x16_R50.pkl" TRAIN.CHECKPOINT_TYPE caffe2 TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.ENABLE True TEST.ENABLE False SOLVER.MAX_EPOCH "$EPOCHS" SOLVER.WARMUP_EPOCHS "$WARMUP" TRAIN.BATCH_SIZE "$BS" TEST.BATCH_SIZE "$BS" \
    TRAIN.EVAL_PERIOD "$EPOCHS" TRAIN.CHECKPOINT_PERIOD "$CK_EVERY" OUTPUT_DIR "$OUT/train" >> "$OUT/train.log" 2>&1 || { say "training failed (see $OUT/train.log)"; exit 3; }
  touch "$OUT/train.done"
fi
CK=$(ls "$OUT"/train/checkpoints/*.pyth 2>/dev/null | sort | tail -n 1); [ -n "$CK" ] || { say "no checkpoint found"; exit 4; }
say "testing checkpoint $CK"

testrun() {  # name, then config overrides
  [ -f "$OUT/$1.done" ] && return 0
  "$PY" -B tools/run_net.py --cfg $CFG "${DATA[@]}" TRAIN.CHECKPOINT_FILE_PATH "$CK" TRAIN.CHECKPOINT_TYPE pytorch TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.ENABLE False TEST.ENABLE True TRAIN.BATCH_SIZE "$BS" TEST.BATCH_SIZE "$BS" "${@:2}" OUTPUT_DIR "$OUT/$1" >> "$OUT/$1.log" 2>&1 || { say "$1 failed (see $OUT/$1.log)"; return 1; }
  touch "$OUT/$1.done"
}
testrun test_pred ENDOVIS_DATASET.USE_PREDS True ENDOVIS_DATASET.INCLUDE_GT False || exit 5
testrun test_given ENDOVIS_DATASET.USE_PREDS False ENDOVIS_DATASET.INCLUDE_GT True || exit 6

score() {  # name, match mode, output dir
  local P; P=$(ls "$OUT/$1"/*preds_instruments.json 2>/dev/null | head -n 1); [ -n "$P" ] || { say "no predictions file in $OUT/$1"; return 1; }
  "$PY" -B "$W/scripts/score_given_masks.py" --gt "$D/annotations/gt_test.json" --preds "$P" --out "$OUT/$3" --match "$2" --tapis-dir "$CODE" 2>&1 | tee "$OUT/$3.log"
}
score test_pred mask score_pred || exit 7
score test_given box score_given || exit 8
say "finished; metrics in $OUT/score_pred/metrics.json and $OUT/score_given/metrics.json"
