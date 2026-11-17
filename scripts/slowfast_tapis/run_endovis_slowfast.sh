#!/bin/bash
# SlowFast baseline on EndoVis 2018 (ISINet split) with the inputs of the retrained TAPIS run: its Mask2Former proposals and region features, 16 consecutive frames (sampling rate 1), the same frame cache.
# Same recipe as the GraSP SlowFast runs (Kinetics-400 init, SGD 0.0125 cosine, 30 epochs, last checkpoint tested, no selection on the held-out part). Tested on one GPU (a second GPU misaligns names and boxes in the loop).
# Scored with the annotated masks: each annotated instrument takes the class distribution of the own proposal that overlaps its mask most, painted with the annotated mask, through TAPIS's evaluator.
# usage (titanxp): CUDA_VISIBLE_DEVICES=0 bash run_endovis_slowfast.sh      Resumable through .done markers and the newest checkpoint.
set -uo pipefail
W=${W:-$HOME/sf30}; D=${D:-$HOME/sf_endovis/tapis_data}; FR=${FRAMES:-$HOME/endovis_isinet/frames}
CODE=${CODE:-$HOME/baselines/GraSP/TAPIS_gtbox}; PY=${PY:-$HOME/tapis_venv/bin/python}; OUT=${OUT:-$HOME/sf_endovis/run}
EPOCHS=${EPOCHS:-30}; WARMUP=${WARMUP:-5.0}; BS=${BS:-6}; WORKERS=${WORKERS:-4}; CK_EVERY=${CK_EVERY:-10}
say() { echo "[$(date +%F\ %T)] endovis: $*"; }
for p in "$D/annotations/train_anns.json" "$D/features/train_region_features.pth" "$D/frame_lists/train.csv" "$W/weights/SLOWFAST_4x16_R50.pkl" "$CODE/tools/run_net.py" "$FR/seq_2_frame000.jpg"; do [ -e "$p" ] || { say "missing $p"; exit 2; }; done
mkdir -p "$OUT"
cp "$W/scripts/SLOWFAST_INSTRUMENTS.yaml" "$CODE/configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml"
cd "$CODE"; export PYTHONPATH=$CODE/tapis:${PYTHONPATH:-}
DATA=(TRAIN.DATASET endovis_2018 TEST.DATASET endovis_2018 DATA.NUM_FRAMES 16 DATA.SAMPLING_RATE 1 DATA.MAX_BBOXES 6
  ENDOVIS_DATASET.FRAME_DIR "$FR" ENDOVIS_DATASET.FRAME_LIST_DIR "$D/frame_lists" ENDOVIS_DATASET.TRAIN_LISTS train.csv ENDOVIS_DATASET.TEST_LISTS val.csv
  ENDOVIS_DATASET.ANNOTATION_DIR "$D/annotations/" ENDOVIS_DATASET.TEST_COCO_ANNS "$D/annotations/val_anns_rle.json"
  ENDOVIS_DATASET.TRAIN_GT_BOX_JSON train_anns.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON train_preds.json
  ENDOVIS_DATASET.TEST_GT_BOX_JSON val_anns.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON val_preds.json
  FEATURES.TRAIN_FEATURES_PATH "$D/features/train_region_features.pth" FEATURES.TEST_FEATURES_PATH "$D/features/val_region_features.pth" FEATURES.USE_RPN False
  DATA_LOADER.NUM_WORKERS "$WORKERS" NUM_GPUS 1)
CFG=configs/GraSP/SLOWFAST/SLOWFAST_INSTRUMENTS.yaml

if [ ! -f "$OUT/train.done" ]; then
  say "training: $EPOCHS epochs, batch $BS"
  "$PY" -B tools/run_net.py --cfg $CFG "${DATA[@]}" TRAIN.CHECKPOINT_FILE_PATH "$W/weights/SLOWFAST_4x16_R50.pkl" TRAIN.CHECKPOINT_TYPE caffe2 TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.ENABLE True TEST.ENABLE False SOLVER.MAX_EPOCH "$EPOCHS" SOLVER.WARMUP_EPOCHS "$WARMUP" TRAIN.BATCH_SIZE "$BS" TEST.BATCH_SIZE "$BS" \
    TRAIN.EVAL_PERIOD "$EPOCHS" TRAIN.CHECKPOINT_PERIOD "$CK_EVERY" OUTPUT_DIR "$OUT/train" >> "$OUT/train.log" 2>&1 || { say "training failed (see $OUT/train.log)"; exit 3; }
  touch "$OUT/train.done"
fi
CK=$(ls "$OUT"/train/checkpoints/*.pyth 2>/dev/null | sort | tail -n 1); [ -n "$CK" ] || { say "no checkpoint found"; exit 4; }
say "testing checkpoint $CK"
if [ ! -f "$OUT/test.done" ]; then
  "$PY" -B tools/run_net.py --cfg $CFG "${DATA[@]}" TRAIN.CHECKPOINT_FILE_PATH "$CK" TRAIN.CHECKPOINT_TYPE pytorch TRAIN.CHECKPOINT_EPOCH_RESET True TRAIN.ENABLE False TEST.ENABLE True \
    TRAIN.BATCH_SIZE "$BS" TEST.BATCH_SIZE "$BS" ENDOVIS_DATASET.USE_PREDS True ENDOVIS_DATASET.INCLUDE_GT False OUTPUT_DIR "$OUT/test" >> "$OUT/test.log" 2>&1 || { say "test failed (see $OUT/test.log)"; exit 5; }
  touch "$OUT/test.done"
fi
P=$(ls "$OUT"/test/*preds_instruments.json 2>/dev/null | head -n 1); [ -n "$P" ] || { say "no predictions file"; exit 6; }
"$PY" -B "$W/scripts/score_given_masks.py" --gt "$D/annotations/val_anns_rle.json" --preds "$P" --out "$OUT/score_pred" --match mask --tapis-dir "$CODE" 2>&1 | tee "$OUT/score_pred.log"
say "finished; metrics in $OUT/score_pred/metrics.json"
