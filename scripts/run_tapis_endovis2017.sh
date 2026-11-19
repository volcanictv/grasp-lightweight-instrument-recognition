#!/bin/bash
# TAPIS retrained on one fold of the EndoVis 2017 4-fold cross-validation (scripts/endovis2017_tapis_prep.py): Mask2Former region stage (ResNet-50, COCO-initialised, as for the EndoVis 2018 run), prediction
# and feature extraction on the fold's train and validation sequences, matching, the TAPIS MViT head (Kinetics-initialised), scored on one GPU with TAPIS's own scorer. Mirrors scripts/run_tapis_endovis.sh.
# The TAPIS checkout needs scripts/patch_tapis_endovis2017.py applied to tapis/datasets/endovis_2017.py.
# Usage (titanxp): nohup bash scripts/run_tapis_endovis2017.sh <fold 0-3> $HOME/endovis2017_tapis > ~/endovis_tapis2017_fold<k>.log 2>&1 &
#   smoke test (tiny schedule, 1 GPU):  NGPU=1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_tapis_endovis2017.sh 0 $HOME/endovis2017_tapis smoke
set -uo pipefail
FOLD=$1; PREP=$2; SMOKE=${3:-}
NGPU=${NGPU:-2}
W=$HOME/endovis_tapis2017_fold$FOLD${SMOKE:+_smoke}
T=$HOME/baselines/GraSP/TAPIS
REPO=$HOME/grasp_yolo26
VPY=$HOME/tapis_venv/bin/python
WEIGHTS=$HOME/endovis_tapis/weights
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $W
export DETECTRON2_DATASETS=$W
source $HOME/tapis_venv/bin/activate

if [ -n "$SMOKE" ]; then ITERS=20; STEPS="(10,15)"; IMS=2; EPOCHS=1; BS=2; else ITERS=6000; STEPS="(2667,2889,5333,5778)"; IMS=$((2 * NGPU)); EPOCHS=20; BS=8; fi
ANN=$PREP/annotations/Fold$FOLD
SIZE=$($VPY -c "import json;d=json.load(open('$ANN/val.json'));print(max(d['images'][0]['height'],d['images'][0]['width']))")
MINS=$($VPY -c "import json;d=json.load(open('$ANN/val.json'));print(min(d['images'][0]['height'],d['images'][0]['width']))")
TRAIN_SIZE=$(( SIZE > 768 ? 768 : SIZE ))  # 1024 crops at 2 images per GPU do not fit 12 GB
say "fold $FOLD, frame size ${SIZE}x${MINS}, training crop $TRAIN_SIZE, gpus $NGPU${SMOKE:+, SMOKE}"

cd $T/region_proposals
CFG=configs/endovis_2017/Base-COCO-EndoVis-InstanceSegmentation.yaml
TRN="endovis-2017_train_fold$FOLD"; VAL="endovis-2017_val_fold$FOLD"
say "mask2former training"
[ -f $W/out_m2f/model_final.pth ] || python -u train_net.py --num-gpus $NGPU --config-file $CFG DATASETS.DATA_PATH $PREP DATASETS.TRAIN "('$TRN',)" DATASETS.TEST "('$VAL',)" \
  MODEL.WEIGHTS $WEIGHTS/model_final_R50.pkl INPUT.IMAGE_SIZE $TRAIN_SIZE INPUT.MIN_SIZE_TEST $MINS INPUT.MAX_SIZE_TEST $SIZE \
  SOLVER.IMS_PER_BATCH $IMS SOLVER.MAX_ITER $ITERS SOLVER.STEPS "$STEPS" TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD $ITERS \
  DATALOADER.NUM_WORKERS 4 OUTPUT_DIR $W/out_m2f > $W/m2f_train.log 2>&1 || { echo "m2f training failed"; exit 1; }

for S in train val; do
  [ -f $W/tapis_data/features/${S}_region_features.pth ] && continue
  say "inference and matching: $S"
  NAME=$TRN; [ $S = val ] && NAME=$VAL
  python -u train_net.py --eval-only --num-gpus $NGPU --config-file $CFG DATASETS.DATA_PATH $PREP DATASETS.TRAIN "('$TRN',)" DATASETS.TEST "('$NAME',)" \
    MODEL.WEIGHTS $W/out_m2f/model_final.pth INPUT.IMAGE_SIZE $SIZE INPUT.MIN_SIZE_TEST $MINS INPUT.MAX_SIZE_TEST $SIZE \
    DATALOADER.NUM_WORKERS 4 OUTPUT_DIR $W/out_m2f_$S > $W/m2f_inf_$S.log 2>&1 || { echo "inference $S failed"; exit 1; }
  SEL="--selection all --segmentation"
  [ $S = val ] && SEL="--selection topk_thresh --selection_info 5,0.1 --segmentation --validation"
  mkdir -p $W/tapis_data/annotations $W/tapis_data/features
  (cd $T/data && $VPY match_annots_n_preds.py --coco_anns_path $ANN/$S.json --preds_path $W/out_m2f_$S/inference/instances_predictions.pth \
    --out_coco_anns_path $W/tapis_data/annotations/${S}_anns.json --out_coco_preds_path $W/tapis_data/annotations/${S}_preds.json \
    --out_features_path $W/tapis_data/features/${S}_region_features.pth $SEL --features_key decoder_out > $W/match_$S.log 2>&1) || { echo "match $S failed"; exit 1; }
  rm -rf $W/out_m2f_$S
done
$VPY $REPO/scripts/tapis_endovis2017_post.py --work $W --dataset 2017 || { echo "post failed"; exit 1; }
$VPY -c "
import json
a=json.load(open('$ANN/val.json'));p=json.load(open('$W/tapis_data/annotations/val_preds.json'))
print('val instances',len(a['annotations']),'predictions kept',len(p['annotations']))"

say "tapis head"
D=$W/tapis_data
cd $T
export PYTHONPATH=$T/tapis:${PYTHONPATH:-}
python -u tools/run_net.py --cfg configs/GraSP/TAPIS/TAPIS_INSTRUMENTS.yaml \
 NUM_GPUS $NGPU TRAIN.ENABLE True TEST.ENABLE False TRAIN.DATASET endovis_2017 TEST.DATASET endovis_2017 \
 TRAIN.CHECKPOINT_FILE_PATH $WEIGHTS/mvit_s_k400.pyth TRAIN.CHECKPOINT_EPOCH_RESET True TRAIN.AUTO_RESUME True \
 DATA.NUM_FRAMES 16 DATA.SAMPLING_RATE 1 DATA.MAX_BBOXES 6 SOLVER.MAX_EPOCH $EPOCHS TRAIN.EVAL_PERIOD 1000 TRAIN.CHECKPOINT_PERIOD $EPOCHS \
 ENDOVIS_DATASET.FRAME_DIR $PREP/images ENDOVIS_DATASET.FRAME_LIST_DIR $D/frame_lists \
 ENDOVIS_DATASET.TRAIN_LISTS train.csv ENDOVIS_DATASET.TEST_LISTS val.csv ENDOVIS_DATASET.ANNOTATION_DIR $D/annotations/ \
 ENDOVIS_DATASET.TEST_COCO_ANNS $D/annotations/val_anns_rle.json \
 ENDOVIS_DATASET.TRAIN_GT_BOX_JSON train_anns.json ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON train_preds.json \
 ENDOVIS_DATASET.TEST_GT_BOX_JSON val_anns.json ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON val_preds.json \
 FEATURES.TRAIN_FEATURES_PATH $D/features/train_region_features.pth FEATURES.TEST_FEATURES_PATH $D/features/val_region_features.pth \
 FEATURES.USE_RPN False TRAIN.BATCH_SIZE $BS TEST.BATCH_SIZE $BS OUTPUT_DIR $W/out_tapis > $W/tapis_train.log 2>&1
CK=$W/out_tapis/checkpoints/checkpoint_epoch_$(printf '%05d' $EPOCHS).pyth
[ -f $CK ] || { echo "tapis failed"; exit 1; }
# validation inside a multi-GPU loop misaligns names and boxes when the last batch is short, so score on one GPU
bash $REPO/scripts/tapis_test_1gpu_2017.sh $W $PREP/images $CK final_1gpu endovis_2017 > $W/tapis_test.log 2>&1
grep '"mode": "val", "task"' $W/tapis_test.log | tail -1 > $W/val_metrics.txt
say "done"
echo "elapsed $SECONDS s"
