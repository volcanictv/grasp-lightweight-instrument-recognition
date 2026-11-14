#!/bin/bash
# Two-fold validation of the ensemble + revisit with ground-truth masks (GraSP folds: train on one fold, evaluate on the other), one seed (42), last-epoch members (the held-out fold never chooses a checkpoint):
# train 4 members per fold (arm NL), ground-truth masks and single-pass logits of the held-out fold, SAM2 tracking of the instruments with S1 >= tau, scoring.
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
PY=$HOME/miniconda3/envs/surgical/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }
T0=$(date +%s)
MEM="resnet50_320 resnet50_224 baseline letterbox_crop"
say "training fold1 members (held-out fold1, train fold2)"
$PY scripts/run_training_arms.py --fold fold1 --arms NL --seeds 42 --members $MEM --neighbour-dir experiments/temporal_neighbors/fold2 --data-root "$GRASP_DATA_ROOT"
say "training fold2 members (held-out fold2, train fold1)"
$PY scripts/run_training_arms.py --fold fold2 --arms NL --seeds 42 --members $MEM --neighbour-dir experiments/temporal_neighbors/fold1 --data-root "$GRASP_DATA_ROOT"
for k in 1 2; do $PY scripts/make_armN_ensemble_configs.py --arm NL --fold fold$k --seeds 42 || { say "ensemble config for fold$k failed"; echo "elapsed $(( $(date +%s)-T0 )) s, exit 1"; exit 1; }; done
say "ground-truth masks and single-pass logits (gpu k-1 per fold)"
for k in 1 2; do
  D=experiments/gtbox_sam/gtfold$k; mkdir -p $D
  ( $PY scripts/gt_masks_to_pkl.py --split fold$k --out $D/masks.pkl > $D/masks.log 2>&1 &&
    nice -n 5 $PY -u scripts/extract_logits_from_masks.py --split fold$k --masks $D/masks.pkl --ensemble-config configs/arms/ens4_NL_fold${k}_s42.yaml --device cuda:$((k-1)) --out $D/logits_s42.npz > $D/extract.log 2>&1 ) &
done
wait
for k in 1 2; do
  D=experiments/gtbox_sam/gtfold$k
  [ -f $D/logits_s42.npz ] || { say "no logits for fold$k"; echo "elapsed $(( $(date +%s)-T0 )) s, exit 1"; exit 1; }
  $PY - <<E
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from surgical_ai.evaluation.evidential import variance_scores
z = np.load("$D/logits_s42.npz")
s1 = variance_scores(alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]))["epistemic"]
need = sorted(int(i) for i in np.where((s1 >= 1.7e-5) & z["done"])[0])
json.dump({"errors": [{"index": i} for i in need]}, open("$D/track_idx.json", "w"))
print("fold$k: tracking", len(need), "of", int(z["done"].sum()), "instruments; single-pass accuracy", round(float((alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]).argmax(1) == z["y"])[z["done"]].mean()), 4))
E
done
say "tracking, fold1 on gpu 0 and fold2 on gpu 1"
for k in 1 2; do
  D=experiments/gtbox_sam/gtfold$k; mkdir -p $D/tracked
  $PY scripts/evaluate_temporal_track_ensemble.py --split fold$k --window 10 --ensemble-config configs/arms/ens4_NL_fold${k}_s42.yaml \
    --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --init-masks $D/masks.pkl --error-cases-json $D/track_idx.json --tmp-dir /tmp/gtfold_track_$k \
    --frame-logits-out $D/tracked/frames_shard0.npz --masks-out $D/tracked/masks_shard0.pkl --save-every 25 --device cuda:$((k-1)) \
    --out $D/tracked/tracked_shard0.json > $D/track.log 2>&1 &
done
wait
say "tracking done; scoring"
for k in 1 2; do
  D=experiments/gtbox_sam/gtfold$k
  $PY scripts/gtbox_sam_final_eval.py --split fold$k --masks $D/masks.pkl --logits $D/logits_s42.npz --tracked-dir $D/tracked --tau 1.7e-5 --budgets 941 --out $D/refined_s42.json > $D/score.log 2>&1
  say "fold$k scored"
done
say "done"
echo "elapsed $(( $(date +%s)-T0 )) s, exit $?"
