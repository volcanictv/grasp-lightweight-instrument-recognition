#!/bin/bash
# EndoVis 2018 (ISINet split) with the ANNOTATED masks instead of predicted ones: one classifier seed (42, GraSP recipe members of experiments/endovis_isinet_grasprecipe),
# single-pass logits, SAM2 tracking of the gated 32.9% from the annotation masks, track logits and scoring with the same scripts as scripts/run_endovis_gtbox.sh.
# Waits for the two-fold GraSP run (scripts/run_gtfold_gtmask.sh) to release the GPUs. Usage (titanxp, repo root): nohup bash scripts/run_endovis_gtmask.sh > ~/gtev.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENDOVIS_GT=$HOME/endovis_isinet ENDOVIS_ISINET=1
PY=$HOME/miniconda3/envs/surgical/bin/python
R=experiments/endovis_isinet_gtmask
say() { echo "[$(date +%H:%M:%S)] $*"; }
T0=$(date +%s)
fail() { say "FAILED: $*"; echo "elapsed $(( $(date +%s)-T0 )) s, exit 1"; exit 1; }
until grep -q "^elapsed" ~/gtfold.log 2>/dev/null; do sleep 120; done
say "gpus free; annotation masks"
mkdir -p $R/masks $R/tracks
$PY scripts/endovis_gt_masks_pkl.py --out $R/masks/masks_ens.pkl || fail "masks"
$PY scripts/endovis_gtbox_logits.py --members $R/members/s42 --masks $R/masks/masks_ens.pkl --device cuda:0 --out $R/single_ens_s42.npz > $R/logits.log 2>&1 || fail "single-pass logits"
say "single-pass logits done"
for g in 0 1; do
  $PY -u scripts/endovis_gtbox_track.py --masks $R/masks/masks_ens.pkl --logits $R/single_ens_s42.npz --shard-id $g --num-shards 2 --device cuda:$g --out-dir $R/tracks > $R/track$g.log 2>&1 &
done
wait
for g in 0 1; do grep -q "^done" $R/track$g.log || fail "tracking shard $g"; done
say "tracking done"
$PY scripts/endovis_gtbox_logits.py --members $R/members/s42 --tracks $R/tracks/tracks_shard0.pkl $R/tracks/tracks_shard1.pkl --device cuda:0 --out $R/tracked_s42.npz > $R/tracklogits.log 2>&1 || fail "track logits"
$PY scripts/endovis_gtbox_eval.py --run-dir $R --seeds 42 --variants ens --out $R/results.json > $R/eval.log 2>&1 || fail "eval"
tail -n 12 $R/eval.log
say "done"
echo "elapsed $(( $(date +%s)-T0 )) s, exit 0"
