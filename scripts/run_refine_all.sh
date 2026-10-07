#!/bin/bash
# Refines every instrument of the official test split that the registered final run did not track (docs/reports/gtbox_sam_protocol.md, 2026-10-07 second addendum): the same SAM2-large tracking over
# +-10 frames from the final pipeline's SAM2 + SAM3 masks, seed 42 members, then the saved masks re-classified with the members of seeds 43 and 44. New directories only.
# Usage (titanxp, repo root, two idle GPUs): scripts/run_refine_all.sh
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/final
ens() { echo configs/arms/ens4_N_official_s$1.yaml; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

$PY - <<E
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
from gtbox_sam_final_eval import load_tracked
z = np.load("$D/logits_s42.npz")
done = z["done"]
have = set(load_tracked(Path("$D/tracked")).keys())
rest = sorted(int(i) for i in np.where(done)[0] if int(i) not in have)
print(f"{len(have)} instruments already tracked, {len(rest)} to track")
for shard in (0, 1):
    json.dump({"errors": [{"index": i} for i in rest[shard::2]]}, open(f"$D/refine_all_shard{shard}.json", "w"))
E
mkdir -p $D/tracked_rest $D/tracked_rest_s43 $D/tracked_rest_s44
say "tracking the rest, two GPUs"
for g in 0 1; do
  $PY scripts/evaluate_temporal_track_ensemble.py --split test --window 10 --ensemble-config $(ens 42) \
    --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --init-masks $D/masks.pkl --error-cases-json $D/refine_all_shard$g.json --tmp-dir /tmp/refine_all_$g \
    --frame-logits-out $D/tracked_rest/frames_shard$g.npz --masks-out $D/tracked_rest/masks_shard$g.pkl --device cuda:$g --save-every 25 \
    --out $D/tracked_rest/tracked_shard$g.json > $D/refine_all_track$g.log 2>&1 &
done
wait
say "tracking done"
for s in 43 44; do
  $PY scripts/reclassify_tracked_masks.py --masks $D/tracked_rest/masks_shard0.pkl $D/tracked_rest/masks_shard1.pkl --ensemble-config $(ens $s) \
    --device cuda:0 --out $D/tracked_rest_s$s/frames_shard0.npz >> $D/refine_all_reclassify.log 2>&1
  say "reclassified with the seed $s members"
done
echo done > $D/refine_all.done
