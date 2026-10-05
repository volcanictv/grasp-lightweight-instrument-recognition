#!/bin/bash
# Rung B of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-05 second): rung A plus causal EdgeTAM tracking (past 20 frames only, initialised from the
# segmenter's masks) of the gated instruments; settings of docs/reports/causal_tracker_preregistration.md unchanged, nothing tuned. Gate: S1 >= 2.85e-4 and, separately,
# the 539 most uncertain. Tracks are made once for the union of the three seeds' gated sets (seed 42 members), the other seeds' members classify the same tracks.
# Usage (titanxp, repo root, after run_rt_rung_a.sh): scripts/run_rt_rung_b.sh VARIANT [GPU]
set -uo pipefail
VARIANT=$1; GPU=${2:-0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/$VARIANT
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $D/tracked_b

$PY - <<E
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from surgical_ai.evaluation.evidential import variance_scores
need = set()
for seed in (42, 43, 44):
    z = np.load(f"$D/logits_s{seed}.npz")
    s1 = variance_scores(alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]))["epistemic"]
    need |= set(np.where((s1 >= 2.85e-4) & z["done"])[0].tolist()) | set(np.argsort(-s1, kind="stable")[:539].tolist())
need = sorted(i for i in need if z["done"][i])
json.dump({"errors": [{"index": int(i)} for i in need]}, open("$D/tracked_b/idx.json", "w"))
print(f"tracking {len(need)} instruments (union over the three seeds)")
E
say "causal EdgeTAM tracking"
~/edgetam_venv/bin/python scripts/evaluate_temporal_track_ensemble.py --split test --causal --window 20 --ensemble-config configs/arms/ens4_N_official_s42.yaml \
  --sam2-checkpoint ~/EdgeTAM/checkpoints/edgetam.pt --sam2-config configs/edgetam.yaml --init-masks $D/masks.pkl --error-cases-json $D/tracked_b/idx.json \
  --tmp-dir /tmp/rt_track_$VARIANT --frame-logits-out $D/tracked_b/frames_shard0.npz --masks-out $D/tracked_b/masks_shard0.pkl --device cuda:$GPU \
  --out $D/tracked_b/tracked_shard0.json > $D/track_b.log 2>&1
say "tracking done"
for s in 43 44; do
  mkdir -p $D/tracked_b_s$s
  $PY scripts/reclassify_tracked_masks.py --masks $D/tracked_b/masks_shard0.pkl --ensemble-config configs/arms/ens4_N_official_s$s.yaml --device cuda:$GPU \
    --out $D/tracked_b_s$s/frames_shard0.npz >> $D/track_b.log 2>&1
done
for s in 42 43 44; do
  tdir=$D/tracked_b_s$s; [ $s = 42 ] && tdir=$D/tracked_b
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --tau 2.85e-4 --budgets 539 \
    --out docs/reports/gtbox_sam/${VARIANT}_causal_s$s.json >> $D/score_b.log 2>&1
done
$PY scripts/aggregate_gtbox_seeds.py --variant ${VARIANT}_causal --seeds 42 43 44 | tee $D/aggregate_b.txt
echo done > $D/rung_b.done
