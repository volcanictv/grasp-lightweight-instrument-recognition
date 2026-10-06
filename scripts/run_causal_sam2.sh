#!/bin/bash
# Causal SAM2-large propagation (docs/reports/gtbox_sam_protocol.md, 2026-10-06 second): past 20 frames only, initialised from the segmenter's masks of VARIANT
# (final = the final pipeline's masks, rt_tiny = the real-time pipeline's masks), the arm N members of seeds 42, 43, 44. Gates: S1 >= 2.85e-4, the 539 and the 833 most uncertain.
# Usage (titanxp, repo root, after the variant's logits exist): scripts/run_causal_sam2.sh VARIANT [GPU]
set -uo pipefail
VARIANT=$1; GPU=${2:-0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/$VARIANT
T=$D/tracked_causal_sam2
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $T docs/reports/gtbox_sam

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
    need |= set(np.where((s1 >= 2.85e-4) & z["done"])[0].tolist()) | set(np.argsort(-s1, kind="stable")[:833].tolist())
need = sorted(i for i in need if z["done"][i])
json.dump({"errors": [{"index": int(i)} for i in need]}, open("$T/idx.json", "w"))
print(f"tracking {len(need)} instruments (union over the three seeds)")
E
say "causal SAM2-large tracking"
$PY scripts/evaluate_temporal_track_ensemble.py --split test --causal --window 20 --ensemble-config configs/arms/ens4_N_official_s42.yaml \
  --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --init-masks $D/masks.pkl \
  --error-cases-json $T/idx.json --tmp-dir /tmp/causal_sam2_$VARIANT --frame-logits-out $T/frames_shard0.npz --masks-out $T/masks_shard0.pkl \
  --device cuda:$GPU --out $T/tracked_shard0.json > $T/track.log 2>&1
say "tracking done"
for s in 43 44; do
  mkdir -p ${T}_s$s
  $PY scripts/reclassify_tracked_masks.py --masks $T/masks_shard0.pkl --ensemble-config configs/arms/ens4_N_official_s$s.yaml --device cuda:$GPU \
    --out ${T}_s$s/frames_shard0.npz >> $T/track.log 2>&1
done
for s in 42 43 44; do
  tdir=${T}_s$s; [ $s = 42 ] && tdir=$T
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --tau 2.85e-4 --budgets 539 833 \
    --out docs/reports/gtbox_sam/${VARIANT}_causalsam2_s$s.json >> $T/score.log 2>&1
  say "seed $s scored"
done
$PY scripts/aggregate_gtbox_seeds.py --variant ${VARIANT}_causalsam2 --seeds 42 43 44 | tee $T/aggregate.txt
echo done > $D/causal_sam2.done
