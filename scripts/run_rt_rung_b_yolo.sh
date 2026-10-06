#!/bin/bash
# Rung B2 of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-06): rung A masks plus causal YOLO26s-seg tracking (15 past frames, min-IoU 0.1, coast 3,
# centre fallback, official-split last.pt) of the gated instruments, the settings of docs/reports/causal_tracker_preregistration.md. Gate: S1 >= 5.75e-4 and, separately, the 539
# most uncertain. Detections are cached once for the union of the three seeds' gated sets; each seed's members classify the tracks from the same cache.
# Usage (titanxp, repo root, after run_rt_rung_a.sh): scripts/run_rt_rung_b_yolo.sh VARIANT [GPU]
set -uo pipefail
VARIANT=$1; GPU=${2:-0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
YPY=$HOME/yolo26_venv/bin/python
W=experiments/yolo26_seg_official_20260930-131921/weights/last.pt
D=experiments/gtbox_sam/$VARIANT
Y=$D/yolo_b
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $Y docs/reports/gtbox_sam

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
    need |= set(np.where((s1 >= 5.75e-4) & z["done"])[0].tolist()) | set(np.argsort(-s1, kind="stable")[:539].tolist())
need = sorted(i for i in need if z["done"][i])
json.dump({"errors": [{"index": int(i)} for i in need]}, open("$Y/idx.json", "w"))
print(f"tracking {len(need)} instruments (union over the three seeds)")
E
say "detections"
$YPY scripts/cache_yolo_detections.py --yolo-weights $W --split test --causal --window 15 --error-cases-json $Y/idx.json --device cuda:$GPU --out $Y/dets.pkl > $Y/cache.log 2>&1
say "tracking and classifying"
for s in 42 43 44; do
  mkdir -p $Y/s$s
  $YPY scripts/evaluate_temporal_track_yolo.py --yolo-weights $W --split test --causal --window 15 --min-iou 0.1 --coast 3 --center-fallback \
    --detections-cache $Y/dets.pkl --ensemble-config configs/arms/ens4_N_official_s$s.yaml --init-masks $D/masks.pkl --error-cases-json $Y/idx.json \
    --device cuda:$GPU --frame-logits-out $Y/s$s/frames_shard0.npz --out $Y/s$s/tracked_shard0.json > $Y/track_s$s.log 2>&1
  say "seed $s tracked"
done
for s in 42 43 44; do
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $Y/s$s --tau 5.75e-4 --budgets 539 \
    --out docs/reports/gtbox_sam/${VARIANT}_yolo_s$s.json >> $Y/score.log 2>&1
done
$PY scripts/aggregate_gtbox_seeds.py --variant ${VARIANT}_yolo --seeds 42 43 44 | tee $Y/aggregate.txt
echo done > $D/rung_b_yolo.done
