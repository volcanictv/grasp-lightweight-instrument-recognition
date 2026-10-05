#!/bin/bash
# Unattended full run of the PI's direction (ground-truth box -> SAM mask -> evidential classifier -> gate -> SAM2-large
# tracking of the predicted mask -> label -> mIoU / IoU / mcIoU), for the three evidential seeds so the result can be set
# against the 3-seed mean of the shipped pipeline. The SAM2 propagation, the slow part, runs once on the union of the
# instruments any seed needs (S1 >= tau, or in its top 833); the tracked masks are saved and the seed-43 and seed-44 members
# classify the same tracks afterwards. Tracking is non-causal (10 frames each way): accuracy first, real time is parked.
# Usage (titanxp, repo root): scripts/run_gtbox_sam_pipeline.sh finetuned
set -uo pipefail
VARIANT=${1:-finetuned}
SEEDS="42 43 44"
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
ENS_TPL=${ENS_TPL:-configs/evidential/ens_E_official_s%s_lam0p01a10.yaml}  # %s = seed; the arm N run sets configs/arms/ens4_N_official_s%s.yaml
ens() { printf "$ENS_TPL" "$1"; }
D=experiments/gtbox_sam/$VARIANT
LOG=experiments/gtbox_sam/pipeline_$VARIANT.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "waiting for the $VARIANT masks"
until [ -f $D/summary.json ]; do sleep 60; done
say "masks ready: $($PY -c "import json;d=json.load(open('$D/summary.json'));print('mean mask IoU', round(d['mean_mask_iou'],4), 'IoU>=0.5', round(d['iou_ge_0.5'],4))")"

for s in $SEEDS; do
  $PY scripts/extract_logits_from_masks.py --masks $D/masks.pkl --ensemble-config $(ens $s) --device cuda:0 --out $D/logits_s$s.npz >> $LOG 2>&1
  say "single-pass logits, seed $s, done"
done

$PY - <<E >> $LOG 2>&1
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from surgical_ai.evaluation.evidential import variance_scores
need = set()
for seed in (42, 43, 44):
    z = np.load(f"$D/logits_s{seed}.npz")
    s1 = variance_scores(alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]))["epistemic"]
    need |= set(np.where((s1 >= 1.7e-5) & z["done"])[0].tolist()) | set(np.argsort(-s1, kind="stable")[:833].tolist())
need = sorted(i for i in need if z["done"][i])
print(f"tracking {len(need)} instruments (union over the three seeds)")
for shard in (0, 1):
    json.dump({"errors": [{"index": int(i)} for i in need[shard::2]]}, open(f"$D/track_shard{shard}.json", "w"))
E
mkdir -p $D/tracked
say "tracking, two GPUs: $(grep -h 'tracking ' $LOG | tail -1)"
for g in 0 1; do
  $PY scripts/evaluate_temporal_track_ensemble.py --split test --window 10 --ensemble-config $(ens 42) \
    --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --init-masks $D/masks.pkl --error-cases-json $D/track_shard$g.json --tmp-dir /tmp/gtbox_track_$g \
    --frame-logits-out $D/tracked/frames_shard$g.npz --masks-out $D/tracked/masks_shard$g.pkl --device cuda:$g \
    --out $D/tracked/tracked_shard$g.json > $D/track$g.log 2>&1 &
done
wait
say "tracking done"

for s in 43 44; do
  $PY scripts/reclassify_tracked_masks.py --masks $D/tracked/masks_shard0.pkl $D/tracked/masks_shard1.pkl --ensemble-config $(ens $s) \
    --device cuda:0 --out $D/tracked_s$s/frames_shard0.npz >> $LOG 2>&1
  say "tracks reclassified with the seed $s members"
done

mkdir -p docs/reports/gtbox_sam
for s in $SEEDS; do
  tdir=$D/tracked_s$s; [ $s = 42 ] && tdir=$D/tracked
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --out docs/reports/gtbox_sam/${VARIANT}_s$s.json >> $LOG 2>&1
  say "scored seed $s"
done
$PY scripts/aggregate_gtbox_seeds.py --variant $VARIANT --seeds 42 43 44 >> $LOG 2>&1
say "aggregated"
echo done > $D/pipeline.done
