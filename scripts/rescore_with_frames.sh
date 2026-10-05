#!/bin/bash
# Re-scores the four rungs of the ablation ladder (gtft, gtft_ens, gtft_ens_armN, final) for the three seeds, now also saving every
# configuration's per-frame class IoUs (scripts/bootstrap_cis.py reads them), and checks each re-score against the committed result
# JSON: a deterministic re-score must reproduce the published numbers exactly, which makes this a reproducibility check as well.
# Usage (titanxp, repo root): nohup scripts/rescore_with_frames.sh > experiments/rescore/driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
G=experiments/gtbox_sam
OUT=experiments/rescore
mkdir -p $OUT
say() { echo "[$(date +%H:%M:%S)] $*"; }

# the seed-42 tracked masks live in tracked/, the others in tracked_s43 and tracked_s44
rescore_one() {
  local variant=$1 masks=$2 logits_tpl=$3 tracked_root=$4 s t
  for s in 42 43 44; do
    t=$tracked_root/tracked_s$s; [ $s = 42 ] && t=$tracked_root/tracked
    $PY scripts/gtbox_sam_final_eval.py --masks $masks --logits $(printf "$logits_tpl" $s) --tracked-dir $t \
      --out $OUT/${variant}_s$s.json --save-frames $OUT/${variant}_s$s.frames.pkl > $OUT/${variant}_s$s.log 2>&1
    say "$variant seed $s scored"
  done
}
rescore_armN() {
  local s
  for s in 42 43 44; do
    $PY scripts/gtbox_sam_final_eval.py --masks $G/gtft_ens/masks.pkl --logits $G/gtft_ens/logits_armN_s$s.npz --tracked-dir $G/gtft_ens/tracked_armN_s$s \
      --out $OUT/gtft_ens_armN_s$s.json --save-frames $OUT/gtft_ens_armN_s$s.frames.pkl > $OUT/gtft_ens_armN_s$s.log 2>&1
    say "gtft_ens_armN seed $s scored"
  done
}

rescore_one gtft $G/gtft/masks.pkl "$G/gtft/logits_s%s.npz" $G/gtft &
rescore_one gtft_ens $G/gtft_ens/masks.pkl "$G/gtft_ens/logits_s%s.npz" $G/gtft_ens &
rescore_armN &
rescore_one final $G/final/masks.pkl "$G/final/logits_s%s.npz" $G/final &
wait

$PY - <<'E'
import json, math
bad = 0
for variant in ("gtft", "gtft_ens", "gtft_ens_armN", "final"):
    for s in (42, 43, 44):
        new = json.load(open(f"experiments/rescore/{variant}_s{s}.json"))
        old = json.load(open(f"docs/reports/gtbox_sam/{variant}_s{s}.json"))
        for cfg in ("gated (top 833)", "gated (top 833) +box clip", "single pass", "oracle classes"):
            if cfg not in old:
                continue
            for k in ("mIoU", "IoU", "mcIoU"):
                if not math.isclose(new[cfg][k], old[cfg][k], abs_tol=1e-9):
                    bad += 1
                    print("MISMATCH", variant, s, cfg, k, new[cfg][k], old[cfg][k])
print("reproduced exactly" if bad == 0 else f"{bad} mismatches")
E
say "rescore finished"
