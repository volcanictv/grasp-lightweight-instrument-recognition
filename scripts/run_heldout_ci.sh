#!/bin/bash
# Per-frame scores of the held-out-threshold pipeline (and single pass, refine-all) for the three seeds, then the bootstrap (scripts/bootstrap_heldout_row.py). CPU only; run niced.
# The tracks of the registered final run and of the refine-all run are merged per seed into experiments/gtbox_sam/final/tracked_all_s<seed> (copies of the small npz files, nothing existing is changed).
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/final
mkdir -p experiments/rescore docs/reports/gtbox_sam
say() { echo "[$(date +%H:%M:%S)] $*"; }
for s in 42 43 44; do
  if [ "$s" = 42 ]; then A=$D/tracked; B=$D/tracked_rest; else A=$D/tracked_s$s; B=$D/tracked_rest_s$s; fi
  T=$D/tracked_all_s$s
  mkdir -p $T
  i=0
  for f in $A/*frames*.npz $B/*frames*.npz; do cp -n "$f" "$T/frames_part$i.npz"; i=$((i+1)); done
  say "seed $s: merged $i track files into $T"
  nice -n 10 $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $T --tau 1.7e-5 --budgets 833 2861 \
    --out docs/reports/gtbox_sam/finalall_s$s.json --save-frames experiments/rescore/finalall_s$s.frames.pkl > experiments/rescore/finalall_s$s.log 2>&1
  say "seed $s scored: $(grep -E 'gated \(tau' experiments/rescore/finalall_s$s.log | head -n 1 | cut -c1-120)"
done
nice -n 10 $PY scripts/bootstrap_heldout_row.py --dir experiments/rescore --prefix finalall --out docs/reports/gtbox_sam/bootstrap_heldout_row.json
say "done"
