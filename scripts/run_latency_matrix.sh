#!/bin/bash
# Frame-level latency of the three trackers on a busy segment (about 4 instruments per frame), one GPU, nothing
# else running. Every refinement finishes before the next frame (--sync-refine), so a frame's latency is the full
# cost of ingesting it, all instruments as one unit. Settings per tracker:
#   natural   official gate (2.85e-4)
#   share13   gate set so about 13% of instruments fire, the share seen on the official test
#   allgated  gate 0, every instrument tracked: the worst case
# Usage (titanxp, repo root): scripts/run_latency_matrix.sh SEGMENT_DIR YOLO_WEIGHTS
set -uo pipefail
SEG=$1; W=$2
PY=$HOME/yolo26_venv/bin/python
OUT=experiments/rt_latency
mkdir -p $OUT
BASE=configs/realtime_grasp_edgetam.yaml

mkcfg() {  # name tracker sam_config threshold
  sed -e "s/name: edgetam .*/name: $2/" -e "s#sam_config: .*#sam_config: $3#" -e "s/threshold: 0.000285/threshold: $4/" $BASE > $OUT/cfg_$1.yaml
}

run() {  # cfgname outname extra-args
  $PY scripts/realtime_stream_test.py --config $OUT/cfg_$1.yaml --frames-dir $SEG --yolo-weights "$W" --device cuda:0 \
    --sam-checkpoint "$SAMCKPT" --sync-refine --out $OUT/$2.json "${@:3}" > $OUT/$2.log 2>&1
}

mkcfg probe none configs/edgetam.yaml 0.000285
SAMCKPT=none; run probe s1_probe
TAU13=$($PY -c "import json,numpy as np;print(float(np.quantile(json.load(open('$OUT/s1_probe.json'))['s1_values'],0.87)))")
echo "share13 threshold $TAU13" > $OUT/tau13.txt

for tracker in match edgetam sam2; do
  case $tracker in
    match)   sam=configs/edgetam.yaml; SAMCKPT=none; frames_all=90; pyx=$PY;;
    edgetam) sam=configs/edgetam.yaml; SAMCKPT=$HOME/EdgeTAM/checkpoints/edgetam.pt; frames_all=40;;
    sam2)    sam=configs/sam2.1/sam2.1_hiera_l.yaml; SAMCKPT=$HOME/sam2/checkpoints/sam2.1_hiera_large.pt; frames_all=20;;
  esac
  for setting in natural share13 allgated; do
    case $setting in natural) tau=0.000285; mf=90;; share13) tau=$TAU13; mf=90;; allgated) tau=0.0; mf=$frames_all;; esac
    mkcfg ${tracker}_$setting $tracker $sam $tau
    if [ $tracker = edgetam ]; then
      PYTHONPATH=$HOME/EdgeTAM:$HOME/edgetam_venv/lib/python3.11/site-packages run ${tracker}_$setting ${tracker}_$setting --max-frames $mf
    else
      run ${tracker}_$setting ${tracker}_$setting --max-frames $mf
    fi
  done
done
echo done > $OUT/done.flag
