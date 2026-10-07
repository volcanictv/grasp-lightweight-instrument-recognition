#!/bin/bash
# The A100 latency measurements, run INSIDE a Slurm job on one GPU (slurm/rc_latency.sbatch) or on titanxp for a dry run. Two groups, each step writing a JSON.
#
# Components (scripts/benchmark_rt_segmenters.py, scripts/benchmark_tracker_latency.py):
#   segmenter_latency   SAM2.1 large / small / tiny (fp32, bf16, flip pass) and the four-member classifier, 60 test frames
#   classify            each classifier member and the ensemble on one synthetic crop
#   yolo                YOLO26s-seg on one frame, imgsz 640
#   sam2_large_noncausal / sam2_large_causal   SAM2-large propagation per instrument: +-10 frames both ways / the past 20 frames only
#   edgetam_noncausal / edgetam_causal         the same for EdgeTAM, in its own environment
#
# Whole pipelines (scripts/benchmark_pipeline_e2e.py): decoded frame to final label per keyframe, 32 test keyframes with their windows, gate tau 2.85e-4, tracks timed for every instrument too
#   p1_rt_none          real time, SAM2.1 tiny + flip, four members, gate, no tracking (also the mask/gate source of p3)
#   p2_rt_yolo          real time, causal YOLO26s tracking of the gated instruments (past 20 frames)
#   p3_rt_edgetam       real time, causal EdgeTAM tracking (own environment: tracking timed on p1's masks and gate)
#   p4_causal_sam2      causal SAM2-large tracking of the tiny masks (past 20 frames)
#   p5_noncausal_final  the final pipeline: SAM2.1 large + SAM3 masks (flip-averaged), SAM2-large tracking over +-10 frames (non-causal)
#   p6_causal_final     the same masks, SAM2-large tracking over the past 20 frames (causal)
# then scripts/summarize_pipeline_latency.py writes pipeline_latency.json.
#
# All paths come from environment variables with defaults under $WORK; nothing outside $WORK and the private temp directory is read or written, and the only deletion is that private temp
# directory. DRY=1 uses few repetitions and keyframes.
set -uo pipefail
WORK=${WORK:-$HOME/grasp_work}
CODE=${CODE:-$WORK/code}
BUNDLE=${BUNDLE:-$WORK/bundle}
CK=${CK:-$WORK/checkpoints/sam2}
EDGETAM=${EDGETAM:-$WORK/EdgeTAM}
YOLOW=${YOLOW:-$WORK/weights_extra/yolo26s_official_last.pt}
PY_MAIN=${PY_MAIN:-$WORK/envs/main/bin/python}
PY_YOLO=${PY_YOLO:-$PY_MAIN}   # the environment that holds ultralytics (the cluster main environment does)
PY_ET=${PY_ET:-$WORK/envs/edgetam/bin/python}
PY_SAM3=${PY_SAM3:-$PY_MAIN}   # the environment with transformers 5.x for SAM3 (the cluster main environment has it)
WT=${WT:-$WORK/weights_ft}   # the fine-tuned segmenters of the paper: tiny_all8.pt, large_all8.pt (SAM2), sam3_all8.pt
GATE_TINY=""; GATE_FINAL=""
[ "${USE_SAVED_GATE:-0}" = 1 ] && { GATE_TINY="--gate-saved $BUNDLE/gate_saved_rt.json"; GATE_FINAL="--gate-saved $BUNDLE/gate_saved_final.json"; }  # dry runs with base checkpoints only
SAM3_FLAG=""; [ "${DRY:-0}" = 1 ] && [ "${SAM3_RANDOM:-0}" = 1 ] && SAM3_FLAG="--sam3-random-init"
RES=${RES:-$WORK/results/run_${SLURM_JOB_ID:-manual}}
RUNS=200; FRAMES=60; LIMIT=0; NINST=20
if [ "${DRY:-0}" = 1 ]; then RUNS=10; LIMIT=${LIMIT_DRY:-6}; NINST=4; fi  # FRAMES must stay 60: the bundle holds exactly the 63 frames that rule selects (60 timed plus 3 warm-up)
TMPD=/tmp/${USER}_latency_${SLURM_JOB_ID:-manual}
ENS=configs/rc_ens4_N_s44.yaml
SAM2L=configs/sam2.1/sam2.1_hiera_l.yaml
say() { echo "[$(date +%H:%M:%S)] $*"; }

cd "$CODE" || { echo "no code directory $CODE"; exit 1; }
mkdir -p "$RES" "$TMPD"
export GRASP_DATA_ROOT=$BUNDLE/GraSP SAM2_CKPT_DIR=$CK BENCH_TMP=$TMPD/prop PYTHONUNBUFFERED=1
say "node $(hostname), job ${SLURM_JOB_ID:-none}"
(cd "$BUNDLE" && sha256sum -c --quiet MANIFEST.sha256) || { echo "the test-data bundle differs from its manifest"; exit 1; }
say "bundle verified"
[ "${DRY:-0}" = 1 ] || WORK=$WORK bash "$CODE/scripts/rc_verify.sh" strict > "$RES/verify_files.txt" 2>&1 || { cat "$RES/verify_files.txt"; echo "a required file is missing or differs"; exit 1; }
say "weights verified"
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu,clocks.max.sm --format=csv > "$RES/gpu.txt" 2>&1
$PY_MAIN -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda)" > "$RES/versions_main.txt" 2>&1
$PY_ET -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda)" > "$RES/versions_edgetam.txt" 2>&1
git -C "$CODE" rev-parse HEAD > "$RES/code_commit.txt" 2>&1
say "idle check: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"

step() { # name, command...  : runs the command with its output in $RES/<name>.log
  local n=$1; shift
  say "$n"
  "$@" > "$RES/$n.log" 2>&1 || say "FAILED $n"
}
prop() { # name, python, config, checkpoint, extra args : propagation of one instrument
  local n=$1 py=$2 cfg=$3 ck=$4; shift 4
  step "$n" $py scripts/benchmark_tracker_latency.py --stage propagate --sam2-config "$cfg" --sam2-checkpoint "$ck" --error-cases-json "$BUNDLE/idx_tracker.json" --split test \
    --n-instances $NINST --out "$RES/$n.json" "$@"
}
e2e() { # name, python, extra args
  local n=$1 py=$2; shift 2
  step "$n" $py scripts/benchmark_pipeline_e2e.py --ensemble-config $ENS --keyframes "$BUNDLE/e2e_keyframes.json" --limit $LIMIT --all-instruments --tau 2.85e-4 --out "$RES/$n.json" "$@"
}

step segmenter_latency $PY_MAIN scripts/benchmark_rt_segmenters.py --frames $FRAMES --masks "$BUNDLE/masks_subset.pkl" --ensemble-config $ENS --out "$RES/segmenter_latency.json"
step classify $PY_MAIN scripts/benchmark_tracker_latency.py --stage classify --ensemble-config $ENS --runs $RUNS --out "$RES/classify.json"
step yolo $PY_YOLO scripts/benchmark_tracker_latency.py --stage yolo --yolo-weights "$YOLOW" --runs $RUNS --out "$RES/yolo.json"
prop sam2_large_noncausal $PY_MAIN $SAM2L "$CK/sam2.1_hiera_large.pt" --window 10
prop sam2_large_causal $PY_MAIN $SAM2L "$CK/sam2.1_hiera_large.pt" --window 20 --causal
prop edgetam_noncausal $PY_ET configs/edgetam.yaml "$EDGETAM/checkpoints/edgetam.pt" --window 10
prop edgetam_causal $PY_ET configs/edgetam.yaml "$EDGETAM/checkpoints/edgetam.pt" --window 20 --causal

e2e p1_rt_none $PY_MAIN --mode full --seg tiny --tracker none $GATE_TINY --sam2-weights "$WT/tiny_all8.pt" --causal --window 20
e2e p2_rt_yolo $PY_YOLO --mode full --seg tiny --tracker yolo $GATE_TINY --sam2-weights "$WT/tiny_all8.pt" --yolo-weights "$YOLOW" --causal --window 20
e2e p3_rt_edgetam $PY_ET --mode track --tracker sam2 --gate-in "$RES/p1_rt_none.json" --sam2-config configs/edgetam.yaml --sam2-checkpoint "$EDGETAM/checkpoints/edgetam.pt" --causal --window 20
e2e p4_causal_sam2 $PY_MAIN --mode full --seg tiny --tracker sam2 $GATE_TINY --sam2-weights "$WT/tiny_all8.pt" --sam2-config $SAM2L --sam2-checkpoint "$CK/sam2.1_hiera_large.pt" --causal --window 20
e2e p5_noncausal_final $PY_SAM3 --mode full --seg sam23 $GATE_FINAL --sam2-weights "$WT/large_all8.pt" --sam3-weights "$WT/sam3_all8.pt" $SAM3_FLAG --tracker sam2 --sam2-config $SAM2L --sam2-checkpoint "$CK/sam2.1_hiera_large.pt" --window 10
e2e p6_causal_final $PY_SAM3 --mode full --seg sam23 $GATE_FINAL --sam2-weights "$WT/large_all8.pt" --sam3-weights "$WT/sam3_all8.pt" $SAM3_FLAG --tracker sam2 --sam2-config $SAM2L --sam2-checkpoint "$CK/sam2.1_hiera_large.pt" --causal --window 20
step summary $PY_MAIN scripts/summarize_pipeline_latency.py rt_none="$RES/p1_rt_none.json@rt" rt_yolo="$RES/p2_rt_yolo.json@rt" rt_edgetam="$RES/p3_rt_edgetam.json@rt" \
  causal_sam2="$RES/p4_causal_sam2.json@rt" noncausal_final="$RES/p5_noncausal_final.json@final" causal_final="$RES/p6_causal_final.json@final" --tau-file "$BUNDLE/gate_tau.json" --out "$RES/pipeline_latency.json"

case "$TMPD" in /tmp/${USER}_latency_*) rm -rf "$TMPD";; esac  # the only deletion: this job's private temp directory
say "done: $(ls "$RES"/*.json 2>/dev/null | wc -l) result files in $RES"
