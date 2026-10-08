#!/bin/bash
# Stages what the campaign (scripts/campaign.py) needs on the RIT cluster, through ONE ssh connection (one login, one Duo push, which only YOU can approve).
# Run it yourself in a terminal on the laptop (Git Bash) from the repository root:   bash scripts/rc_stage_campaign.sh
# Data flows titanxp -> this laptop -> cluster as one tar stream (the laptop only relays it; nothing is stored on the laptop).
#
# On the cluster it creates only NEW paths under $HOME/grasp_work and never overwrites (tar --skip-old-files), moves, deletes or chmods anything:
#   code_<commit>/                         `git archive` of the committed HEAD of this repository
#   data/GraSP/annotations/                the short-term annotation files, the 5-fold registry and its split files (no long-term files, no segmentation PNGs)
#   data/GraSP/frames-001/frames/CASE*/    the frames of the 13 cases (about 14 GB)
#   data/temporal_neighbors/fold{1,2}/     the neighbour crops of the official training cases (about 0.8 GB)
#   torch_home/hub/checkpoints/            the two torchvision ImageNet weights the classifier members start from (the compute nodes must not download anything)
# It refuses to run if data/GraSP/frames-001 already exists on the cluster (an earlier, possibly interrupted run): then ask before doing anything, because a half-written file would be kept.
# At the end the cluster side prints the file count of every case and the script compares it with titanxp's.
set -uo pipefail
SRC=titanxp
DST=sporc
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "STOPPED: $*" >&2; exit 1; }

COMMIT=$(git rev-parse --short HEAD) || die "run this from inside the repository"
[ -z "$(git status --porcelain --untracked-files=no)" ] || say "NOTE: uncommitted changes exist and are NOT included (only the committed HEAD is staged)"
ssh -o BatchMode=yes -o ConnectTimeout=25 $SRC 'cd "$HOME/Desktop/Classification Surgurical Tools/GraSP/frames-001/frames" && for c in CASE*; do echo "$c $(find -L $c -type f | wc -l)"; done' > /tmp/rc_stage_titanxp_counts.txt || die "titanxp not reachable"
say "titanxp: $(wc -l < /tmp/rc_stage_titanxp_counts.txt) cases, $(awk '{s+=$2} END {print s}' /tmp/rc_stage_titanxp_counts.txt) frame files"

REMOTE='set -e; W=$HOME/grasp_work; F=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$F" -ge 60 ] || { echo "only $F GB free in the home directory" >&2; exit 3; }
[ ! -e "$W/data/GraSP/frames-001" ] || { echo "data/GraSP/frames-001 already exists on the cluster: not touching it" >&2; exit 4; }
mkdir -p "$W" && cd "$W" && tar -xif - --skip-old-files
echo "extracted into $W"; du -sh "$W/data" "$W/torch_home" "$W"/code_*
for c in "$W"/data/GraSP/frames-001/frames/CASE*; do echo "COUNT $(basename $c) $(find $c -type f | wc -l)"; done'

SRCSIDE='set -e; R="$HOME/Desktop/Classification Surgurical Tools/GraSP"
cd "$R" && tar -chf - --transform="s,^,data/GraSP/," annotations/cv5_splits.json annotations/grasp_short-term_*.json
cd ~/.cache/torch/hub/checkpoints && tar -cf - --transform="s,^,torch_home/hub/checkpoints/," resnet50-11ad3fa6.pth mobilenet_v3_small-047dcff4.pth
cd ~/grasp_yolo26/experiments/temporal_neighbors && tar -cf - --transform="s,^,data/temporal_neighbors/," fold1 fold2
cd "$R/frames-001/frames" && tar -chf - --transform="s,^,data/GraSP/frames-001/frames/," CASE*'

say "streaming code, annotations, weights, neighbour crops and frames (one login: approve the Duo push on your phone)"
{ git archive --format=tar --prefix="code_$COMMIT/" HEAD; ssh -o BatchMode=yes $SRC "$SRCSIDE"; } | dd bs=4M status=progress | ssh $DST "$REMOTE" | tee /tmp/rc_stage_cluster_report.txt   # dd prints the bytes streamed so far (about 15 GB in total)
rc=${PIPESTATUS[2]}
[ "$rc" -eq 0 ] || die "the transfer failed (exit $rc); nothing was retried"
grep '^COUNT ' /tmp/rc_stage_cluster_report.txt | awk '{print $2, $3}' | sort > /tmp/rc_stage_cluster_counts.txt
sort /tmp/rc_stage_titanxp_counts.txt | diff - /tmp/rc_stage_cluster_counts.txt && say "every case has the same file count on titanxp and the cluster: staging complete, code is in code_$COMMIT" || die "file counts differ (see the diff above)"
