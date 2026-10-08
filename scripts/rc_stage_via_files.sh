#!/bin/bash
# Staging through plain files, for when a piped ssh cannot show its Duo/passphrase prompt (Windows OpenSSH). Three steps, each run by you in a terminal on the laptop from the repository root:
#   bash scripts/rc_stage_via_files.sh build    no cluster login: collects everything from titanxp into one tar file per part under $STAGE (needs about 16 GB free on the laptop)
#   bash scripts/rc_stage_via_files.sh send     ONE scp of all the tar files to ~/grasp_work_incoming on the cluster (one login, one Duo push); nothing existing is touched
#   bash scripts/rc_stage_via_files.sh unpack   ONE ssh that extracts the tar files into ~/grasp_work (new paths only, --skip-old-files), checks every case's file count, and prints a report
# The cluster-side layout and the safety rules are those of scripts/rc_stage_campaign.sh. Nothing on the cluster is deleted or overwritten; the tar files in ~/grasp_work_incoming stay there until you delete them yourself.
set -uo pipefail
STAGE=${STAGE:-$HOME/rc_stage}
SRC=titanxp
DST=sporc
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "STOPPED: $*" >&2; exit 1; }
R='$HOME/Desktop/Classification Surgurical Tools/GraSP'
case "${1:-}" in
build)
  mkdir -p "$STAGE" && cd "$STAGE" || die "cannot create $STAGE"
  COMMIT=$(git -C "$OLDPWD" rev-parse --short HEAD) || die "run this from inside the repository"
  echo "$COMMIT" > commit.txt
  git -C "$OLDPWD" archive --format=tar --prefix="code_$COMMIT/" HEAD > 00_code.tar || die "git archive failed"
  ssh -o BatchMode=yes $SRC "cd \"$R\" && tar -chf - --transform='s,^,data/GraSP/,' annotations/cv5_splits.json annotations/grasp_short-term_*.json" > 01_annotations.tar || die "annotations"
  ssh -o BatchMode=yes $SRC "cd ~/.cache/torch/hub/checkpoints && tar -cf - --transform='s,^,torch_home/hub/checkpoints/,' resnet50-11ad3fa6.pth mobilenet_v3_small-047dcff4.pth" > 02_weights.tar || die "weights"
  ssh -o BatchMode=yes $SRC "cd ~/grasp_yolo26/experiments/temporal_neighbors && tar -cf - --transform='s,^,data/temporal_neighbors/,' fold1 fold2" > 03_neighbours.tar || die "neighbour crops"
  ssh -o BatchMode=yes $SRC "cd \"$R/frames-001/frames\" && for c in CASE*; do echo \"\$c \$(find -L \$c -type f | wc -l)\"; done" > counts_titanxp.txt || die "counts"
  for c in $(awk '{print $1}' counts_titanxp.txt); do
    say "$c"; ssh -o BatchMode=yes $SRC "cd \"$R/frames-001/frames\" && tar -chf - --transform='s,^,data/GraSP/frames-001/frames/,' $c" > "10_$c.tar" || die "$c"
  done
  say "built: $(ls | wc -l) files, $(du -sh . | cut -f1) in $STAGE"; ;;
send)
  [ -f "$STAGE/commit.txt" ] || die "run the build step first"
  cd "$STAGE" || die "no $STAGE"
  say "one scp of $(ls *.tar | wc -l) tar files; approve the Duo push when asked"
  scp -o ConnectTimeout=30 *.tar commit.txt counts_titanxp.txt "$DST":grasp_work_incoming/ || die "scp failed (no retry); if it says the directory does not exist, run: ssh $DST 'mkdir -p grasp_work_incoming' and then this step again"
  say "sent"; ;;
unpack)
  ssh $DST 'set -e; W=$HOME/grasp_work; I=$HOME/grasp_work_incoming
F=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9); [ "$F" -ge 40 ] || { echo "only $F GB free" >&2; exit 3; }
[ ! -e "$W/data/GraSP/frames-001" ] || { echo "data/GraSP/frames-001 already exists: not touching it" >&2; exit 4; }
mkdir -p "$W" && cd "$W"; for t in "$I"/*.tar; do echo "extracting $t"; tar -xf "$t" --skip-old-files; done
du -sh "$W/data" "$W/torch_home" "$W"/code_*
for c in "$W"/data/GraSP/frames-001/frames/CASE*; do echo "COUNT $(basename $c) $(find $c -type f | wc -l)"; done
cat "$I/counts_titanxp.txt" | sed "s/^/TITANXP /"' | tee "$STAGE/unpack_report.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "unpack failed"
  diff <(sort "$STAGE/counts_titanxp.txt") <(grep '^COUNT ' "$STAGE/unpack_report.txt" | awk '{print $2, $3}' | sort) && say "every case has the same file count on titanxp and the cluster: staging complete, code is in code_$(cat "$STAGE/commit.txt")" || die "file counts differ"; ;;
*) echo "usage: bash scripts/rc_stage_via_files.sh build|send|unpack"; exit 2; ;;
esac
