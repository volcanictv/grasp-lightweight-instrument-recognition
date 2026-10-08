#!/bin/bash
# Stages what the campaign (scripts/campaign.py) needs on the RIT cluster. Run on the LAPTOP (Git Bash) after `ssh sporc hostname` works without a password or Duo.
# Everything goes under $WORK (default ~/grasp_work) on the cluster; NOTHING that already exists there is overwritten, moved, deleted or chmod-ed, and only NEW directories and files are created:
#   code_<commit>/                         `git archive` of the committed HEAD of this repository (a new directory named after the commit)
#   data/GraSP/annotations/                the short-term annotation files, the 5-fold registry and its split files (no long-term files, no segmentation PNGs)
#   data/GraSP/frames-001/frames/CASE*/    the frames of the 13 cases (about 14 GB), one case at a time; a case is only marked complete (.complete_CASE*) after its file count matches titanxp's
#   data/temporal_neighbors/fold{1,2}/     the neighbour crops of the official training cases (about 0.8 GB)
#   torch_home/hub/checkpoints/            the two torchvision ImageNet weights the classifier members start from (the compute nodes should not download anything)
# Data flows titanxp -> this laptop -> cluster (the two machines do not talk to each other). Safe to rerun: finished cases are skipped; an interrupted case is re-sent into a fresh directory name.
# A failed ssh stops the script at once and is never retried (too many failed logins can disable the RIT account).
set -uo pipefail
SRC=titanxp
DST=sporc
SRC_ROOT='$HOME/Desktop/Classification Surgurical Tools'
WORKR='$HOME/grasp_work'
SO="ssh -o BatchMode=yes -o ConnectTimeout=30"
say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "STOPPED: $*" >&2; exit 1; }
rtest() { $SO $DST "test -e $1"; local rc=$?; [ $rc -eq 255 ] && die "ssh to the cluster failed (no retry)"; return $rc; }

$SO $DST 'echo "cluster login ok: $(hostname)"; df -BG --output=avail "$HOME" | tail -1' || die "key login to the cluster failed (no retry); run scripts/rc_ssh_setup.ps1 first"
FREE=$($SO $DST 'df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9') || die "could not read free space"
[ "${FREE:-0}" -ge 60 ] || die "less than 60 GB free in the cluster home (${FREE:-?} GB)"
$SO $SRC true || die "titanxp not reachable"

COMMIT=$(git rev-parse --short HEAD) || die "not in the repository"
say "code at commit $COMMIT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || say "NOTE: uncommitted changes exist and are NOT included (only the committed HEAD is staged)"
if rtest "$WORKR/code_$COMMIT"; then say "code_$COMMIT already exists, left as it is"
else
  git archive --format=tar HEAD | $SO $DST "mkdir -p $WORKR && mkdir $WORKR/code_$COMMIT && tar -x -C $WORKR/code_$COMMIT" || die "code transfer failed"
  say "code staged in code_$COMMIT"
fi

say "annotations"
if rtest "$WORKR/data/GraSP/annotations/cv5_splits.json"; then say "annotations already staged"
else
  $SO $SRC "cd \"$SRC_ROOT/GraSP\" && tar -chf - annotations/cv5_splits.json annotations/grasp_short-term_*.json" \
    | $SO $DST "mkdir -p $WORKR/data/GraSP && tar -x -C $WORKR/data/GraSP --skip-old-files" || die "annotation transfer failed"
fi

say "torchvision ImageNet weights"
$SO $SRC 'cd ~/.cache/torch/hub/checkpoints && tar -cf - resnet50-11ad3fa6.pth mobilenet_v3_small-047dcff4.pth' \
  | $SO $DST "mkdir -p $WORKR/torch_home/hub/checkpoints && tar -x -C $WORKR/torch_home/hub/checkpoints --skip-old-files" || die "weights transfer failed"

for f in fold1 fold2; do
  if rtest "$WORKR/data/temporal_neighbors/$f/.complete"; then say "neighbour crops $f already staged"; continue; fi
  say "neighbour crops $f"
  $SO $SRC "cd ~/grasp_yolo26/experiments/temporal_neighbors && tar -cf - $f" | $SO $DST "mkdir -p $WORKR/data/temporal_neighbors && tar -x -C $WORKR/data/temporal_neighbors --skip-old-files" || die "neighbour crops $f failed"
  N1=$($SO $SRC "find ~/grasp_yolo26/experiments/temporal_neighbors/$f -type f | wc -l"); N2=$($SO $DST "find $WORKR/data/temporal_neighbors/$f -type f | wc -l")
  [ "$N1" = "$N2" ] || die "neighbour crops $f: $N1 files on titanxp, $N2 on the cluster"
  $SO $DST "touch $WORKR/data/temporal_neighbors/$f/.complete"
done

CASES=$($SO $SRC "ls \"$SRC_ROOT/GraSP/frames-001/frames\"") || die "could not list the cases"
for c in $CASES; do
  if rtest "$WORKR/data/GraSP/frames-001/frames/.complete_$c"; then say "$c already staged"; continue; fi
  if rtest "$WORKR/data/GraSP/frames-001/frames/$c"; then die "$c exists on the cluster but is not marked complete (an earlier run was interrupted): ask before continuing, nothing is overwritten"; fi
  say "$c"
  $SO $SRC "cd \"$SRC_ROOT/GraSP/frames-001/frames\" && tar -chf - $c" | $SO $DST "mkdir -p $WORKR/data/GraSP/frames-001/frames && tar -x -C $WORKR/data/GraSP/frames-001/frames --skip-old-files" || die "$c transfer failed"
  N1=$($SO $SRC "find -L \"$SRC_ROOT/GraSP/frames-001/frames/$c\" -type f | wc -l"); N2=$($SO $DST "find $WORKR/data/GraSP/frames-001/frames/$c -type f | wc -l")
  [ "$N1" = "$N2" ] || die "$c: $N1 files on titanxp, $N2 on the cluster"
  $SO $DST "touch $WORKR/data/GraSP/frames-001/frames/.complete_$c"
  say "$c complete ($N2 files)"
done
say "all staged:"; $SO $DST "du -sh $WORKR/data $WORKR/torch_home; ls $WORKR"
