#!/bin/bash
# Checks every file of configs/rc_expected_sha256.txt under $WORK against its known SHA-256 (read-only).
# Usage: rc_verify.sh strict   : a missing or different file is an error (the start of the latency run)
#        rc_verify.sh lenient  : a missing file is only reported (setup, before the user has copied the laptop-side files)
MODE=${1:-strict}
WORK=${WORK:-$HOME/grasp_work}
LIST=$WORK/code/configs/rc_expected_sha256.txt
bad=0
while read -r sha path; do
  p=$WORK/$(echo "$path" | sed 's#@CK@#checkpoints/sam2#; s#@ET@#EdgeTAM/checkpoints#; s#@CODE@#code#')
  if [ ! -e "$p" ]; then
    echo "missing: $p"; [ "$MODE" = strict ] && bad=1; continue
  fi
  if [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$sha" ]; then echo "ok: ${p#$WORK/}"; else echo "DIFFERENT: $p"; bad=1; fi
done < "$LIST"
exit $bad
