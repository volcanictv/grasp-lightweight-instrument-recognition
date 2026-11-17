#!/bin/bash
# Copy the staged SlowFast windows and run directories from the cluster to titanxp through this machine, once, and verify them. Runs from Git Bash on the laptop.
# Why a relay: it needs no new credentials or network rules (the laptop already reaches both machines), and the data is copied to titanxp's local disk so training never reads over the network.
# The cluster key needs its passphrase: the caller sets SSH_ASKPASS, SSH_ASKPASS_REQUIRE=force and DISPLAY. Refuses to run before the staging job has written the _ready marker or if titanxp lacks space.
set -uo pipefail
CL=/c/Windows/System32/OpenSSH/ssh.exe; TX=${TX:-titanxp}
G=${G:-${GRASP30FPS_DIR:?set G or GRASP30FPS_DIR}/windows}; RUNS_ROOT=${RUNS_ROOT:-'~/grasp_work/slowfast/data'}; DEST=${DEST:-'~/sf30'}
say() { echo "[$(date +%T)] $*"; }
cs() { "$CL" -o PreferredAuthentications=publickey -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 sporc "$@"; }
tx() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$TX" "$@"; }

cs "cat $G/_ready" 2>/dev/null | head -n 1 | grep -q files || { say "windows are not ready on the cluster (no _ready marker); stopping"; exit 2; }
bytes=$(cs "cd $G && xargs -a _files.txt -d '\n' stat -c %s | paste -sd+ | bc") || exit 3
free=$(tx 'df --output=avail -B1 ~ | tail -n 1') || exit 3
say "windows: $((bytes / 1048576)) MB; free on $TX: $((free / 1048576)) MB"
[ "$free" -gt $((bytes + 8 * 1073741824)) ] || { say "not enough free space on $TX (need the data plus 8 GB for checkpoints); stopping"; exit 4; }

tx "mkdir -p $DEST/windows $DEST/data $DEST/runs $DEST/scripts"
say "streaming windows"
cs "cd $G && tar cf - -T _files.txt" | tx "tar xf - -C $DEST/windows" || { say "window stream failed"; exit 5; }
cs "cat $G/_manifest.sha256" | tx "cat > $DEST/windows/_manifest.sha256"
say "streaming run directories (dereferenced)"
cs "cd $RUNS_ROOT && tar chf - run_official run_fold1 run_fold2" | tx "tar xf - -C $DEST/data" || { say "run directory stream failed"; exit 6; }
say "verifying checksums on $TX"
tx "cd $DEST/windows && sha256sum -c --quiet _manifest.sha256 && touch _verified && echo all \$(wc -l < _manifest.sha256) files verified" || { say "CHECKSUM MISMATCH"; exit 7; }
say "done"
