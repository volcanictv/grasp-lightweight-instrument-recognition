"""How often does a propagated track stay on one object? For the instruments that tracking fixed (keyframe class wrong, gated, fused class right), the number of consecutive tracked
frames, walking outward from the keyframe, whose masks overlap the previous one with IoU >= 0.3. At one frame per second the instruments move a lot, so this is a conservative
measure; it shows how much of the correction can rely on following a single instrument. Reads saved masks, logits and tracks only.

Usage (titanxp, surgical environment, repo root; needs experiments/figure_examples/<name>.json from scripts/find_figure_examples.py):
    python scripts/track_stability.py --out docs/reports/realtime/track_stability.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec

RUNS = {"causal EdgeTAM (SAM2-tiny masks)": ("rt_tiny", "tracked_b", "cand_rt"), "non-causal SAM2-large (SAM2 + SAM3 masks)": ("final", "tracked", "candidates")}


def kept_frames(track: dict, offsets: list[int]) -> int:
    prev = track[0][1]
    good = 0
    for o in offsets:
        cur = track[o][1]
        if mask_codec.iou([cur], [prev], [0])[0][0] >= 0.3:
            good += 1
            prev = cur
        else:
            break
    return good


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for name, (run, tracks_dir, cand) in RUNS.items():
        cands = json.loads((REPO_ROOT / "experiments" / "figure_examples" / f"{cand}.json").read_text())
        tracks = {}
        for p in sorted((REPO_ROOT / "experiments" / "gtbox_sam" / run / tracks_dir).glob("masks_shard*.pkl")):
            tracks.update(pickle.loads(p.read_bytes())["masks"])
        kept = []
        for c in cands:
            t = tracks[c["index"]]
            offs = sorted(t)
            kept.append(kept_frames(t, [o for o in offs if o > 0]) + kept_frames(t, [o for o in sorted(offs, reverse=True) if o < 0]))
        kept = np.array(kept)
        out[name] = {"fixed_instruments": int(len(kept)), "share_kept_ge_8_frames": float((kept >= 8).mean()), "share_kept_ge_3_frames": float((kept >= 3).mean()),
                     "median_frames_kept": float(np.median(kept)), "max_possible_frames": int(max(len(tracks[c["index"]]) - 1 for c in cands))}
        print(name, out[name])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
