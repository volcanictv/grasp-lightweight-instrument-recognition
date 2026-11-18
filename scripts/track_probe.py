"""Per-frame ensemble probabilities along the +-10 s track of chosen test instruments (ground-truth masks), to see whether a distinctive cue (for LG the fenestrated jaw) appears in any frame.

    python scripts/track_probe.py --dir experiments/gtbox_sam/gtmask --sets /tmp/insp_sets.json --seed 42 --cls 6 --out /tmp/insp/track_probe.json
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

from evidential_seeds_e2e_eval import CONFIGS, ORDER
from gtbox_sam_final_eval import load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.evaluation.evidential import alpha_from_logits

CLASSES = ["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]


def frame_probs(det: np.ndarray, weights: dict) -> np.ndarray:
    alpha = sum(w * alpha_from_logits(det[:, ORDER.index(k), :]) for k, w in weights.items())
    return alpha / alpha.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cls", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    spec = json.loads(args.sets.read_text())
    tracked: dict = {}
    for d in TRACK_DIRS[args.seed]:
        tracked.update(load_tracked(args.dir / d))
    offsets: dict = {}
    for d in TRACK_DIRS[42]:
        for path in sorted((args.dir / d).glob("masks_shard*.pkl")):
            for idx, fr in pickle.loads(path.read_bytes())["masks"].items():
                offsets[idx] = [off for off in sorted(fr) if off == 0 or mask_codec.area(fr[off][1]) > 0]
    out = {}
    for set_name, idxs in spec["sets"].items():
        for idx in idxs:
            if idx not in tracked or len(offsets.get(idx, [])) != tracked[idx].shape[0]:
                print(set_name, idx, "no usable track")
                continue
            mu = frame_probs(tracked[idx], weights)
            pc = mu[:, args.cls]
            out[str(idx)] = {"set": set_name, "case": spec["meta"][str(idx)]["case"], "offsets": offsets[idx], "p_cls": [round(float(v), 3) for v in pc], "argmax": [CLASSES[int(a)] for a in mu.argmax(1)]}
            frames_over = int((pc >= 0.5).sum())
            print(f"{set_name[:2]} #{idx} {spec['meta'][str(idx)]['case']} true {spec['meta'][str(idx)]['true']}: frames {len(pc)}, max P_{CLASSES[args.cls]} {pc.max():.2f} (offset {offsets[idx][int(pc.argmax())]:+d}s), frames with P>=0.5: {frames_over}, mean {pc.mean():.2f}", flush=True)
            print("    " + " ".join(f"{o:+d}:{v:.2f}" for o, v in zip(offsets[idx], pc)))
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
