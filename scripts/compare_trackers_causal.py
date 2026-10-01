"""Scores trackers on the same gated fold1 instances with the full +/-window and with causal windows
(the last k past frames plus the annotated frame), from cached per-frame member logits. Past frames
of SAM2 / EdgeTAM / YOLO tracks do not depend on the forward pass, so no tracking is re-run.

Usage:
    python scripts/compare_trackers_causal.py --tau 0.000575 \\
        --tracker sam2=<dir with fold1_frames_shard*.npz> --tracker edgetam=<dir> --tracker yolo=<dir>
A tracker directory holds *frames*.npz (det_<idx> logits shaped (frames, members, classes); center_<idx>
when the writer stored it) and, for YOLO, fold1_tracked_shard0.json (frames_backward gives the center).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

import evidential_three_member_yolo as t


def load(directory: Path) -> dict[int, tuple[np.ndarray, int]]:
    """index -> (frame logits, center position)."""
    centers_json: dict[int, int] = {}
    for path in directory.glob("*tracked_shard*.json"):
        for r in json.loads(path.read_text())["instances"]:
            if "frames_backward" in r:
                centers_json[r["index"]] = r["frames_backward"]
    out = {}
    for path in sorted(directory.glob("fold1_frames*.npz")):  # official files share the det_<idx> key space
        z = np.load(path)
        for key in z.files:
            if not key.startswith("det_"):
                continue
            idx = int(key[4:])
            if f"center_{idx}" in z.files:
                center = int(z[f"center_{idx}"])
            elif idx in centers_json:
                center = centers_json[idx]
            else:
                continue
            out[idx] = (z[key], center)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--tracker", action="append", required=True, help="name=directory")
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_fold1_s42.npz")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_fold1_s42_lam0p01a10.yaml")
    ap.add_argument("--ks", default="3,5,10")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    order = t.member_order(args.config)
    y, base, s1 = t.base_scores(args.extract)
    gated = np.where(s1 >= args.tau)[0]
    trackers = {n: load(Path(d)) for n, d in (x.split("=", 1) for x in args.tracker)}
    usable = [i for i in gated if all(int(i) in tr for tr in trackers.values())]
    print(f"gated {len(gated)}, instances present in every tracker {len(usable)}; base accuracy {(base == y).mean():.4f}")
    modes = {"full window": None, **{f"causal k={k}": int(k) for k in args.ks.split(",")}}
    rows = {}
    print(f"{'tracker':<12}{'window':<16}{'acc':>8}{'macroF1':>9}{'GraspF1':>9}{'ClipF1':>8}{'fix':>5}{'brk':>5}{'frames':>8}")
    for name, tr in trackers.items():
        for mode, k in modes.items():
            pred = base.copy()
            used = []
            for i in usable:
                logits, c = tr[int(i)]
                lo = 0 if k is None else max(0, c - k)
                hi = len(logits) if k is None else c + 1
                pred[i] = t.track_predict(logits[lo:hi], order)
                used.append(hi - lo)
            f = f1_score(y, pred, average=None, labels=range(7))
            flag = np.zeros(len(y), bool)
            flag[usable] = True
            fixed, broken = int(((base != y) & (pred == y) & flag).sum()), int(((base == y) & (pred != y) & flag).sum())
            rows[f"{name}|{mode}"] = {"accuracy": float((pred == y).mean()), "macro_f1": float(f.mean()), "per_class_f1": f.round(4).tolist(),
                                      "fixed": fixed, "broken": broken, "mean_frames": float(np.mean(used))}
            print(f"{name:<12}{mode:<16}{(pred == y).mean():>8.4f}{f.mean():>9.4f}{f[6]:>9.3f}{f[5]:>8.3f}{fixed:>5}{broken:>5}{np.mean(used):>8.1f}")
    if args.out:
        args.out.write_text(json.dumps({"tau": args.tau, "n_gated": int(len(usable)), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
