"""Adaptive temporal depth: stop adding neighbouring frames once the fused belief is confident enough.

Frames are added outward from the keyframe, one step k = 1, 2, ..., 10 at a time (both sides, all valid frames with |offset| <= k). Tracking stops at the first step whose
frame-weighted evidential fusion (the paper's rule) has confidence >= theta. One parameter, theta, from a fixed grid.

  select   Choose theta on the official-training cases of the cross-validation fold caches (models that never saw those cases). Pre-stated rule, written before any
           test result is read: the smallest mean number of tracked frames among the grid values whose net corrections (corrections minus introduced errors, against the
           single pass) are at least 98% of those of the full +-10 window on the same instruments. No test case is read in this mode.
  apply    Apply the chosen theta to the official-split final models (three seeds): per-instrument stop depth for every instrument, the transition counts and mean frames
           for the instruments the budget-matched gate refines (the 941 highest S_1), and truncated track files for rescoring mIoU with scripts/gtbox_sam_final_eval.py.

    python scripts/adaptive_depth.py select --cv-dir ~/cv_gate --out select.json
    python scripts/adaptive_depth.py apply  --final-dir experiments/gtbox_sam/final --select select.json --out apply.json --trunc-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from gtbox_sam_final_eval import alpha_from_logits, fuse
from surgical_ai.evaluation.evidential import variance_scores

GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
KMAX = 10
TEST_CASES = {"CASE041", "CASE047", "CASE050", "CASE051", "CASE053"}
WEIGHTS = CONFIGS["four"]


def load_tracks(dirs: list[Path]) -> dict[int, tuple[np.ndarray, int, np.ndarray]]:
    out: dict = {}
    for d in dirs:
        for path in sorted(d.glob("*frames*.npz")):
            z = np.load(path)
            for key in z.files:
                if not key.startswith("det_"):
                    continue
                i = int(key[4:])
                if i in out:
                    continue
                det, center = z[key], int(z[f"center_{i}"])
                # the seed-43 and seed-44 track files predate the stored frame indices; their frames are taken as consecutive around the keyframe
                local = z[f"frames_{i}"] if f"frames_{i}" in z.files else np.arange(len(det))
                if det.ndim == 3 and len(det) >= 1 and np.isfinite(det).all():
                    out[i] = (det, center, local - local[center])
    return out


def stop_selection(det: np.ndarray, offsets: np.ndarray, theta: float | None) -> np.ndarray:
    """Boolean mask over the stored frames that tracking uses; theta None means the full window."""
    if theta is None:
        return np.ones(len(det), bool)
    sel = np.abs(offsets) <= 0
    for k in range(1, KMAX + 1):
        sel = np.abs(offsets) <= k
        if fuse(det[sel], WEIGHTS)[1] >= theta:
            return sel
    return sel


def cmd_select(args) -> None:
    os.environ.setdefault("GRASP_EXTRA_SPLITS", str(Path(os.environ["GRASP_DATA_ROOT"]) / "annotations" / "cv5_splits.json"))
    from surgical_ai.data.region_dataset import GraspRegionDataset

    data = Path(os.environ["GRASP_DATA_ROOT"])
    rows = []
    for k in range(5):
        fd = args.cv_dir / f"cv5_f{k}" / "final"
        z = np.load(fd / "logits.npz")
        ds = GraspRegionDataset(data, f"cv5_f{k}_test", letterbox=True)
        cases = np.array([f.split("/")[0] for f, _s, _b, _l in ds.instances])
        alpha = alpha_mix(lambda kk: z[f"det_{kk}"], WEIGHTS)
        base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(1)
        for i, (det, center, offs) in load_tracks([fd / "tracked"]).items():
            if cases[i] in TEST_CASES or not 0 <= i < len(base):
                continue
            rows.append((int(z["y"][i]), int(base[i]), det, offs))
    print(f"{len(rows)} refined instruments on official-training cases")
    res = {"n_instruments": len(rows), "grid": {}}
    for theta in [None] + GRID:
        corr = intro = frames = 0
        for gt, base, det, offs in rows:
            sel = stop_selection(det, offs, theta)
            pred = fuse(det[sel], WEIGHTS)[0]
            corr += base != gt and pred == gt
            intro += base == gt and pred != gt
            frames += int(sel.sum()) - 1
        res["grid"]["full" if theta is None else str(theta)] = {"corrections": int(corr), "introduced": int(intro), "net": int(corr - intro), "mean_frames": frames / len(rows)}
        print("full" if theta is None else theta, res["grid"]["full" if theta is None else str(theta)])
    full_net = res["grid"]["full"]["net"]
    ok = [(v["mean_frames"], t) for t, v in res["grid"].items() if t != "full" and v["net"] >= 0.98 * full_net]
    res["chosen_theta"] = float(min(ok)[1]) if ok else None
    res["rule"] = "smallest mean tracked frames among theta with net corrections >= 98% of the full window, on official-training-case instruments"
    args.out.write_text(json.dumps(res, indent=1))
    print("chosen theta:", res["chosen_theta"])


def cmd_apply(args) -> None:
    sel_cfg = json.loads(args.select.read_text())
    theta = sel_cfg["chosen_theta"]
    assert theta is not None, "no theta met the rule"
    out = {"theta": theta, "seeds": {}}
    for seed in (42, 43, 44):
        z = np.load(args.final_dir / f"logits_s{seed}.npz")
        y = z["y"]
        alpha = alpha_mix(lambda kk: z[f"det_{kk}"], WEIGHTS)
        base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(1)
        s1 = variance_scores(alpha)["epistemic"]
        gated = np.zeros(len(y), bool)
        gated[np.argsort(-s1, kind="stable")[:941]] = True
        tracks = load_tracks([args.final_dir / f"tracked_all_s{seed}"])
        tdir = args.trunc_dir / f"adaptive_s{seed}"
        tdir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        stat = {"full": [0, 0, 0], "early": [0, 0, 0]}  # corrections, introduced, tracked frames, over the gated instruments
        n_g = 0
        for i, (det, center, offs) in tracks.items():
            sel = stop_selection(det, offs, theta)
            new_center = int(np.cumsum(sel)[center] - 1)
            arrays[f"det_{i}"], arrays[f"center_{i}"] = det[sel], np.array(new_center)
            arrays[f"frames_{i}"] = np.where(sel)[0]
            if gated[i]:
                n_g += 1
                for tag, m in (("full", np.ones(len(det), bool)), ("early", sel)):
                    pred = fuse(det[m], WEIGHTS)[0]
                    stat[tag][0] += base[i] != y[i] and pred == y[i]
                    stat[tag][1] += base[i] == y[i] and pred != y[i]
                    stat[tag][2] += int(m.sum()) - 1
        np.savez_compressed(tdir / "frames_part0.npz", **arrays)
        out["seeds"][str(seed)] = {"gated": n_g, **{t: {"corrections": int(v[0]), "introduced": int(v[1]), "net": int(v[0] - v[1]), "mean_frames": v[2] / n_g} for t, v in stat.items()}}
        print(seed, out["seeds"][str(seed)])
    args.out.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("select")
    a.add_argument("--cv-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("apply")
    b.add_argument("--final-dir", type=Path, required=True)
    b.add_argument("--select", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--trunc-dir", type=Path, required=True)
    args = ap.parse_args()
    {"select": cmd_select, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
