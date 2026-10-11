"""Paired bootstrap intervals for the routing-signal (gate) comparison, from the per-frame class IoUs that scripts/gate_baselines_analysis.py --save-frames writes
(gate_frames_s<seed>.pkl, one entry per ordering and share of instruments refined).

For every share, the difference "S minus another ordering" in mIoU, IoU and mcIoU (3-seed mean) is bootstrapped on the same resample for both orderings:
    case   resample the 5 test cases with replacement (cluster bootstrap, only 126 distinct resamples exist, coarse but honest about the number of independent units)
    frame  resample frames with replacement (iid, narrower, optimistic)
as in scripts/bootstrap_heldout_row.py. The random ordering is the mean of its three permutations per seed.

    python scripts/bootstrap_gate_curves.py --dir ~/gate_frames --out docs/reports/gtbox_sam/bootstrap_gate_curves.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_cis import METRICS, ci, frame_arrays  # noqa: E402

SHARES = ["10%", "20%", "30%", "45%", "60%", "100%"]
BASE = "S"


def metrics_matrix(a: dict, W: np.ndarray) -> np.ndarray:
    """mIoU, IoU, mcIoU (fractions) for every weight row of W (B x n): the vectorised form of bootstrap_cis.metrics."""
    miou = (W @ (a["has_gt"] * a["m"])) / (W @ a["has_gt"])
    iou = (W @ (a["has_any"] * a["i"])) / (W @ a["has_any"])
    den = W @ a["P"]
    num = W @ (a["P"] * a["V"])
    mc = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).mean(axis=1)
    return np.stack([miou, iou, mc], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--b-case", type=int, default=10000)
    ap.add_argument("--b-frame", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    logs = [pickle.loads((args.dir / f"gate_frames_s{s}.pkl").read_bytes()) for s in args.seeds]
    names = [f[0] for f in logs[0]["single"]]
    for lg in logs:
        assert [f[0] for f in lg["single"]] == names, "frame list differs between seeds"
    cases = np.array([n.split("/")[0] for n in names])
    case_list = sorted(set(cases.tolist()))
    case_idx = {c: np.where(cases == c)[0] for c in case_list}
    n = len(names)
    case_w = np.zeros((args.b_case, n))
    for b in range(args.b_case):
        for d in rng.choice(len(case_list), size=len(case_list), replace=True):
            case_w[b, case_idx[case_list[d]]] += 1
    frame_w = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.b_frame)], dtype=float)
    ones = np.ones((1, n))

    orderings = sorted({k.split("|")[0] for k in logs[0] if "|" in k and not k.startswith("random")})
    orderings.append("random")

    def setting(order: str, share: str):
        """Per-seed list of frame-array groups; random is a group of its three permutations."""
        out = []
        for lg in logs:
            if order == "random":
                out.append([frame_arrays(lg[f"random{p}|{share}"]) for p in range(3)])
            elif share == "100%":
                out.append([frame_arrays(lg["refine_all"])])
            else:
                out.append([frame_arrays(lg[f"{order}|{share}"])])
        return out

    def boot(groups, W):
        """3-seed mean of the metrics under weights W (B x n); a random group is first averaged over its permutations."""
        return np.mean([np.mean([metrics_matrix(a, W) for a in g], axis=0) for g in groups], axis=0)

    single = [[frame_arrays(lg["single"])] for lg in logs]
    res: dict = {"seeds": args.seeds, "b_case": args.b_case, "b_frame": args.b_frame, "cases": {c: int(len(ix)) for c, ix in case_idx.items()},
                 "single": {}, "estimates": {}, "paired_S_minus": {}}
    sp, sc, sf = boot(single, ones)[0], boot(single, case_w), boot(single, frame_w)
    res["single"] = dict(zip(METRICS, (100 * sp).tolist()))
    cache: dict = {}
    for share in SHARES:
        for o in orderings:
            if share == "100%" and o != BASE:
                continue  # every ordering at 100% is refine-all
            g = setting(o, share)
            cache[(o, share)] = (boot(g, ones)[0], boot(g, case_w), boot(g, frame_w))
        res["estimates"][share] = {}
        res["paired_S_minus"][share] = {}
        for o in orderings:
            key = (o, share) if (o, share) in cache else (BASE, share)
            p, c, f = cache[key]
            res["estimates"][share][o] = {m: {"point": 100 * p[j], "ci95_case": [100 * v for v in ci(c[:, j])], "ci95_frame": [100 * v for v in ci(f[:, j])]} for j, m in enumerate(METRICS)}
            if o == BASE or share == "100%":
                continue
            pb, cb, fb = cache[(BASE, share)]
            res["paired_S_minus"][share][o] = {}
            for j, m in enumerate(METRICS):
                dc, df = cb[:, j] - c[:, j], fb[:, j] - f[:, j]
                res["paired_S_minus"][share][o][m] = {"point": 100 * (pb[j] - p[j]), "ci95_case": [100 * v for v in ci(dc)], "ci95_frame": [100 * v for v in ci(df)],
                                                      "share_resamples_positive_case": float((dc > 0).mean()), "share_resamples_positive_frame": float((df > 0).mean())}
    # the gain of every ordering over the single pass, with its own interval (the effort curve's error bars)
    res["gain_over_single"] = {}
    for share in SHARES:
        res["gain_over_single"][share] = {}
        for o in orderings:
            key = (o, share) if (o, share) in cache else (BASE, share)
            p, c, f = cache[key]
            res["gain_over_single"][share][o] = {m: {"point": 100 * (p[j] - sp[j]), "ci95_case": [100 * v for v in ci(c[:, j] - sc[:, j])], "ci95_frame": [100 * v for v in ci(f[:, j] - sf[:, j])]}
                                                 for j, m in enumerate(METRICS)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("mIoU, S minus other (points), case [95%] / frame [95%]")
    for share in SHARES[:-1]:
        print(f"-- {share} refined")
        for o, d in res["paired_S_minus"][share].items():
            v = d["mIoU"]
            print(f"   S - {o:22s} {v['point']:+.2f} case[{v['ci95_case'][0]:+.2f},{v['ci95_case'][1]:+.2f}] frame[{v['ci95_frame'][0]:+.2f},{v['ci95_frame'][1]:+.2f}]")


if __name__ == "__main__":
    main()
