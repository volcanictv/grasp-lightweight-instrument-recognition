"""Bootstrap intervals for the held-out-threshold row of the paper (and its comparators), from the per-frame class IoUs that scripts/gtbox_sam_final_eval.py --save-frames writes for the refine-all tracks.

Configurations compared (3-seed means, the same resample for every seed and configuration): "single pass", the held-out-threshold gate "gated (tau 1.7e-05)", and refining every instrument "gated (top 2861)".
Two bootstraps as in scripts/bootstrap_cis.py: case (resample the 5 test cases, clustered, only 126 distinct resamples exist) and frame (iid, narrower, optimistic). Paired differences use the same resample.

    python scripts/bootstrap_heldout_row.py --dir experiments/rescore --prefix finalall --out docs/reports/gtbox_sam/bootstrap_heldout_row.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_cis import METRICS, ci, frame_arrays, metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--prefix", default="finalall")
    ap.add_argument("--configs", nargs="+", default=["single pass", "gated (tau 1.7e-05)", "gated (top 2861)"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--b-case", type=int, default=10000)
    ap.add_argument("--b-frame", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    data: dict[str, list] = {}
    names = None
    for cfg in args.configs:
        per_seed = []
        for s in args.seeds:
            frames = pickle.loads((args.dir / f"{args.prefix}_s{s}.frames.pkl").read_bytes())[cfg]
            fn = [f[0] for f in frames]
            names = names or fn
            assert fn == names, f"frame list differs for {cfg} seed {s}"
            per_seed.append(frame_arrays(frames))
        data[cfg] = per_seed
    cases = np.array([n.split("/")[0] for n in names])
    case_list = sorted(set(cases.tolist()))
    case_idx = {c: np.where(cases == c)[0] for c in case_list}
    n = len(names)
    ones = np.ones(n)
    case_w = np.zeros((args.b_case, n))
    for b in range(args.b_case):
        for d in rng.choice(len(case_list), size=len(case_list), replace=True):
            case_w[b, case_idx[case_list[d]]] += 1
    frame_w = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.b_frame)], dtype=float)

    def mean_over_seeds(arrs: list, w: np.ndarray) -> np.ndarray:
        return np.mean([metrics(a, w) for a in arrs], axis=0)

    boot = {cfg: {"point": mean_over_seeds(arrs, ones), "case": np.array([mean_over_seeds(arrs, w) for w in case_w]),
                  "frame": np.array([mean_over_seeds(arrs, w) for w in frame_w])} for cfg, arrs in data.items()}
    res: dict = {"configs": args.configs, "seeds": args.seeds, "b_case": args.b_case, "b_frame": args.b_frame, "cases": {c: int(len(ix)) for c, ix in case_idx.items()}, "estimates": {}, "paired_differences": {}}
    for cfg in args.configs:
        res["estimates"][cfg] = {"per_seed_point": {str(s): dict(zip(METRICS, (100 * metrics(a, ones)).tolist())) for s, a in zip(args.seeds, data[cfg])}, "three_seed_mean": {}, "leave_one_case_out": {}}
        for j, m in enumerate(METRICS):
            res["estimates"][cfg]["three_seed_mean"][m] = {"point": 100 * boot[cfg]["point"][j], "ci95_case": [100 * v for v in ci(boot[cfg]["case"][:, j])], "ci95_frame": [100 * v for v in ci(boot[cfg]["frame"][:, j])]}
        for c, ix in case_idx.items():
            w = np.ones(n)
            w[ix] = 0
            res["estimates"][cfg]["leave_one_case_out"][c] = dict(zip(METRICS, (100 * mean_over_seeds(data[cfg], w)).tolist()))
    for a, b in ((args.configs[0], args.configs[1]), (args.configs[2], args.configs[1]), (args.configs[0], args.configs[2])):   # (single, held-out) , (refine-all, held-out), (single, refine-all)
        key = f"{b} minus {a}"
        res["paired_differences"][key] = {}
        for j, m in enumerate(METRICS):
            dc, df = boot[b]["case"][:, j] - boot[a]["case"][:, j], boot[b]["frame"][:, j] - boot[a]["frame"][:, j]
            res["paired_differences"][key][m] = {"point": 100 * (boot[b]["point"][j] - boot[a]["point"][j]), "ci95_case": [100 * v for v in ci(dc)], "ci95_frame": [100 * v for v in ci(df)],
                                                  "share_resamples_positive_case": float((dc > 0).mean()), "share_resamples_positive_frame": float((df > 0).mean())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    for cfg in args.configs:
        print(f"{cfg:24s}" + "  ".join(f"{m} {v['point']:.2f} case[{v['ci95_case'][0]:.2f},{v['ci95_case'][1]:.2f}] frame[{v['ci95_frame'][0]:.2f},{v['ci95_frame'][1]:.2f}]" for m, v in res["estimates"][cfg]["three_seed_mean"].items()))
    for k, d in res["paired_differences"].items():
        print(f"{k}: " + "  ".join(f"{m} {v['point']:+.2f} case[{v['ci95_case'][0]:+.2f},{v['ci95_case'][1]:+.2f}] frame[{v['ci95_frame'][0]:+.2f},{v['ci95_frame'][1]:+.2f}]" for m, v in d.items()))


if __name__ == "__main__":
    main()
