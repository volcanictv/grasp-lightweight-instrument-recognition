"""The registered decision rule of docs/reports/protocol_addendum_2026-10-08_tta.md: how much of the gain of tracking every instrument do 21 augmented views of the keyframe crop recover?

From the per-frame class IoUs written by scripts/gtbox_sam_final_eval.py --save-frames: experiments/rescore/finalall_s<seed>.frames.pkl (tracking, refine-all tracks) and experiments/rescore/tta_s<seed>.frames.pkl (TTA tracks).
g = (TTA-on-all mIoU - single pass) / (tracking-on-all mIoU - single pass), 3-seed mean; paired case and frame bootstrap of (tracking-on-all minus TTA-on-all), and the same for the gate at tau. mcIoU and IoU are checked in the same direction.

    python scripts/bootstrap_tta_vs_tracking.py --out docs/reports/gtbox_sam/tta_vs_tracking.json
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

SEEDS = (42, 43, 44)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("experiments/rescore"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b-case", type=int, default=10000)
    ap.add_argument("--b-frame", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = {"single": ("finalall", "single pass"), "tracking all": ("finalall", "gated (top 2861)"), "tracking at tau": ("finalall", "gated (tau 1.7e-05)"),
            "TTA all": ("tta", "gated (top 2861)"), "TTA at tau": ("tta", "gated (tau 1.7e-05)")}
    data, names = {}, None
    for key, (prefix, cfg) in spec.items():
        per = []
        for s in SEEDS:
            frames = pickle.loads((args.dir / f"{prefix}_s{s}.frames.pkl").read_bytes())[cfg]
            fn = [f[0] for f in frames]
            names = names or fn
            assert fn == names
            per.append(frame_arrays(frames))
        data[key] = per
    cases = np.array([n.split("/")[0] for n in names])
    ucases = sorted(set(cases.tolist()))
    case_idx = {c: np.where(cases == c)[0] for c in ucases}
    n = len(names)
    ones = np.ones(n)
    case_w = np.zeros((args.b_case, n))
    for b in range(args.b_case):
        for d in rng.choice(len(ucases), size=len(ucases), replace=True):
            case_w[b, case_idx[ucases[d]]] += 1
    frame_w = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.b_frame)], dtype=float)

    def mean_over_seeds(arrs, w):
        return np.mean([metrics(a, w) for a in arrs], axis=0)

    boot = {k: {"point": mean_over_seeds(v, ones), "case": np.array([mean_over_seeds(v, w) for w in case_w]), "frame": np.array([mean_over_seeds(v, w) for w in frame_w])} for k, v in data.items()}
    res = {"point": {k: dict(zip(METRICS, (100 * b["point"]).tolist())) for k, b in boot.items()},
           "per_seed_mIoU": {k: [100 * metrics(a, ones)[0] for a in v] for k, v in data.items()}, "g": {}, "paired": {}}
    for tta, trk, tag in (("TTA all", "tracking all", "on all instruments"), ("TTA at tau", "tracking at tau", "at tau")):
        g_point = [(boot[tta]["point"][j] - boot["single"]["point"][j]) / (boot[trk]["point"][j] - boot["single"]["point"][j]) for j in range(3)]
        g_case = np.array([(boot[tta]["case"][:, j] - boot["single"]["case"][:, j]) / (boot[trk]["case"][:, j] - boot["single"]["case"][:, j]) for j in range(3)])
        res["g"][tag] = {m: {"point": float(g_point[j]), "ci95_case": [float(np.percentile(g_case[j], 2.5)), float(np.percentile(g_case[j], 97.5))]} for j, m in enumerate(METRICS)}
        per_seed_g = [(metrics(data[tta][i], ones)[0] - metrics(data["single"][i], ones)[0]) / (metrics(data[trk][i], ones)[0] - metrics(data["single"][i], ones)[0]) for i in range(len(SEEDS))]
        res["g"][tag]["per_seed_g_mIoU"] = [float(x) for x in per_seed_g]
        res["paired"][f"{trk} minus {tta}"] = {m: {"point": 100 * float(boot[trk]["point"][j] - boot[tta]["point"][j]),
                                                    "ci95_case": [100 * float(np.percentile(boot[trk]["case"][:, j] - boot[tta]["case"][:, j], q)) for q in (2.5, 97.5)],
                                                    "ci95_frame": [100 * float(np.percentile(boot[trk]["frame"][:, j] - boot[tta]["frame"][:, j], q)) for q in (2.5, 97.5)]} for j, m in enumerate(METRICS)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    for k, v in res["point"].items():
        print(f"{k:16s}" + "  ".join(f"{m} {x:.2f}" for m, x in v.items()), " per-seed mIoU", [round(x, 2) for x in res["per_seed_mIoU"][k]])
    for tag, v in res["g"].items():
        print(f"g {tag}: " + "  ".join(f"{m} {v[m]['point']:.3f} [{v[m]['ci95_case'][0]:.3f}, {v[m]['ci95_case'][1]:.3f}]" for m in METRICS), " per-seed", [round(x, 3) for x in v["per_seed_g_mIoU"]])
    for k, v in res["paired"].items():
        print(f"{k}: " + "  ".join(f"{m} {v[m]['point']:+.2f} case[{v[m]['ci95_case'][0]:+.2f},{v[m]['ci95_case'][1]:+.2f}] frame[{v[m]['ci95_frame'][0]:+.2f},{v[m]['ci95_frame'][1]:+.2f}]" for m in METRICS))


if __name__ == "__main__":
    main()
