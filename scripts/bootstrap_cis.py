"""Confidence intervals for the GT-box pipeline's mIoU, IoU and mcIoU on the official test set, from the per-frame class IoUs that
scripts/gtbox_sam_final_eval.py --save-frames writes (experiments/rescore/<variant>_s<seed>.frames.pkl).

Two bootstraps of the same estimator, B resamples each:
    case   resample the 5 test cases with replacement (cluster bootstrap; frames of a case move together). Honest about the number of
           independent units, but only 126 distinct resamples exist with 5 cases, so the interval is coarse.
    frame  resample frames with replacement. Ignores that frames of one case are correlated, so it is narrower and optimistic.
The three-seed mean is bootstrapped on the same resample for every seed. Paired differences between rungs of the ablation ladder use
the same resample too. The published TAPIS numbers have no spread, so they are compared as fixed values: the share of resamples in
which our metric exceeds them is reported, which says nothing about TAPIS's own variability.

Also written: the metrics of each case alone, and leave-one-case-out ranges.

Usage (titanxp, any python with numpy): python scripts/bootstrap_cis.py --dir experiments/rescore --out docs/reports/gtbox_sam/bootstrap_cis.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surgical_ai.evaluation.semantic_iou import aggregate  # noqa: E402

TAPIS = {"mIoU": 86.61, "IoU": 83.38, "mcIoU": 77.42}
TAPIS_VST = {"mIoU": 86.36, "IoU": 83.51, "mcIoU": 77.54}
VARIANTS = ["gtft", "gtft_ens", "gtft_ens_armN", "final"]
METRICS = ["mIoU", "IoU", "mcIoU"]
N_CLASSES = 7


def frame_arrays(frames: list) -> dict:
    """Per-frame arrays: mean IoU over ground-truth classes, over present classes, and the per-class values with presence flags."""
    n = len(frames)
    has_gt, m, has_any, i = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    V, P = np.zeros((n, N_CLASSES)), np.zeros((n, N_CLASSES))
    for k, (_name, class_ious, gt_classes) in enumerate(frames):
        if gt_classes:
            has_gt[k], m[k] = 1.0, float(np.mean([class_ious[c] for c in gt_classes]))
        if class_ious:
            has_any[k], i[k] = 1.0, float(np.mean(list(class_ious.values())))
        for label, value in class_ious.items():
            V[k, label - 1], P[k, label - 1] = value, 1.0
    return {"has_gt": has_gt, "m": m, "has_any": has_any, "i": i, "V": V, "P": P}


def metrics(a: dict, w: np.ndarray) -> np.ndarray:
    """mIoU, IoU, mcIoU (as fractions) of the frames weighted by w; w = 1 reproduces semantic_iou.aggregate."""
    miou = (w * a["has_gt"] * a["m"]).sum() / (w * a["has_gt"]).sum()
    iou = (w * a["has_any"] * a["i"]).sum() / (w * a["has_any"]).sum()
    den = (w[:, None] * a["P"]).sum(axis=0)
    num = (w[:, None] * a["P"] * a["V"]).sum(axis=0)
    mc = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).mean()
    return np.array([miou, iou, mc])


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--config", default="gated (top 833)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--b-case", type=int, default=10000)
    ap.add_argument("--b-frame", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    res: dict = {"config": args.config, "b_case": args.b_case, "b_frame": args.b_frame, "variants": {}}
    names = None
    data: dict = {}   # (variant, clip) -> list over seeds of arrays
    for variant in VARIANTS:
        for clip in (False, True):
            cfg = args.config + (" +box clip" if clip else "")
            per_seed = []
            for s in args.seeds:
                frames = pickle.loads((args.dir / f"{variant}_s{s}.frames.pkl").read_bytes())[cfg]
                fn = [f[0] for f in frames]
                if names is None:
                    names = fn
                assert fn == names, f"frame list differs for {variant} seed {s}"
                arrays = frame_arrays(frames)
                agg = aggregate([(f[1], f[2]) for f in frames])  # the repository's own metric code
                mine = metrics(arrays, np.ones(len(frames)))
                assert np.allclose(mine, [agg["mIoU"], agg["IoU"], agg["mcIoU"]], atol=1e-12), f"vectorised metric differs from aggregate: {mine} vs {agg}"
                per_seed.append(arrays)
            data[(variant, clip)] = per_seed
    cases = np.array([n.split("/")[0] for n in names])
    case_list = sorted(set(cases.tolist()))
    case_idx = {c: np.where(cases == c)[0] for c in case_list}
    res["cases"] = {c: int(len(ix)) for c, ix in case_idx.items()}
    n_frames = len(names)

    def mean_over_seeds(arrs: list, w: np.ndarray) -> np.ndarray:
        return np.mean([metrics(a, w) for a in arrs], axis=0)

    # weights of every resample: cluster (cases) and iid (frames)
    ones = np.ones(n_frames)
    case_w = np.zeros((args.b_case, n_frames))
    for b in range(args.b_case):
        draw = rng.choice(len(case_list), size=len(case_list), replace=True)
        for d in draw:
            case_w[b, case_idx[case_list[d]]] += 1
    frame_w = np.array([np.bincount(rng.integers(0, n_frames, n_frames), minlength=n_frames) for _ in range(args.b_frame)], dtype=float)

    boot: dict = {}
    for key, arrs in data.items():
        point = mean_over_seeds(arrs, ones)
        # the point estimate must equal semantic_iou.aggregate on the unweighted frames
        boot[key] = {"point": point,
                     "case": np.array([mean_over_seeds(arrs, w) for w in case_w]),
                     "frame": np.array([mean_over_seeds(arrs, w) for w in frame_w])}

    for variant in VARIANTS:
        for clip in (False, True):
            key = (variant, clip)
            arrs = data[key]
            entry = {"three_seed_mean": {}, "per_seed_point": {}, "per_case": {}, "leave_one_case_out": {}}
            for j, mname in enumerate(METRICS):
                p = 100 * boot[key]["point"][j]
                entry["three_seed_mean"][mname] = {
                    "point": p, "ci95_case": [100 * v for v in ci(boot[key]["case"][:, j])], "ci95_frame": [100 * v for v in ci(boot[key]["frame"][:, j])],
                    "share_above_tapis_case": float((100 * boot[key]["case"][:, j] > TAPIS[mname]).mean()),
                    "share_above_tapis_frame": float((100 * boot[key]["frame"][:, j] > TAPIS[mname]).mean()),
                    "share_above_tapis_vst_frame": float((100 * boot[key]["frame"][:, j] > TAPIS_VST[mname]).mean())}
            for s, a in zip(args.seeds, arrs):
                entry["per_seed_point"][str(s)] = dict(zip(METRICS, (100 * metrics(a, ones)).tolist()))
            for c, ix in case_idx.items():
                w = np.zeros(n_frames)
                w[ix] = 1
                entry["per_case"][c] = dict(zip(METRICS, (100 * mean_over_seeds(arrs, w)).tolist()))
                w = ones - w
                entry["leave_one_case_out"][c] = dict(zip(METRICS, (100 * mean_over_seeds(arrs, w)).tolist()))
            res["variants"][f"{variant}{' +box clip' if clip else ''}"] = entry

    # paired differences between rungs, unclipped, three-seed mean
    pairs = [("gtft_ens", "gtft"), ("gtft_ens_armN", "gtft_ens"), ("final", "gtft_ens_armN"), ("final", "gtft_ens"), ("final", "gtft")]
    res["paired_differences"] = {}
    for a_name, b_name in pairs:
        ka, kb = (a_name, False), (b_name, False)
        d = {}
        for j, mname in enumerate(METRICS):
            dc = 100 * (boot[ka]["case"][:, j] - boot[kb]["case"][:, j])
            df = 100 * (boot[ka]["frame"][:, j] - boot[kb]["frame"][:, j])
            d[mname] = {"point": float(100 * (boot[ka]["point"][j] - boot[kb]["point"][j])), "ci95_case": ci(dc), "ci95_frame": ci(df),
                        "share_positive_frame": float((df > 0).mean())}
        res["paired_differences"][f"{a_name} minus {b_name}"] = d

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    for v in ("final", "final +box clip", "gtft_ens", "gtft"):
        e = res["variants"][v]["three_seed_mean"]
        print(f"{v:<18}" + "  ".join(f"{m} {e[m]['point']:.2f} case[{e[m]['ci95_case'][0]:.2f},{e[m]['ci95_case'][1]:.2f}] frame[{e[m]['ci95_frame'][0]:.2f},{e[m]['ci95_frame'][1]:.2f}]" for m in METRICS))
    for k, d in res["paired_differences"].items():
        print(f"{k:<28}" + "  ".join(f"{m} {d[m]['point']:+.2f} frame[{d[m]['ci95_frame'][0]:+.2f},{d[m]['ci95_frame'][1]:+.2f}]" for m in METRICS))


if __name__ == "__main__":
    main()
