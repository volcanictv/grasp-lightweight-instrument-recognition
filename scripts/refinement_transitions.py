"""What temporal refinement fixes and damages, per instrument, for the policies already on record (no training, no new gate, no threshold search).

Inputs are those of scripts/refine_all_analysis.py: the final ensemble logits (logits_s<seed>.npz) and the refine-all tracks (tracked + tracked_rest for seed 42,
tracked_s<seed> + tracked_rest_s<seed> for seeds 43 and 44) of the official test split. Every instrument becomes one row per seed with the single-pass and the refine-all
class, and falls in one of four groups: wrong->correct (tracking fixes it), correct->wrong (tracking damages it), correct->correct, wrong->wrong.

Policies (selected set -> refined; the rest keep the single-pass class, but their stored refine-all outcome is reported as "if refined"):
  tau_1.7e-5           S1 >= tau (the fixed threshold of the shipped pipeline)
  s1_top833            the 833 highest S1 (the exploratory operating point)
  maxbelief_n_tau      the same number as tau_1.7e-5, ranked by 1 - largest belief (same evidential ensemble)
  maxbelief_n_top833   the same number as s1_top833, ranked by 1 - largest belief
  refine_all           everything
Ties in a ranking are broken by instrument index (stable sort). An instrument without a valid track is never silently counted as unchanged: it is excluded from every
count, listed in the summary, and the run fails unless --allow-missing is given.

    python scripts/refinement_transitions.py --dir <final dir> --data-root <GraSP> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import ORDER, fuse
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores

TRACK_DIRS = {42: ["tracked", "tracked_rest"], 43: ["tracked_s43", "tracked_rest_s43"], 44: ["tracked_s44", "tracked_rest_s44"]}
ALL_DIRS = {42: "tracked_all_s42", 43: "tracked_all_s43", 44: "tracked_all_s44"}
GROUPS = ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]


def load_tracks(dirs: list[Path]) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, None]], list[str]]:
    """det arrays and (position of the centre frame in the track) per instrument index; duplicates across files are reported."""
    det: dict[int, np.ndarray] = {}
    meta: dict[int, tuple[int, None]] = {}
    dup: list[str] = []
    for d in dirs:
        for path in sorted(d.glob("*frames*.npz")):
            z = np.load(path)
            for key in z.files:
                if not key.startswith("det_"):
                    continue
                i = int(key[4:])
                if i in det:
                    dup.append(f"{d.name}/{path.name}:{i}")
                    continue
                det[i] = z[key]
                if f"center_{i}" in z.files:
                    meta[i] = (int(z[f"center_{i}"]), None)
    return det, meta, dup


def counts(correct_before: np.ndarray, correct_after: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    b, a = correct_before[mask], correct_after[mask]
    return {"wrong_to_correct": int((~b & a).sum()), "correct_to_wrong": int((b & ~a).sum()),
            "correct_to_correct": int((b & a).sum()), "wrong_to_wrong": int((~b & ~a).sum()), "n": int(mask.sum())}


def ratio(num: int, den: int):
    return None if den == 0 else num / den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/gtbox_sam/final")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--top", type=int, default=833)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    n = len(ds.instances)
    files = [ds.instances[i][0] for i in range(n)]
    cases = [f.split("/")[0] for f in files]
    stems = [int(Path(f).stem) for f in files]
    boxes = [tuple(int(v) for v in ds.instances[i][2]) for i in range(n)]
    gt_ds = np.array([int(ds.instances[i][3]) for i in range(n)])
    if len(set(zip(files, boxes))) != n:
        sys.exit("instrument identity (file, box) is not unique")

    summary: dict = {"n_instruments": n, "tau": args.tau, "top": args.top, "tie_rule": "stable sort, lower instrument index first", "seeds": {}, "checks": {}}
    rows: list[dict] = []
    for seed, dirs in TRACK_DIRS.items():
        z = np.load(args.dir / f"logits_s{seed}.npz")
        chk: dict = {}
        # identity: lengths are not enough, the stored labels must equal the dataset's label for every instrument
        if len(z["y"]) != n or not np.array_equal(z["y"], gt_ds):
            sys.exit(f"seed {seed}: logits labels do not match the dataset instruments")
        chk["labels_match_dataset"] = True
        chk["classified_all"] = bool(z["done"].all())
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        base, base_conf = mu.argmax(axis=1), mu.max(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        det, meta, dup = load_tracks([args.dir / d for d in dirs])
        chk["duplicate_track_entries"] = dup
        # identity by content: the centre frame of instrument i's track must reproduce instrument i's single-pass logits (every member), not just share its index
        ref_all = {k: z[f"det_{k}"] for k in ORDER}
        gap = {i: (float(np.abs(det[i][meta[i][0]] - np.stack([ref_all[k][i] for k in ORDER])).max()) if i in meta and 0 <= meta[i][0] < len(det[i]) else float("inf")) for i in det if 0 <= i < n}
        bad_id = [i for i, g in gap.items() if not g < 1e-3]
        chk["tracks_checked_by_centre_frame_logits"] = len(gap)
        chk["max_centre_frame_logit_gap"] = max(gap.values()) if gap else None
        chk["tracks_failing_centre_frame_check"] = bad_id[:20]
        out_of_range = [i for i in det if not 0 <= i < n]
        chk["track_indices_out_of_range"] = out_of_range
        valid = np.zeros(n, bool)
        ref_pred, ref_conf = np.full(n, -1), np.full(n, np.nan)
        for i, d in det.items():
            if 0 <= i < n and d.ndim == 3 and len(d) > 0 and np.isfinite(d).all():
                p, c = fuse(d, weights)
                ref_pred[i], ref_conf[i], valid[i] = p, c, True
        missing = np.where(~valid)[0]
        chk["missing_or_invalid_tracks"] = [int(i) for i in missing]
        if bad_id or out_of_range or dup:
            sys.exit(f"seed {seed}: track identity check failed: {chk}")
        if len(missing) and not args.allow_missing:
            sys.exit(f"seed {seed}: {len(missing)} instruments without a valid track (first: {missing[:10].tolist()}); use --allow-missing to exclude them")
        # control: the separately stored all-instrument tracks, if present, should reproduce the same refine-all labels
        alld = args.dir / ALL_DIRS[seed]
        if alld.is_dir():
            det2, _, _ = load_tracks([alld])
            same = [i for i in det2 if i in det and det[i].shape == det2[i].shape and np.allclose(det[i], det2[i], atol=1e-5)]
            chk["tracked_all_agrees_with_used_tracks"] = {"compared": len(det2), "identical": len(same)}

        y = gt_ds
        before, after = base == y, np.where(valid, ref_pred, base) == y
        after = np.where(valid, ref_pred == y, False)
        grp = np.where(~valid, "missing_track", np.where(~before & after, GROUPS[0], np.where(before & ~after, GROUPS[1], np.where(before & after, GROUPS[2], GROUPS[3]))))

        # policies, restricted to instruments with a valid track
        order_s1 = np.array([i for i in np.argsort(-s1, kind="stable") if valid[i]])
        order_mb = np.array([i for i in np.argsort(-(1 - base_conf), kind="stable") if valid[i]])
        sel: dict[str, np.ndarray] = {}
        sel["tau_1.7e-5"] = (s1 >= args.tau) & valid
        sel["s1_top833"] = np.zeros(n, bool)
        sel["s1_top833"][order_s1[: args.top]] = True
        for tag, ref in (("tau", "tau_1.7e-5"), ("top833", "s1_top833")):
            m = np.zeros(n, bool)
            m[order_mb[: int(sel[ref].sum())]] = True
            sel[f"maxbelief_n_{tag}"] = m
        sel["refine_all"] = valid.copy()

        nv = int(valid.sum())
        all_wc, all_cw = int((~before & after & valid).sum()), int((before & ~after & valid).sum())
        single_correct = int((before & valid).sum())
        pols: dict = {}
        for name, m in sel.items():
            s, u = counts(before, after, m), counts(before, after, valid & ~m)
            net = s["wrong_to_correct"] - s["correct_to_wrong"]
            acc_policy = (np.where(m, after, before) & valid).sum() / nv
            d_acc = acc_policy - single_correct / nv
            if abs(net / nv - d_acc) > 1e-12:
                sys.exit(f"seed {seed} {name}: net corrections / N ({net / nv}) != accuracy change ({d_acc})")
            pols[name] = {"selected": s, "unselected_if_refined": u, "n_selected": s["n"], "selected_fraction": s["n"] / nv,
                          "corrections_obtained": s["wrong_to_correct"], "errors_introduced": s["correct_to_wrong"], "net_corrections": net,
                          "recoverable_errors_captured": ratio(s["wrong_to_correct"], all_wc), "harmful_refinements_avoided": ratio(u["correct_to_wrong"], all_cw),
                          "instance_accuracy": float(acc_policy), "accuracy_change_vs_single": float(d_acc), "net_over_n_equals_accuracy_change": True}
        chk["net_over_n_equals_accuracy_change"] = True
        summary["seeds"][str(seed)] = {"n_valid_tracks": nv, "n_missing": int(len(missing)), "accuracy_single": single_correct / nv,
                                       "all_instruments": counts(before, after, valid), "all_wrong_to_correct": all_wc, "all_correct_to_wrong": all_cw, "policies": pols, "checks": chk}
        summary["checks"][str(seed)] = chk

        per_case: dict = defaultdict(dict)
        for c in sorted(set(cases)):
            cm = np.array([cc == c for cc in cases]) & valid
            per_case[c]["all_instruments"] = counts(before, after, cm)
            for name, m in sel.items():
                per_case[c][name] = {"selected": counts(before, after, cm & m), "unselected_if_refined": counts(before, after, cm & ~m)}
        summary["seeds"][str(seed)]["per_case"] = per_case

        s1_rank = np.full(n, -1)
        s1_rank[order_s1] = np.arange(len(order_s1))
        for i in range(n):
            rows.append({"seed": seed, "instrument": i, "case": cases[i], "frame": stems[i], "file": files[i], "box_xywh": " ".join(map(str, boxes[i])), "gt": int(y[i]),
                         "single_pred": int(base[i]), "single_conf": float(base_conf[i]), "refine_all_pred": int(ref_pred[i]) if valid[i] else "", "refine_all_conf": f"{ref_conf[i]:.9g}" if valid[i] else "",
                         "s1": float(s1[i]), "max_belief": float(base_conf[i]), "s1_rank": int(s1_rank[i]), "has_valid_track": bool(valid[i]), "transition": grp[i],
                         **{f"sel_{k}": bool(m[i]) for k, m in sel.items()}})
        print(f"seed {seed}: valid {nv}/{n}; " + ", ".join(f"{k} {v['n_selected']} sel net {v['net_corrections']:+d}" for k, v in pols.items()), flush=True)

    # mean over seeds of the headline quantities (None-safe)
    mean: dict = {}
    names = next(iter(summary["seeds"].values()))["policies"].keys()
    for name in names:
        mean[name] = {}
        for q in ("selected_fraction", "corrections_obtained", "errors_introduced", "net_corrections", "recoverable_errors_captured", "harmful_refinements_avoided", "accuracy_change_vs_single"):
            vals = [summary["seeds"][s]["policies"][name][q] for s in summary["seeds"]]
            mean[name][q] = None if any(v is None for v in vals) else float(np.mean(vals))
    summary["mean_over_seeds"] = mean
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "refinement_summary.json").write_text(json.dumps(summary, indent=1))
    with open(args.out_dir / "refinement_transitions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
