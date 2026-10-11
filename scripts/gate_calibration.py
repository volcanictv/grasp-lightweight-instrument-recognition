"""Development-calibrated operating policy for the refinement gate (follow-up evaluation; docs/reports/gate_calibration/README.md has the design).

Checkpoint sets are the cross-validation folds of the campaign (cv5_f0 .. cv5_f4; arm N classifier members, SAM2 + SAM3, fixed schedules, last epoch kept, nothing selected on held-out cases).
Inside one fold the held-out cases split by role: calibration cases are the held-out official-training cases, evaluation cases are the held-out official-test cases. No learned component of
that checkpoint set saw either. Everything here is CPU work on stored predictions: single-pass logits, SAM masks and the SAM2-large tracks (+-10 frames) that the campaign stored for the 50% of
each fold's instruments with the highest S1.

  calibrate   scores only (no ground truth, no tracks' labels): per fold and per score pick the cutoff whose refinement fraction on the calibration cases is closest to --target; write
              gate_calibration.json (thresholds, tie rule, case roles, input hashes, coverage pre-check). Commit that file before running `evaluate`.
  evaluate    read the frozen file, check it against the inputs, apply the thresholds unchanged to the evaluation cases, score single pass and gated runs, write gate_evaluation.csv / .json.

Scores: S1 (evidential epistemic score, the primary) and 1 - largest belief (comparator), both from the same four-member evidential ensemble. A cutoff selects instruments with score >= cutoff.
Tie rule: candidate cutoffs are the distinct calibration scores; the one whose selected fraction is closest to the target wins; on an exact tie in distance the larger cutoff (fewer refinements).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores

OFFICIAL_TRAIN = {"CASE001", "CASE002", "CASE003", "CASE004", "CASE007", "CASE014", "CASE015", "CASE021"}
OFFICIAL_TEST = {"CASE041", "CASE047", "CASE050", "CASE051", "CASE053"}
SCORES = ("S1", "maxbelief")
SCORE_NAMES = {"S1": "S1 (evidential epistemic score)", "maxbelief": "1 - largest belief (same ensemble)"}
FOLDS = [0, 1, 2, 3, 4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores_of(logits_path: Path) -> dict:
    z = np.load(logits_path)
    alpha = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return {"y": z["y"], "done": z["done"], "mu": mu, "base": mu.argmax(1), "conf": mu.max(1),
            "S1": variance_scores(alpha)["epistemic"].astype(np.float64), "maxbelief": (1.0 - mu.max(1)).astype(np.float64)}


def pick_cutoff(dev_scores: np.ndarray, target: float) -> tuple[float, float, int]:
    """The distinct dev score whose selected fraction (score >= cutoff) is closest to target; ties in distance go to the larger cutoff."""
    s = np.sort(dev_scores)
    cuts = np.unique(s)
    frac = (len(s) - np.searchsorted(s, cuts, side="left")) / len(s)
    dist = np.abs(frac - target)
    best = np.flatnonzero(dist <= dist.min() + 1e-12)
    j = best[-1]                      # ascending cutoffs: the last of the equally close ones is the largest
    return float(cuts[j]), float(frac[j]), int(round(frac[j] * len(s)))


def fold_context(args, k: int):
    ds = GraspRegionDataset(args.data_root, f"cv5_f{k}_test", letterbox=True)
    cases = np.array([f.split("/")[0] for f, _s, _b, _l in ds.instances])
    return ds, cases


def cmd_calibrate(args) -> None:
    out = {"design": {
        "question": "can a threshold calibrated on development cases of the same checkpoint set be transferred to its evaluation cases",
        "checkpoint_sets": "cross-validation folds cv5_f0..cv5_f4 of the campaign (job 21827306), classifier seed 42, one set per fold",
        "roles": "calibration cases = held-out official-training cases of the fold; evaluation cases = held-out official-test cases of the fold; fold 4 holds out only official-training cases, so it is calibrated but not evaluated",
        "target_refinement_fraction": args.target, "primary_score": "S1", "comparator": "1 - largest belief, same evidential ensemble",
        "cutoff_rule": "distinct calibration score whose fraction selected (score >= cutoff) is closest to the target, from scores only; exact ties in distance go to the larger cutoff",
        "fixed": "same masks, preprocessing, ensemble weights, fusion rule and SAM2-large tracking window (+-10) as the stored fold runs; no refitting; the cutoffs are not adjusted on evaluation cases",
        "ensemble_weights": CONFIGS["four"]}, "folds": {}}
    for k in FOLDS:
        d = args.cv_dir / f"cv5_f{k}" / "final"
        ds, cases = fold_context(args, k)
        sc = scores_of(d / "logits.npz")
        if len(cases) != len(sc["y"]) or not np.array_equal(sc["y"], np.array([int(ds.instances[i][3]) for i in range(len(cases))])):
            sys.exit(f"fold {k}: stored labels do not match the dataset instruments")
        dev = np.isin(cases, sorted(OFFICIAL_TRAIN)) & sc["done"]
        ev = np.isin(cases, sorted(OFFICIAL_TEST)) & sc["done"]
        tracked = load_tracked(d / "tracked")
        entry = {"held_out_cases": sorted(set(cases)), "calibration_cases": sorted(set(cases[dev])), "evaluation_cases": sorted(set(cases[ev])),
                 "n_calibration": int(dev.sum()), "n_evaluation": int(ev.sum()), "n_tracked_instruments": len(tracked),
                 "inputs_sha256": {"logits.npz": sha256(d / "logits.npz"), "masks.pkl": sha256(d / "masks.pkl")}, "scores": {}}
        for name in SCORES:
            cut, frac, n_sel = pick_cutoff(sc[name][dev], args.target)
            sel_dev = dev & (sc[name] >= cut)
            entry["scores"][name] = {"cutoff": cut, "cutoff_repr": repr(cut), "calibration_fraction": frac, "calibration_selected": int(sel_dev.sum()),
                                     "calibration_fraction_per_case": {c: float((sc[name][dev & (cases == c)] >= cut).mean()) for c in entry["calibration_cases"]}}
            if ev.any():   # scores only: how many of the instruments this cutoff will select in the evaluation cases already have a stored track
                sel_ev = ev & (sc[name] >= cut)
                missing = [int(i) for i in np.where(sel_ev)[0] if int(i) not in tracked]
                entry["scores"][name]["evaluation_coverage_precheck"] = {"would_select": int(sel_ev.sum()), "selected_without_stored_track": len(missing)}
        out["folds"][str(k)] = entry
        print(f"fold {k}: calibrate on {entry['calibration_cases']} ({entry['n_calibration']}), evaluate on {entry['evaluation_cases']} ({entry['n_evaluation']}); " +
              "; ".join(f"{n}: cutoff {entry['scores'][n]['cutoff']:.3e} dev {100 * entry['scores'][n]['calibration_fraction']:.1f}%"
                        + (f", eval selects {entry['scores'][n]['evaluation_coverage_precheck']['would_select']}, {entry['scores'][n]['evaluation_coverage_precheck']['selected_without_stored_track']} untracked" if 'evaluation_coverage_precheck' in entry['scores'][n] else "")
                        for n in SCORES), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


def cmd_evaluate(args) -> None:
    from pycocotools import mask as mask_codec
    from surgical_ai.data.mask_utils import decode_instance_mask
    from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

    frozen = json.loads(args.calibration.read_text())
    gf = json.loads((REPO_ROOT / "docs/reports/paper_gflops.json").read_text())
    g_single, g_refine = gf["single_pass_per_instrument"], gf["stages_gflops"]["refinement of one instrument"]
    s_frame, s_inst, s_ref = gf["a100_seconds"]["single_pass_per_frame"], gf["instruments_per_frame"], gf["a100_seconds"]["refinement_per_instrument"]
    configs = ("single", "S1", "maxbelief")
    rows, pooled_frames = [], defaultdict(list)
    summary: dict = {"folds": {}, "pooled": {}, "note": "refine-all was not run for these checkpoint sets (only the 50% highest-S1 instruments of each fold are tracked): null"}
    pooled_acc = defaultdict(lambda: [0, 0])
    pooled_sel = {n: [0, 0] for n in SCORES}
    for k in FOLDS:
        entry = frozen["folds"][str(k)]
        if not entry["evaluation_cases"]:
            continue
        d = args.cv_dir / f"cv5_f{k}" / "final"
        for fn, h in entry["inputs_sha256"].items():
            if sha256(d / fn) != h:
                sys.exit(f"fold {k}: {fn} differs from the frozen calibration")
        ds, cases = fold_context(args, k)
        sc = scores_of(d / "logits.npz")
        dev = np.isin(cases, entry["calibration_cases"]) & sc["done"]
        for name in SCORES:   # the frozen cutoffs must be what the stored scores give; nothing is re-chosen here
            cut, _, _ = pick_cutoff(sc[name][dev], frozen["design"]["target_refinement_fraction"])
            if cut != entry["scores"][name]["cutoff"]:
                sys.exit(f"fold {k} {name}: recomputed cutoff {cut!r} differs from the frozen {entry['scores'][name]['cutoff']!r}")
        ev = np.isin(cases, entry["evaluation_cases"]) & sc["done"]
        tracked = load_tracked(d / "tracked")
        masks = pickle.loads((d / "masks.pkl").read_bytes())
        sam = {i: mask_codec.decode(r).astype(bool) for i, r in masks.items()}
        y, base, conf = sc["y"], sc["base"], sc["conf"]
        sel = {n: ev & (sc[n] >= entry["scores"][n]["cutoff"]) for n in SCORES}
        fused = {}
        for n in SCORES:
            missing = [int(i) for i in np.where(sel[n])[0] if int(i) not in tracked]
            if missing:
                sys.exit(f"fold {k} {n}: {len(missing)} selected evaluation instruments have no stored track; not treated as unchanged (first {missing[:5]})")
        pred, pconf = {"single": base.copy()}, {"single": conf.copy()}
        for n in SCORES:
            pred[n], pconf[n] = base.copy(), conf.copy()
            for i in np.where(sel[n])[0]:
                if int(i) not in fused:
                    fused[int(i)] = fuse(tracked[int(i)], CONFIGS["four"])
                pred[n][i], pconf[n][i] = fused[int(i)]
        by_frame = defaultdict(list)
        for i in np.where(ev)[0]:
            by_frame[ds.instances[i][0]].append(int(i))
        frames = [(f, ix) for f, ix in by_frame.items() if all(i in masks for i in ix)]
        if len(frames) != len(by_frame):
            sys.exit(f"fold {k}: {len(by_frame) - len(frames)} evaluation frames lack masks")
        res = {c: [] for c in configs}
        frame_case = []
        for f, ix in frames:
            shape = tuple(ds.instances[ix[0]][1]["size"])
            gt = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(ds.instances[i][3]) + 1, 0.0) for i in ix])
            frame_case.append(f.split("/")[0])
            for c in configs:
                res[c].append(frame_class_ious(paint(shape, [(sam[i], int(pred[c][i]) + 1, float(pconf[c][i])) for i in ix]), gt))
        fold_sum = {"evaluation_cases": entry["evaluation_cases"], "n_instruments": int(ev.sum()), "n_frames": len(frames), "configs": {}, "per_case": {}}
        for c in configs:
            a = aggregate(res[c])
            acc = float((pred[c][ev] == y[ev]).mean())
            fr = 0.0 if c == "single" else float(sel[c].sum() / ev.sum())
            fold_sum["configs"][c] = {"mIoU": 100 * float(a["mIoU"]), "IoU": 100 * float(a["IoU"]), "mcIoU": 100 * float(a["mcIoU"]), "instance_accuracy": acc,
                                      "calibration_fraction": None if c == "single" else entry["scores"][c]["calibration_fraction"], "evaluation_fraction": fr,
                                      "gflops_per_instrument": g_single + fr * g_refine, "a100_s_per_instrument": s_frame / s_inst + fr * s_ref,
                                      "saving_vs_refine_all_gflops": 1 - (g_single + fr * g_refine) / (g_single + g_refine), "refine_all": None}
            pooled_frames[c] += res[c]
            pooled_acc[c][0] += int((pred[c][ev] == y[ev]).sum()); pooled_acc[c][1] += int(ev.sum())
            if c != "single":
                pooled_sel[c][0] += int(sel[c].sum()); pooled_sel[c][1] += int(ev.sum())
            for case in entry["evaluation_cases"]:
                m = [j for j, cc in enumerate(frame_case) if cc == case]
                cm = (cases == case) & ev
                a = aggregate([res[c][j] for j in m])
                fold_sum["per_case"].setdefault(case, {})[c] = {"mIoU": 100 * float(a["mIoU"]), "IoU": 100 * float(a["IoU"]), "mcIoU": 100 * float(a["mcIoU"]), "instance_accuracy": float((pred[c][cm] == y[cm]).mean()),
                                                                "n_instruments": int(cm.sum()), "evaluation_fraction": None if c == "single" else float(sel[c][cm].sum() / cm.sum())}
                r = fold_sum["per_case"][case][c]
                rows.append({"fold": k, "case": case, "config": c, "scope": "case", "n_instruments": r["n_instruments"], "calibration_fraction": "" if c == "single" else f"{entry['scores'][c]['calibration_fraction']:.4f}",
                             "evaluation_fraction": "" if r["evaluation_fraction"] is None else f"{r['evaluation_fraction']:.4f}", "mIoU": f"{r['mIoU']:.2f}", "IoU": f"{r['IoU']:.2f}", "mcIoU": f"{r['mcIoU']:.2f}", "instance_accuracy": f"{r['instance_accuracy']:.4f}"})
            q = fold_sum["configs"][c]
            rows.append({"fold": k, "case": "+".join(entry["evaluation_cases"]), "config": c, "scope": "fold", "n_instruments": int(ev.sum()), "calibration_fraction": "" if q["calibration_fraction"] is None else f"{q['calibration_fraction']:.4f}",
                         "evaluation_fraction": f"{q['evaluation_fraction']:.4f}", "mIoU": f"{q['mIoU']:.2f}", "IoU": f"{q['IoU']:.2f}", "mcIoU": f"{q['mcIoU']:.2f}", "instance_accuracy": f"{q['instance_accuracy']:.4f}"})
        summary["folds"][str(k)] = fold_sum
        print(f"fold {k}: " + ", ".join(f"{c} mIoU {fold_sum['configs'][c]['mIoU']:.2f} (refined {100 * fold_sum['configs'][c]['evaluation_fraction']:.1f}%)" for c in configs), flush=True)
    n_all = pooled_acc["single"][1]
    for c in configs:
        a = aggregate(pooled_frames[c])
        fr = 0.0 if c == "single" else pooled_sel[c][0] / pooled_sel[c][1]
        summary["pooled"][c] = {"mIoU": 100 * float(a["mIoU"]), "IoU": 100 * float(a["IoU"]), "mcIoU": 100 * float(a["mcIoU"]), "instance_accuracy": pooled_acc[c][0] / n_all, "evaluation_fraction": fr,
                                "gflops_per_instrument": g_single + fr * g_refine, "saving_vs_refine_all_gflops": 1 - (g_single + fr * g_refine) / (g_single + g_refine), "n_instruments": n_all, "refine_all": None}
        rows.append({"fold": "pooled", "case": "all evaluation cases", "config": c, "scope": "pooled", "n_instruments": n_all, "calibration_fraction": "", "evaluation_fraction": f"{fr:.4f}",
                     "mIoU": f"{summary['pooled'][c]['mIoU']:.2f}", "IoU": f"{summary['pooled'][c]['IoU']:.2f}", "mcIoU": f"{summary['pooled'][c]['mcIoU']:.2f}", "instance_accuracy": f"{summary['pooled'][c]['instance_accuracy']:.4f}"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_evaluation.json").write_text(json.dumps(summary, indent=1))
    with open(args.out_dir / "gate_evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("pooled: " + ", ".join(f"{c} mIoU {summary['pooled'][c]['mIoU']:.2f} IoU {summary['pooled'][c]['IoU']:.2f} mcIoU {summary['pooled'][c]['mcIoU']:.2f} refined {100 * summary['pooled'][c]['evaluation_fraction']:.1f}%" for c in configs))
    print("wrote", args.out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calibrate", "evaluate"])
    ap.add_argument("--cv-dir", type=Path, required=True, help="folder with cv5_f<k>/final/{logits.npz,masks.pkl,tracked/}")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=Path("gate_calibration.json"))
    ap.add_argument("--calibration", type=Path, default=Path("gate_calibration.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    cmd_calibrate(args) if args.cmd == "calibrate" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
