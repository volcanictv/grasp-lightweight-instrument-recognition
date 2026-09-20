"""Which tracking-gate threshold lets the softmax-free pipeline match or beat the
softmax pipeline? (docs/DECISIONS.md 2026-09-20). Fixed before the extension run.

- gate: track an instance when its vote disagreement u = 1 - top vote share is
  >= t. u moves in steps of 0.01, so t is scanned over 0.03, 0.04, ..., 0.20 with
  real SAM2 outcomes for every point (the softmax-free run tracked u >= 0.20, the
  extension run tracks 0.03 <= u < 0.20).
- combine rule: V2 uncertainty-weighted vote (pre-registered primary).
- targets: the softmax pipeline on the same retrained ensemble, rebuilt from the
  saved files. T1 = softmax-confidence gate (< 0.80) + average-softmax combine
  (the earlier best, 0.9612 / 0.9340). T2 = the earlier MC gate + average-softmax
  combine (0.9595 / 0.9325).
- selection rule: the highest threshold (cheapest tracking) whose accuracy AND
  macro-F1 are both >= the target; if none reaches it, the threshold with the
  smallest worst-case shortfall (ties to the higher threshold).
- This picks a threshold using the official test set, so its accuracy is
  optimistic. A leave-one-case-out check on the coarse grid {0.20, 0.15, 0.10,
  0.05, 0.03} shows what a threshold chosen without the held-out case achieves.

Usage:
    python scripts/evaluate_gate_sweep.py [--allow-partial]
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
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms

FINE = [round(t, 2) for t in np.arange(0.20, 0.0299, -0.01)]
COARSE = [0.20, 0.15, 0.10, 0.05, 0.03]
SECONDS_PER_INSTANCE = 24.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dirs", type=Path, nargs="+", default=[REPO_ROOT / "docs" / "reports" / "tracking_softmax_free",
                                                                  REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep"])
    parser.add_argument("--softmax-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep" / "results.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)
    N = len(y)

    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    ds = GraspRegionDataset(data_root, "test", transform=build_transforms(224, train=False))
    names = ds.class_names_ordered()
    name_to_idx = {n: i for i, n in enumerate(names)}
    case_of = np.array([fn.split("/")[0] for fn, _s, _b, _l in ds.instances])
    labels = list(range(len(names)))

    def macro(p: np.ndarray) -> float:
        return float(f1_score(y, p, average="macro", labels=labels))

    p_soft = sum(w * torch.softmax(torch.from_numpy(cache[f"det_{m['label']}"]), dim=1).numpy() for w, m in zip(weights, members))
    soft_base = p_soft.argmax(axis=1)
    gates = json.loads((args.softmax_dir / "gates.json").read_text())
    soft_tracked = {}
    for shard in (0, 1):
        for r in json.loads((args.softmax_dir / f"tracked_shard{shard}.json").read_text())["instances"]:
            soft_tracked[r["index"]] = name_to_idx[r["avg_softmax_pred"]]

    def softmax_pipeline(flag_idx: list[int]) -> np.ndarray:
        pred = soft_base.copy()
        for i in flag_idx:
            pred[i] = soft_tracked[i]
        return pred

    refs = {"T1 softmax confidence gate + avg-softmax combine": softmax_pipeline(gates["conf_flagged_indices"]),
            "T2 MC gate (softmax-based) + avg-softmax combine": softmax_pipeline(gates["mc_flagged_indices"])}
    targets = {k: {"accuracy": float((p == y).mean()), "macro_f1": macro(p)} for k, p in refs.items()}
    print("softmax pipeline targets (same retrained ensemble):")
    for k, v in targets.items():
        print(f"  {k}: accuracy {v['accuracy']:.4f}, macro-F1 {v['macro_f1']:.4f}")

    tracked: dict[int, dict] = {}
    for d in args.dirs:
        for shard in (0, 1):
            path = d / f"frames_shard{shard}.npz"
            if path.exists():
                data = np.load(path)
                for key in data.files:
                    if key.startswith("det_"):
                        i = int(key[4:])
                        tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    need = set(np.where(u >= min(FINE))[0].tolist())
    missing = need - set(tracked)
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} instances with u >= {min(FINE)} lack tracking data (e.g. {sorted(missing)[:5]})")
    if missing:
        print(f"\nWARNING dry run: {len(missing)} of {len(need)} instances lack tracking data")

    prim = RULES[1]

    def final(t: float) -> np.ndarray:
        pred = base.copy()
        for i in np.where(u >= t)[0]:
            if i in tracked:
                pred[i] = tracked[i][prim]
        return pred

    preds = {t: final(t) for t in FINE}
    print(f"\nvote-based pipeline (rule: {prim}); ensemble alone accuracy {(base == y).mean():.4f}, macro-F1 {macro(base):.4f}")
    print(f"{'threshold':>9}{'tracked':>9}{'%':>7}{'accuracy':>10}{'macro-F1':>10}{'fixed':>7}{'broken':>8}{'~GPU-h':>8}")
    rows = {}
    for t in FINE:
        p, flag = preds[t], u >= t
        fixed = int(((base != y) & (p == y) & flag).sum())
        broken = int(((base == y) & (p != y) & flag).sum())
        rows[str(t)] = {"tracked": int(flag.sum()), "accuracy": float((p == y).mean()), "macro_f1": macro(p), "fixed": fixed, "broken": broken,
                        "gpu_hours": float(flag.sum() * SECONDS_PER_INSTANCE / 3600)}
        r = rows[str(t)]
        print(f"{t:>9.2f}{r['tracked']:>9}{100 * r['tracked'] / N:>6.1f}%{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}{fixed:>7}{broken:>8}{r['gpu_hours']:>8.1f}")

    R: dict = {"targets": targets, "rows": rows, "selection": {}}
    for k, tg in targets.items():
        hit = [t for t in FINE if rows[str(t)]["accuracy"] >= tg["accuracy"] and rows[str(t)]["macro_f1"] >= tg["macro_f1"]]
        if hit:
            t_sel, note = max(hit), "reaches both targets"
        else:
            t_sel = max(FINE, key=lambda t: (-max(tg["accuracy"] - rows[str(t)]["accuracy"], tg["macro_f1"] - rows[str(t)]["macro_f1"]), t))
            note = "no threshold reaches both targets; smallest worst-case shortfall"
        p_sel, p_ref = preds[t_sel], refs[k]
        ok_s, ok_r = p_sel == y, p_ref == y
        a, b = int((ok_s & ~ok_r).sum()), int((ok_r & ~ok_s).sum())
        pv = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        r = rows[str(t_sel)]
        R["selection"][k] = {"threshold": t_sel, "note": note, **r, "only_vote_pipeline": a, "only_softmax_pipeline": b, "mcnemar_p": float(pv)}
        print(f"\n{k}\n  selected threshold {t_sel:.2f} ({note}): tracked {r['tracked']} ({100 * r['tracked'] / N:.1f}%), accuracy {r['accuracy']:.4f} "
              f"(target {tg['accuracy']:.4f}), macro-F1 {r['macro_f1']:.4f} (target {tg['macro_f1']:.4f})")
        print(f"  paired vs the softmax pipeline: correct only under vote pipeline {a}, only under softmax pipeline {b}, exact McNemar p = {pv:.3f}")

    correct = {t: preds[t] == y for t in COARSE}
    chosen, total = {}, 0
    for c in sorted(set(case_of)):
        te = case_of == c
        best = max(COARSE, key=lambda t: (int(correct[t][~te].sum()), t))
        chosen[c] = best
        total += int(correct[best][te].sum())
    R["loco"] = {"grid": COARSE, "chosen": chosen, "accuracy": total / N}
    print(f"\nleave-one-case-out threshold choice on {COARSE}: chosen per held-out case {chosen}; pooled accuracy {total / N:.4f} "
          f"(fixed 0.20: {correct[0.20].mean():.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(R, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
