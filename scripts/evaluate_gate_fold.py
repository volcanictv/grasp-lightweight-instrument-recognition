"""Gate confirmation on held-out cases (docs/DECISIONS.md 2026-09-20, design fixed
before any outcome). Scores the vote-based pipeline on the fold1 held-out cases
for every threshold t in 0.03..0.20 with real SAM2 outcomes, applies the
pre-registered selection rule, and reports the shipped pipeline's official-test
result at the fold-selected threshold next to 9%.

Selection rule: maximise final-pipeline accuracy on fold1; among thresholds within
0.0010 of that maximum, take the highest (fewest tracked).

Usage:
    python scripts/evaluate_gate_fold.py --tracking-dir experiments/gate_confirm_fold1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml
from sklearn.metrics import f1_score

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions

FINE = [round(t, 2) for t in np.arange(0.20, 0.0299, -0.01)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout_fold1.yaml")
    ap.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_fold1.npz")
    ap.add_argument("--tracking-dir", type=Path, required=True)
    ap.add_argument("--test-sweep", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep" / "results.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "gate_confirmation_fold1.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

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
    labels = list(range(n_classes))

    tracked = {}
    for path in sorted(args.tracking_dir.glob("frames_shard*.npz")):
        data = np.load(path)
        for key in data.files:
            if key.startswith("det_"):
                i = int(key[4:])
                tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    need = set(np.where(u >= min(FINE))[0].tolist())
    missing = need - set(tracked)
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} of {len(need)} instances with u >= {min(FINE)} lack tracking data")
    prim = RULES[1]

    def final(t: float) -> np.ndarray:
        pred = base.copy()
        for i in np.where(u >= t)[0]:
            if i in tracked:
                pred[i] = tracked[i][prim]
        return pred

    rows = {}
    print(f"held-out fold1: {N} instances, ensemble alone accuracy {(base == y).mean():.4f}, "
          f"macro-F1 {f1_score(y, base, average='macro', labels=labels):.4f}")
    print(f"{'t':>5}{'tracked':>9}{'%':>7}{'accuracy':>10}{'macro-F1':>10}{'fixed':>7}{'broken':>8}")
    for t in FINE:
        p, flag = final(t), u >= t
        rows[str(t)] = {"tracked": int(flag.sum()), "accuracy": float((p == y).mean()),
                        "macro_f1": float(f1_score(y, p, average="macro", labels=labels)),
                        "fixed": int(((base != y) & (p == y) & flag).sum()), "broken": int(((base == y) & (p != y) & flag).sum())}
        r = rows[str(t)]
        print(f"{t:>5.2f}{r['tracked']:>9}{100 * r['tracked'] / N:>6.1f}%{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}{r['fixed']:>7}{r['broken']:>8}")
    best = max(r["accuracy"] for r in rows.values())
    t_sel = max(t for t in FINE if rows[str(t)]["accuracy"] >= best - 0.0010)
    R = {"n": N, "ensemble_alone_accuracy": float((base == y).mean()), "rows": rows,
         "selected_threshold": t_sel, "max_accuracy": best}
    print(f"\nselected threshold on fold1: {t_sel:.2f} (max accuracy {best:.4f}, rule: highest t within 0.0010 of the max)")
    if args.test_sweep.exists():
        test_rows = json.loads(args.test_sweep.read_text())["rows"]
        for label, t in (("fold1-selected", t_sel), ("9% (test-selected)", 0.09)):
            tr = test_rows.get(str(t))
            if tr:
                R[f"official_test_at_{label}"] = tr
                print(f"official test at {label} t={t:.2f}: tracked {tr['tracked']}, accuracy {tr['accuracy']:.4f}, macro-F1 {tr['macro_f1']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(R, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
