"""Calibrates the hybrid-gate threshold tau on fold1 held-out cases
(docs/DECISIONS.md 2026-09-23): the strictest (numerically largest)
evidential epistemic threshold that still catches every fold1 unanimous
error (all 80 MC-dropout votes agree on the wrong class, among instances
the 9%-equivalent MC-vote gate itself does not already flag) -- i.e. the
one that flags the fewest additional instances while being recall-complete
on the known blind spot, not a threshold picked to hit a fixed budget.
Fixed before application to the official test set; not retuned there.

Usage:
    python scripts/calibrate_hybrid_gate_tau.py \
        --ce-cache experiments/mc_logits_cache_fold1.npz \
        --evidential-cache experiments_edl/extract/E_grasp_fold1_s42.npz \
        --out docs/reports/hybrid_gate/tau_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import f1_score

from build_tracking_sample_b import plurality, vote_share
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

LABELS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
GATE = 0.09
SECONDS_PER_INSTANCE = 24.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ce-cache", type=Path, required=True)
    ap.add_argument("--evidential-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ce = np.load(args.ce_cache)
    y = ce["y_true"]
    n_classes = ce["det_" + LABELS[0]].shape[1]
    v_mc = sum(w * vote_share(ce["mc_" + m], n_classes) for w, m in zip(WEIGHTS, LABELS))
    v_det = sum(w * (ce["det_" + m].argmax(1)[:, None] == np.arange(n_classes)) for w, m in zip(WEIGHTS, LABELS))
    base = plurality(v_mc, v_det)
    u = np.round(1.0 - v_mc.max(axis=1), 9)
    err = base != y
    unanimous = err & (u == 0)  # strict: all 80 pooled MC votes agree, and it's wrong
    already_flagged = u >= GATE
    print(f"fold1: n={len(y)}, ensemble errors={int(err.sum())}, unanimous errors={int(unanimous.sum())}, "
          f"already flagged by 9% gate={int(already_flagged.sum())}")

    ev = np.load(args.evidential_cache)
    assert (ev["y"] == y).all(), "instance order mismatch between CE and evidential fold1 caches"
    alpha_m = [alpha_from_logits(ev["det_" + m]) for m in LABELS]
    alpha = sum(w * a for w, a in zip(WEIGHTS, alpha_m))
    s1 = variance_scores(alpha)["epistemic"]

    # unanimous errors the 9% gate does NOT already catch (the actual blind spot to fix)
    target = unanimous & ~already_flagged
    print(f"unanimous errors already caught by the 9% gate: {int((unanimous & already_flagged).sum())}")
    print(f"unanimous errors the 9% gate misses (target for tau): {int(target.sum())}")

    candidate = ~already_flagged  # tau only ever adds instances the 9% gate doesn't already flag
    taus = np.unique(s1[candidate])[::-1]  # descending: sweep from strictest (fewest flagged) to loosest
    rows = []
    for t in taus:
        add_flag = candidate & (s1 >= t)
        rows.append({
            "tau": float(t), "additional_flagged": int(add_flag.sum()),
            "additional_flagged_share": float(add_flag.sum() / len(y)),
            "additional_gpu_hours": float(add_flag.sum() * SECONDS_PER_INSTANCE / 3600),
            "target_unanimous_caught": int((target & add_flag).sum()),
            "target_unanimous_total": int(target.sum()),
        })
        if target.sum() and (target & add_flag).sum() == target.sum():
            break  # smallest tau (this sweep direction) that catches every target unanimous error
    chosen = rows[-1]
    print(json.dumps({"n_rows_until_full_recall": len(rows), "chosen": chosen}, indent=1))

    out = {
        "gate": GATE, "n": int(len(y)), "unanimous_errors_total": int(unanimous.sum()),
        "unanimous_errors_missed_by_gate": int(target.sum()), "sweep": rows, "chosen_tau": chosen["tau"],
        "chosen_row": chosen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
