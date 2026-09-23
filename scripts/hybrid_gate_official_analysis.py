"""Official-test analysis of the evidential-epistemic hybrid-gate proposal
(docs/DECISIONS.md 2026-09-23), stopping short of actually tracking the
newly-flagged instances: the fold1 calibration (calibrate_hybrid_gate_tau.py)
found that catching every fold1 unanimous error this way requires flagging
59.2% of instances, far past the pre-registered "report rather than silently
proceed" line, so this script reports what the pre-registered tau (and a few
illustrative smaller budgets) would cost and catch on official test, as
projections from the score alone, not as tracked outcomes.

Usage:
    python scripts/hybrid_gate_official_analysis.py \
        --ce-cache experiments/mc_logits_cache_deepdropout.npz \
        --evidential-cache experiments_edl/extract/E_grasp_official_s42.npz \
        --tau-calibration docs/reports/hybrid_gate/tau_calibration.json \
        --out docs/reports/hybrid_gate/official_test_analysis.json
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
from sklearn.metrics import f1_score, roc_auc_score

from build_tracking_sample_b import plurality, vote_share
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

LABELS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
GATE = 0.09
SECONDS_PER_INSTANCE = 24.0
ILLUSTRATIVE_BUDGETS = (0.01, 0.02, 0.05, 0.10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ce-cache", type=Path, required=True)
    ap.add_argument("--evidential-cache", type=Path, required=True)
    ap.add_argument("--tau-calibration", type=Path, required=True)
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
    unanimous = err & (u == 0)
    already_flagged = u >= GATE
    print(f"official test sanity check: n={len(y)}, ensemble accuracy={float((base == y).mean()):.4f}, "
          f"macro-F1={f1_score(y, base, average='macro'):.4f} (shipped: 0.9266 / 0.8898)")
    print(f"unanimous errors on official test: {int(unanimous.sum())}, of which "
          f"{int((unanimous & already_flagged).sum())} already flagged by the 9% gate")

    ev = np.load(args.evidential_cache)
    assert (ev["y"] == y).all(), "instance order mismatch between shipped CE cache and evidential official-test cache"
    alpha_m = [alpha_from_logits(ev["det_" + m]) for m in LABELS]
    alpha = sum(w * a for w, a in zip(WEIGHTS, alpha_m))
    s1 = variance_scores(alpha)["epistemic"]

    auroc_s1 = float(roc_auc_score(err, s1))
    auroc_b1 = float(roc_auc_score(err, u))
    print(f"official-test error-detection AUROC: S1 epistemic {auroc_s1:.4f}, B1 MC-vote disagreement {auroc_b1:.4f} "
          f"(fold1 comparison was 0.927 vs 0.912)")

    target = unanimous & ~already_flagged
    candidate = ~already_flagged
    rank = s1[target].argsort()
    percentile = 100.0 * (s1[:, None] <= s1[target][None, :]).mean(axis=0)
    print(f"{int(target.sum())} unanimous errors not caught by the 9% gate on official test; their S1 percentile "
          f"rank among all {len(y)} instances: {sorted(round(p, 1) for p in percentile)}")

    tau = json.loads(args.tau_calibration.read_text())["chosen_tau"]
    add_flag_full = candidate & (s1 >= tau)
    full = {
        "tau": tau, "additional_flagged": int(add_flag_full.sum()),
        "additional_flagged_share": float(add_flag_full.sum() / len(y)),
        "additional_gpu_hours": float(add_flag_full.sum() * SECONDS_PER_INSTANCE / 3600),
        "target_unanimous_caught": int((target & add_flag_full).sum()),
        "target_unanimous_total": int(target.sum()),
        "note": "projection from the score only; not tracked (cost stop condition triggered on fold1 calibration)",
    }
    print(json.dumps({"pre_registered_tau_projection": full}, indent=1))

    illustrative = []
    for b in ILLUSTRATIVE_BUDGETS:
        k = int(round(b * len(y)))
        if k == 0 or candidate.sum() == 0:
            continue
        thr = np.sort(s1[candidate])[::-1][min(k, candidate.sum()) - 1]
        add_flag = candidate & (s1 >= thr)
        illustrative.append({
            "budget": b, "additional_flagged": int(add_flag.sum()),
            "additional_gpu_hours": float(add_flag.sum() * SECONDS_PER_INSTANCE / 3600),
            "target_unanimous_caught": int((target & add_flag).sum()), "target_unanimous_total": int(target.sum()),
            "note": "projection from the score only; not tracked, so the eventual accuracy effect is unverified",
        })
    print(json.dumps({"illustrative_smaller_budgets": illustrative}, indent=1))

    out = {
        "official_test_sanity": {"n": int(len(y)), "accuracy": float((base == y).mean()),
                                 "macro_f1": float(f1_score(y, base, average="macro"))},
        "unanimous_errors_total": int(unanimous.sum()),
        "unanimous_errors_missed_by_gate": int(target.sum()),
        "unanimous_error_s1_percentiles": [round(float(p), 2) for p in sorted(percentile)],
        "auroc_s1_official_test": auroc_s1, "auroc_b1_official_test": auroc_b1,
        "pre_registered_tau_projection": full, "illustrative_smaller_budgets": illustrative,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
