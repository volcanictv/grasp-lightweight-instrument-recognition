"""Post-hoc extras for the softmax-free pipeline (NOT part of the pre-registered
analysis in evaluate_softmax_free_combine.py; docs/DECISIONS.md 2026-09-19).

1. Uncertainty quality re-measured against the vote-plurality prediction's own
   errors (the report's earlier AUROCs were against the softmax ensemble's
   errors): top-vote-share uncertainty, between-member vote disagreement, and
   the softmax-confidence baseline (against its own errors), plus error recall
   at fixed review budgets.
2. Gate-passed errors: how many are unanimous, and how tracking (primary rule)
   fares on unanimous vs non-unanimous ones.
3. After tracking: recall and false-flag cost of "review if track-level
   uncertainty >= 0.20" among the tracked instances.

Usage:
    python scripts/analyze_vote_pipeline_extras.py
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
import torch
import yaml
from sklearn.metrics import roc_auc_score

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    k = budget * len(score)
    t = np.sort(score)[::-1][int(np.ceil(k)) - 1]
    above, tie = score > t, score == t
    frac = (k - above.sum()) / tie.sum()
    return float((y[above].sum() + frac * y[tie].sum()) / y.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_softmax_free")
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
    pred = plurality(v_mc, v_det)
    wrong = (pred != y).astype(int)
    u = np.round(1 - v_mc.max(axis=1), 9)
    u_between = np.round(1 - v_det.max(axis=1), 9)
    p_soft = sum(w * torch.softmax(torch.from_numpy(cache[f"det_{m['label']}"]), dim=1).numpy() for w, m in zip(weights, members))
    wrong_soft = (p_soft.argmax(axis=1) != y).astype(int)
    conf_unc = 1 - p_soft.max(axis=1)

    out: dict = {}
    print(f"(1) vote-plurality prediction: {int(wrong.sum())} errors; softmax ensemble (baseline only): {int(wrong_soft.sum())} errors")
    out["auroc"] = {"mc_vote_uncertainty_vs_vote_errors": float(roc_auc_score(wrong, u)),
                    "between_member_vote_disagreement_vs_vote_errors": float(roc_auc_score(wrong, u_between)),
                    "softmax_confidence_vs_softmax_errors": float(roc_auc_score(wrong_soft, conf_unc))}
    for k, v in out["auroc"].items():
        print(f"    AUROC {k}: {v:.4f}")
    out["recall_at_budget"] = {}
    for b in (0.10, 0.20, 0.30):
        rv, rc = recall_at_budget(wrong, u, b), recall_at_budget(wrong_soft, conf_unc, b)
        out["recall_at_budget"][str(b)] = {"mc_vote": rv, "softmax_confidence": rc}
        print(f"    recall of errors at {b:.0%} flagged: MC vote {rv:.1%}  vs  confidence {rc:.1%}")

    sets = json.loads((args.dir / "sets.json").read_text())
    tracked = {}
    for shard in (0, 1):
        data = np.load(args.dir / f"frames_shard{shard}.npz")
        for key in data.files:
            if key.startswith("det_"):
                i = int(key[4:])
                tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    prim = RULES[1]

    pw = np.array(sets["passed_wrong"])
    unan = pw[u[pw] == 0]
    non = pw[u[pw] > 0]
    res_u = int(sum(tracked[i][prim] == y[i] for i in unan))
    res_n = int(sum(tracked[i][prim] == y[i] for i in non))
    print(f"\n(2) gate-passed errors: {len(pw)}; unanimous (u = 0): {len(unan)}, rescued by tracking {res_u}; "
          f"not unanimous (0 < u < 0.20): {len(non)}, rescued {res_n}")
    out["passed_errors"] = {"n": int(len(pw)), "unanimous": int(len(unan)), "unanimous_rescued": res_u,
                            "non_unanimous": int(len(non)), "non_unanimous_rescued": res_n}

    fl = [i for i in sets["flagged"]]
    u_track = np.array([tracked[i]["track_uncertainty"] for i in fl])
    wrong_after = np.array([tracked[i][prim] != y[i] for i in fl])
    out["post_tracking_flag"] = {}
    for t in (0.20, 0.30):
        caught, false_flag = int((u_track[wrong_after] >= t).sum()), int((u_track[~wrong_after] >= t).sum())
        out["post_tracking_flag"][str(t)] = {"errors_caught": caught, "errors": int(wrong_after.sum()),
                                             "correct_flagged": false_flag, "correct": int((~wrong_after).sum())}
        print(f"\n(3) 'review if track uncertainty >= {t:.2f}' among the {len(fl)} tracked: catches {caught}/{int(wrong_after.sum())} remaining errors, "
              f"also flags {false_flag}/{int((~wrong_after).sum())} correct ones ({false_flag / (~wrong_after).sum():.0%})")

    (args.dir / "extras.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.dir / 'extras.json'}")


if __name__ == "__main__":
    main()
