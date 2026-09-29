"""Gates and threshold-free signal quality for the EndoVis-trained evidential
ensemble against the EndoVis-trained MC pipeline (docs/DECISIONS.md 2026-09-29).

Reads the evidential extract (scripts/endovis_evidential_stage1.py) and the MC
stage 1 votes.npz for the same dataset (same val instances, asserted), and reports:
base accuracy/macro-F1, gate (A) the GraSP-calibrated evidential tau applied
unchanged, gate (B) budget-matched to the MC pipeline's flagged count, error
coverage of each gate, and AUROC of MC vote disagreement, softmax confidence and
evidential S1 each against its own arm's errors (per-sequence spread when
sequence ids exist). Writes the union of the (A) and (B) flagged indices for
SAM2 tracking.

Usage:
    python scripts/endovis_evidential_analyze.py --dataset 2018 \
        --extract experiments_endovis/evid_stage1_2018/extract.npz \
        --mc-dir experiments_endovis/stage1_2018 --out experiments_endovis/evid_stage1_2018/analysis.json
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

from evidential_analyze import bootstrap_auroc
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}


def coverage(flag: np.ndarray, err: np.ndarray) -> dict:
    return {"n_flagged": int(flag.sum()), "share_flagged": float(flag.mean()),
            "errors_caught": int((flag & err).sum()), "share_of_errors_caught": float((flag & err).sum() / err.sum()),
            "error_rate_inside_flagged": float((flag & err).sum() / max(flag.sum(), 1)),
            "error_rate_outside_flagged": float((~flag & err).sum() / max((~flag).sum(), 1))}


def auroc_block(score: np.ndarray, err: np.ndarray, seq: np.ndarray) -> dict:
    out = {"auroc": float(roc_auc_score(err, score)), "auroc_ci95": bootstrap_auroc(score, err)}
    per = {}
    for s in sorted(set(seq.tolist())):
        m = seq == s
        if m.sum() >= 50 and 0 < err[m].sum() < m.sum():
            per[s] = float(roc_auc_score(err[m], score[m]))
    if len(per) > 1:
        out["auroc_per_sequence"] = per
        out["auroc_per_sequence_min_mean_max"] = [min(per.values()), float(np.mean(list(per.values()))), max(per.values())]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["2017", "2018"], required=True)
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--mc-dir", type=Path, required=True)
    ap.add_argument("--tau-file", type=Path, default=REPO_ROOT / "docs" / "reports" / "evidential_pipeline_switch" / "results.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    e = np.load(args.extract, allow_pickle=True)
    votes = np.load(args.mc_dir / "votes.npz")
    mc_result = json.loads((args.mc_dir / "result.json").read_text())
    y = e["y"]
    assert (votes["y"] == y).all(), "val instance order differs between the evidential and MC runs"
    seq = e["seq"] if "seq" in e.files else np.array(["all"] * len(y))

    alpha = sum(w * alpha_from_logits(e[f"det_{k}"]) for k, w in WEIGHTS.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    pred_e = mu.argmax(axis=1)
    err_e = pred_e != y
    s1 = variance_scores(alpha)["epistemic"]
    present = sorted(set(y.tolist()))

    pred_c = votes["pred"]
    err_c = pred_c != y
    u = votes["u"]
    soft = 1.0 - votes["p_mean"].max(axis=1)

    tau = json.loads(args.tau_file.read_text())["tau"]
    flag_a = s1 >= tau
    k = int(mc_result["n_flagged"])
    order = np.argsort(-s1, kind="stable")
    flag_b = np.zeros(len(y), dtype=bool)
    flag_b[order[:k]] = True
    flag_mc = u >= mc_result["gate"]

    out = {
        "dataset": args.dataset, "n": int(len(y)), "tau": tau, "budget_matched_k": k,
        "evidential": {"accuracy": float((~err_e).mean()),
                       "macro_f1_present": float(f1_score(y, pred_e, labels=present, average="macro", zero_division=0)),
                       "errors": int(err_e.sum())},
        "mc_pipeline_ensemble_alone": {"accuracy": float((~err_c).mean()),
                                       "macro_f1_present": float(f1_score(y, pred_c, labels=present, average="macro", zero_division=0)),
                                       "errors": int(err_c.sum())},
        "gate_A_fixed_tau": coverage(flag_a, err_e),
        "gate_B_budget_matched": coverage(flag_b, err_e),
        "mc_gate_9pct_for_comparison": coverage(flag_mc, err_c),
        "auroc": {"mc_vote_disagreement": auroc_block(u, err_c, seq),
                  "softmax_confidence_of_mc_arm": auroc_block(soft, err_c, seq),
                  "evidential_S1": auroc_block(s1, err_e, seq)},
        "overlap_A_vs_MC_flagged": int((flag_a & flag_mc).sum()),
        "union_A_B_size": int((flag_a | flag_b).sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    (args.out.parent / "flagged_union.json").write_text(json.dumps({
        "indices": [int(i) for i in np.where(flag_a | flag_b)[0]],
        "gate_A": [int(i) for i in np.where(flag_a)[0]], "gate_B": [int(i) for i in np.where(flag_b)[0]]}))
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "auroc"}, indent=1))
    for name, v in out["auroc"].items():
        print(f"AUROC {name:<32} {v['auroc']:.3f}  ci95 [{v['auroc_ci95'][0]:.3f}, {v['auroc_ci95'][1]:.3f}]"
              + (f"  per-sequence min/mean/max {[round(x, 3) for x in v['auroc_per_sequence_min_mean_max']]}"
                 if "auroc_per_sequence_min_mean_max" in v else ""))


if __name__ == "__main__":
    main()
