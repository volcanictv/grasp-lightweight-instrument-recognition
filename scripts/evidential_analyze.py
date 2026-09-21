"""Analysis for the evidential validation (docs/DECISIONS.md 2026-09-21): error-detection
AUROC, error recall at review budgets, unanimous-error coverage, per-case AUROC,
bootstrap intervals, ensemble accuracy/macro-F1 of both arms, the fixed-threshold
stability test across folds and EndoVis, and the stage-B go rule.

Usage:
    python scripts/evidential_analyze.py --extract-dir DIR --out docs/reports/evidential/results.json
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
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score

from build_tracking_sample_b import plurality, vote_share
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

LABELS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
BUDGETS = (0.10, 0.20, 0.30)
E_SIGNALS = ["S1_epistemic", "S2_aleatoric", "S3_total", "S4_evidence_only", "S5_member_disagreement", "S6_rank_avg_S4_B1"]
B_SIGNALS = ["B1_mc_vote_disagreement", "B2_softmax_confidence", "B3_member_disagreement_dropout_off"]


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def arm_c(d: dict) -> dict:
    n_classes = d["det_" + LABELS[0]].shape[1]
    v_mc = sum(w * vote_share(d["mc_" + m], n_classes) for w, m in zip(WEIGHTS, LABELS))
    v_det = sum(w * (d["det_" + m].argmax(1)[:, None] == np.arange(n_classes)) for w, m in zip(WEIGHTS, LABELS))
    pred = plurality(v_mc, v_det)
    u = np.round(1.0 - v_mc.max(axis=1), 9)
    probs = sum(w * softmax(d["det_" + m], axis=1) for w, m in zip(WEIGHTS, LABELS))
    b3 = 1.0 - v_det[np.arange(len(pred)), pred]
    y = d["y"]
    err = pred != y
    return {"pred": pred, "err": err, "y": y, "case": d["case"],
            "signals": {"B1_mc_vote_disagreement": u, "B2_softmax_confidence": 1.0 - probs.max(axis=1),
                        "B3_member_disagreement_dropout_off": b3},
            "unanimous": err & (u == 0)}


def arm_e(d: dict, u_c: np.ndarray | None) -> dict:
    n_classes = d["det_" + LABELS[0]].shape[1]
    alpha_m = [alpha_from_logits(d["det_" + m]) for m in LABELS]
    alpha = sum(w * a for w, a in zip(WEIGHTS, alpha_m))
    votes = sum(w * (a.argmax(1)[:, None] == np.arange(n_classes)) for w, a in zip(WEIGHTS, alpha_m))
    evidence = sum(w * (a - 1.0) for w, a in zip(WEIGHTS, alpha_m))
    tie = evidence / evidence.sum(axis=1, keepdims=True)
    pred = plurality(votes, tie)
    y = d["y"]
    err = pred != y
    sc = variance_scores(alpha)
    s5 = 1.0 - votes[np.arange(len(pred)), pred]
    sig = {"S1_epistemic": sc["epistemic"], "S2_aleatoric": sc["aleatoric"], "S3_total": sc["total"],
           "S4_evidence_only": sc["evidence_only"], "S5_member_disagreement": s5}
    if u_c is not None:
        sig["S6_rank_avg_S4_B1"] = (rankdata(sc["evidence_only"]) + rankdata(u_c)) / (2.0 * len(pred))
    return {"pred": pred, "err": err, "y": y, "case": d["case"], "signals": sig,
            "unanimous": err & (s5 == 0)}


def recall_at_budget(score: np.ndarray, mask: np.ndarray, budget: float) -> float:
    """Expected fraction of `mask` instances flagged when the top `budget` share of all instances is
    flagged; instances tied at the cutoff are flagged in proportion (no arbitrary tie-breaking)."""
    n = len(score)
    k = int(round(budget * n))
    if mask.sum() == 0 or k == 0:
        return float("nan")
    thr = np.sort(score)[::-1][k - 1]
    above, tied = score > thr, score == thr
    share = (k - above.sum()) / tied.sum()
    return float((mask[above].sum() + share * mask[tied].sum()) / mask.sum())


def bootstrap_auroc(score, err, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(score), len(score))
        if 0 < err[i].sum() < len(i):
            vals.append(roc_auc_score(err[i], score[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate_arm(a: dict, present: list[int]) -> dict:
    err, score = a["err"], a["signals"]
    out = {"n": int(len(err)), "errors": int(err.sum()), "accuracy": float(1 - err.mean()),
           "macro_f1": float(f1_score(a["y"], a["pred"], labels=present, average="macro", zero_division=0)),
           "n_unanimous_errors": int(a["unanimous"].sum()), "signals": {}}
    for name, s in score.items():
        r = {"auroc": float(roc_auc_score(err, s)), "auroc_ci95": bootstrap_auroc(s, err)}
        for b in BUDGETS:
            r[f"error_recall_at_{int(b * 100)}pct"] = recall_at_budget(s, err, b)
        r["unanimous_flagged_at_20pct"] = recall_at_budget(s, a["unanimous"], 0.20)
        cases = sorted(set(a["case"].tolist()))
        if len(cases) > 1:
            r["auroc_per_case"] = {c: float(roc_auc_score(err[a["case"] == c], s[a["case"] == c]))
                                   for c in cases if 0 < err[a["case"] == c].sum() < (a["case"] == c).sum()}
        out["signals"][name] = r
    return out


def threshold_for_share(score: np.ndarray, share: float = 0.25) -> tuple[float, float]:
    """Threshold t (flag if score >= t) whose flagged share on the reference set is closest to `share`."""
    vals = np.unique(score)
    shares = np.array([(score >= v).mean() for v in vals])
    i = int(np.argmin(np.abs(shares - share)))
    return float(vals[i]), float(shares[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    X = args.extract_dir

    def get(arm, target, seed):
        p = X / f"{arm}_{target}_s{seed}.npz"
        return load(p) if p.exists() else None

    results: dict = {"fold1": {}, "fold2": {}, "stability": {}}
    per_seed = {}
    for fold in ("fold1", "fold2"):
        seeds = (42, 43, 44) if fold == "fold1" else (42,)
        for seed in seeds:
            dc, de = get("C", f"grasp_{fold}", seed), get("E", f"grasp_{fold}", seed)
            if dc is None or de is None:
                continue
            c = arm_c(dc)
            e = arm_e(de, c["signals"]["B1_mc_vote_disagreement"])
            present = sorted(set(c["y"].tolist()))
            results[fold][str(seed)] = {"C": evaluate_arm(c, present), "E": evaluate_arm(e, present)}
            per_seed[(fold, seed)] = (c, e)

    def mean_std(fold, arm, sig, key):
        vals = [results[fold][s][arm]["signals"][sig][key] for s in results[fold]]
        vals = [v for v in vals if v == v]
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)

    summary = {}
    for fold in ("fold1", "fold2"):
        if not results[fold]:
            continue
        summary[fold] = {"acc_C": [results[fold][s]["C"]["accuracy"] for s in results[fold]],
                         "acc_E": [results[fold][s]["E"]["accuracy"] for s in results[fold]],
                         "f1_C": [results[fold][s]["C"]["macro_f1"] for s in results[fold]],
                         "f1_E": [results[fold][s]["E"]["macro_f1"] for s in results[fold]], "signals": {}}
        for arm, sigs in (("E", E_SIGNALS), ("C", B_SIGNALS)):
            for sig in sigs:
                summary[fold]["signals"][sig] = {k: mean_std(fold, arm, sig, k) for k in
                    ("auroc", "error_recall_at_10pct", "error_recall_at_20pct", "error_recall_at_30pct", "unanimous_flagged_at_20pct")}
    results["summary"] = summary

    # fixed-threshold stability: thresholds at 25% flagged on fold1 seed 42, applied unchanged elsewhere
    if ("fold1", 42) in per_seed:
        c1, e1 = per_seed[("fold1", 42)]
        thr = {}
        for sig, s in e1["signals"].items():
            if sig != "S6_rank_avg_S4_B1":
                thr[sig] = threshold_for_share(s)
        for sig, s in c1["signals"].items():
            thr[sig] = threshold_for_share(s)
        targets = {}
        if ("fold2", 42) in per_seed:
            targets["grasp_fold2 (fold2-direction models)"] = per_seed[("fold2", 42)]
        for tname in ("endovis2018", "endovis2017"):
            dc, de = get("C", f"{tname}_fold1", 42), get("E", f"{tname}_fold1", 42)
            if dc is not None and de is not None:
                c = arm_c(dc)
                targets[tname] = (c, arm_e(de, c["signals"]["B1_mc_vote_disagreement"]))
        stab = {"reference_fold1_s42": {}}
        for sig, (t, sh) in thr.items():
            arm = e1 if sig in e1["signals"] else c1
            flag = arm["signals"][sig] >= t
            stab["reference_fold1_s42"][sig] = {"threshold": t, "share_flagged": sh,
                                                "error_catch": float(flag[arm["err"]].mean())}
        for tname, (c, e) in targets.items():
            block = {"n": int(len(c["y"])), "acc_C": float(1 - c["err"].mean()), "acc_E": float(1 - e["err"].mean())}
            for sig, (t, _) in thr.items():
                arm = e if sig in e["signals"] else c
                flag = arm["signals"][sig] >= t
                block[sig] = {"share_flagged": float(flag.mean()),
                              "error_catch": float(flag[arm["err"]].mean()) if arm["err"].any() else None}
            stab[tname] = block
        results["stability"] = stab

    # stage-B go rule
    go = {}
    if "fold1" in summary:
        b1 = summary["fold1"]["signals"]["B1_mc_vote_disagreement"]
        for sig in ("S1_epistemic", "S2_aleatoric", "S3_total", "S4_evidence_only"):
            s = summary["fold1"]["signals"][sig]
            cond_auc = s["auroc"][0] >= b1["auroc"][0] - 0.01
            cond_unan = s["unanimous_flagged_at_20pct"][0] > b1["unanimous_flagged_at_20pct"][0]
            cond_shift = None
            st = results["stability"]
            if all(k in st for k in ("endovis2018", "endovis2017")):
                dev = lambda blk, name: abs(blk[name]["share_flagged"] - 0.25)
                s_ok = all(dev(st[k], sig) <= 0.10 for k in ("endovis2018", "endovis2017"))
                b_ok = all(dev(st[k], "B1_mc_vote_disagreement") <= 0.10 for k in ("endovis2018", "endovis2017"))
                cond_shift = bool(s_ok and not b_ok)
            go[sig] = {"auroc_condition": bool(cond_auc), "unanimous_more_than_B1": bool(cond_unan),
                       "endovis_share_within_10pts_while_B1_not": cond_shift,
                       "go": bool(cond_auc and (cond_unan or bool(cond_shift)))}
    results["go_rule"] = go
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=lambda o: None))
    print(json.dumps({"summary": summary, "go_rule": go}, indent=1, default=lambda o: None)[:6000])


if __name__ == "__main__":
    main()
