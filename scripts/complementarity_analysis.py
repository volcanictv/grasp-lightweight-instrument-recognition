"""Does model disagreement carry error-triage information that confidence does
not? (docs/DECISIONS.md 2026-09-18)

Protocol, fixed before looking at results:
- Data: the cached MC-dropout logits (analyze_mc_logit_signals.py) for the 5
  official-test cases, the only held-out data the deployed ensemble has.
- "Error" = the deterministic ensemble prediction is wrong.
- Any fitted combination is a 2-6 feature logistic regression (C=1, features
  standardized on the training fold) evaluated leave-one-case-out (5 folds,
  never split by frame), scores pooled out-of-fold. No hyperparameter is tuned.
- Judged at matched flag budget (share of instances sent to review), not only
  AUROC, since triage is a budget problem.
- Controls: a permuted disagreement feature (same marginal, no information)
  and a second confidence-type feature, to show what a non-complementary
  second feature does in this same pipeline.

Experiments: E0 single signals; E1 signal dependence; E2 leave-one-case-out
combinations vs confidence alone (+ per-case deltas, controls); E3 matched
budget recall; E4 disagreement within confidence strata; E5 what the
confidence gate misses; E6 which kind of disagreement adds most.

Usage:
    python scripts/complementarity_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CONF_TIERS = [(0.0, 0.80), (0.80, 0.95), (0.95, 1.01)]
DIS_TIERS = [("=0", 0.0, 0.0), ("(0,0.2]", 1e-9, 0.2), (">0.2", 0.2 + 1e-9, 1.0)]


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + 1e-12)).sum(axis=-1)


def vote_share(logits: np.ndarray, n_classes: int) -> np.ndarray:
    if logits.ndim == 2:
        logits = logits[None]
    return (logits.argmax(axis=2)[..., None] == np.arange(n_classes)).mean(axis=0)


def build_signals(det, mc, weights, y_pred):
    n_classes = det[0].shape[1]
    rows = np.arange(len(y_pred))
    p_det = sum(w * torch.softmax(torch.from_numpy(d), dim=1).numpy() for w, d in zip(weights, det))
    mc_p = [torch.softmax(torch.from_numpy(x), dim=2).numpy() for x in mc]
    p_mc = sum(w * p.mean(axis=0) for w, p in zip(weights, mc_p))
    mi = sum(w * (entropy(p.mean(axis=0)) - entropy(p).mean(axis=0)) for w, p in zip(weights, mc_p))
    v_mc = sum(w * vote_share(x, n_classes) for w, x in zip(weights, mc))
    v_det = sum(w * vote_share(d, n_classes) for w, d in zip(weights, det))
    return {
        "conf_unc": 1 - p_det[rows, y_pred],
        "conf_mc_unc": 1 - p_mc[rows, y_pred],
        "entropy_det": entropy(p_det),
        "entropy_mc": entropy(p_mc),
        "vote_dis": 1 - v_mc[rows, y_pred],
        "det_vote_dis": 1 - v_det[rows, y_pred],
        "mutual_info": mi,
    }


def features(sig: dict, names: list) -> np.ndarray:
    cols = []
    for n in names:
        x = sig[n]
        if n in ("conf_unc", "conf_mc_unc"):
            x = np.log(x + 1e-6) - np.log(1 - x + 1e-6)
        elif n in ("entropy_det", "entropy_mc", "mutual_info"):
            x = np.log(x + 1e-3)
        cols.append(x)
    return np.stack(cols, axis=1)


def loco_scores(x: np.ndarray, y: np.ndarray, case_of: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for c in np.unique(case_of):
        te = case_of == c
        scaler = StandardScaler().fit(x[~te])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x[~te]), y[~te])
        out[te] = clf.predict_proba(scaler.transform(x[te]))[:, 1]
    return out


def recall_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Ties at the threshold are flagged proportionally (expected recall under random tie-breaking)."""
    k = budget * len(score)
    t = np.sort(score)[::-1][int(np.ceil(k)) - 1]
    above, tie = score > t, score == t
    frac = (k - above.sum()) / tie.sum()
    return float((y[above].sum() + frac * y[tie].sum()) / y.sum())


def per_case_auroc(y, s, case_of):
    return {c: float(roc_auc_score(y[case_of == c], s[case_of == c])) for c in np.unique(case_of)}


def paired_bootstrap(fn, y, a, b, rng, n_boot=2000):
    deltas = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        deltas.append(fn(y[i], b[i]) - fn(y[i], a[i]))
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_mcdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "complementarity_analysis.json")
    args = parser.parse_args()
    rng = np.random.default_rng(42)

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = [w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members]
    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    det = [cache[f"det_{m['label']}"] for m in members]
    mc = [cache[f"mc_{m['label']}"] for m in members]

    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    ds = GraspRegionDataset(data_root, "test", transform=build_transforms(224, train=False))
    assert (np.array([lbl for _f, _s, _b, lbl in ds.instances]) == y_true).all(), "cache/dataset order mismatch"
    case_of = np.array([fn.split("/")[0] for fn, _s, _b, _l in ds.instances])

    p_det = sum(w * torch.softmax(torch.from_numpy(d), dim=1).numpy() for w, d in zip(weights, det))
    y_pred = p_det.argmax(axis=1)
    y = (y_pred != y_true).astype(int)
    sig = build_signals(det, mc, weights, y_pred)
    cases = sorted(np.unique(case_of))
    print(f"n={len(y)}, errors={y.sum()} ({y.mean():.1%}); cases: " + ", ".join(f"{c}: {int((case_of == c).sum())}/{int(y[case_of == c].sum())}" for c in cases))
    R: dict = {"n": int(len(y)), "n_errors": int(y.sum()), "cases": {c: [int((case_of == c).sum()), int(y[case_of == c].sum())] for c in cases}}

    print("\n== E0 single signals (higher = more uncertain) ==")
    R["E0"] = {}
    for n, s in sig.items():
        R["E0"][n] = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
        print(f"  {n:<14} AUROC {R['E0'][n]['auroc']:.4f}  AUPRC {R['E0'][n]['auprc']:.4f}")
    print(f"  (error prevalence = AUPRC of a random score = {y.mean():.4f})")

    print("\n== E1 Spearman between signals (all / errors only / correct only) ==")
    names = list(sig)
    R["E1"] = {}
    for grp, mask in [("all", np.ones(len(y), bool)), ("errors", y == 1), ("correct", y == 0)]:
        m = np.array([[spearmanr(sig[a][mask], sig[b][mask])[0] for b in names] for a in names])
        R["E1"][grp] = m.round(3).tolist()
    for a in ["vote_dis", "det_vote_dis", "mutual_info"]:
        print(f"  {a:<12} vs conf_unc: " + "  ".join(f"{g} {R['E1'][g][names.index(a)][names.index('conf_unc')]:+.2f}" for g in ("all", "errors", "correct")))

    print("\n== E2 leave-one-case-out combinations (pooled out-of-fold) ==")
    models = {
        "A conf": ["conf_unc"],
        "B vote_dis": ["vote_dis"],
        "C conf+vote_dis": ["conf_unc", "vote_dis"],
        "D conf+mutual_info": ["conf_unc", "mutual_info"],
        "E conf+det_vote_dis": ["conf_unc", "det_vote_dis"],
        "F conf+vote_dis+det_vote_dis+mi": ["conf_unc", "vote_dis", "det_vote_dis", "mutual_info"],
        "G conf+conf_mc (control)": ["conf_unc", "conf_mc_unc"],
        "H conf+entropy_det (control)": ["conf_unc", "entropy_det"],
    }
    oof = {k: loco_scores(features(sig, v), y, case_of) for k, v in models.items()}
    r_conf = np.argsort(np.argsort(sig["conf_unc"])) / len(y)
    r_dis = np.argsort(np.argsort(sig["vote_dis"])) / len(y)
    oof["R rank-avg(conf, vote_dis), no fit"] = r_conf + r_dis
    R["E2"] = {}
    base = oof["A conf"]
    print(f"  {'model':<36} {'AUROC':>7} {'AUPRC':>7}  {'dAUROC vs A [95% boot CI]':<30} per-case dAUROC (>0 in n/5)")
    a_pc = per_case_auroc(y, base, case_of)
    for k, s in oof.items():
        auc, ap = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
        pc = per_case_auroc(y, s, case_of)
        d = [pc[c] - a_pc[c] for c in cases]
        entry = {"auroc": auc, "auprc": ap, "per_case_auroc": pc, "per_case_delta_vs_A": dict(zip(cases, d))}
        line = f"  {k:<36} {auc:>7.4f} {ap:>7.4f}  "
        if k != "A conf":
            m, lo, hi = paired_bootstrap(roc_auc_score, y, base, s, rng)
            entry["delta_auroc_vs_A"] = {"mean": m, "ci95": [lo, hi]}
            line += f"{auc - roc_auc_score(y, base):+.4f} [{lo:+.4f}, {hi:+.4f}]   " + " ".join(f"{x:+.3f}" for x in d) + f"  ({sum(x > 0 for x in d)}/5)"
        R["E2"][k] = entry
        print(line)
    print("  per-case AUROC of A:", {c: round(v, 3) for c, v in a_pc.items()})

    print("\n  controls: what a useless second feature does in the same pipeline (20 draws)")
    ctrl = []
    for _ in range(20):
        perm = dict(sig)
        perm["vote_dis"] = rng.permutation(sig["vote_dis"])
        ctrl.append(roc_auc_score(y, loco_scores(features(perm, models["C conf+vote_dis"]), y, case_of)) - roc_auc_score(y, base))
    R["E2_control_permuted_vote_dis"] = {"mean_delta_auroc": float(np.mean(ctrl)), "sd": float(np.std(ctrl))}
    print(f"  conf + PERMUTED vote_dis: dAUROC {np.mean(ctrl):+.4f} +/- {np.std(ctrl):.4f}")

    print("\n== E3 matched-budget recall of real errors (share of instances flagged) ==")
    incumbent_rate = float((sig["conf_unc"] > 0.20).mean())
    budgets = BUDGETS + [incumbent_rate]
    cand = {"raw conf": sig["conf_unc"], "raw vote_dis": sig["vote_dis"], "A conf (oof)": oof["A conf"],
            "C conf+vote_dis (oof)": oof["C conf+vote_dis"], "F all four (oof)": oof["F conf+vote_dis+det_vote_dis+mi"],
            "R rank-avg (no fit)": oof["R rank-avg(conf, vote_dis), no fit"]}
    R["E3"] = {"incumbent_flag_rate": incumbent_rate, "recall": {}, "delta_C_vs_A": {}}
    print(f"  {'':<24}" + "".join(f"{b:>8.1%}" for b in BUDGETS) + f"{incumbent_rate:>9.1%}*")
    for k, s in cand.items():
        r = [recall_at_budget(y, s, b) for b in budgets]
        R["E3"]["recall"][k] = dict(zip(map(str, budgets), r))
        print(f"  {k:<24}" + "".join(f"{v:>8.1%}" for v in r))
    print("  * = flag rate of the incumbent gate (confidence < 0.80)")
    print("  paired-bootstrap dRecall (C - A) [95% CI]:")
    for b in budgets:
        m, lo, hi = paired_bootstrap(lambda yy, ss, b=b: recall_at_budget(yy, ss, b), y, oof["A conf"], oof["C conf+vote_dis"], rng, 1000)
        R["E3"]["delta_C_vs_A"][str(b)] = {"mean": m, "ci95": [lo, hi]}
        print(f"    budget {b:>6.1%}: {m:+.1%} [{lo:+.1%}, {hi:+.1%}]")

    print("\n== E4 disagreement within confidence strata ==")
    R["E4"] = {"tiers": [], "quintiles": []}
    print(f"  {'conf tier (det prob of pred)':<30}{'n':>6}{'errors':>8}{'err rate':>10}{'AUROC(vote_dis)':>17}")
    conf = 1 - sig["conf_unc"]
    for lo_, hi_ in CONF_TIERS:
        m = (conf >= lo_) & (conf < hi_)
        auc = float(roc_auc_score(y[m], sig["vote_dis"][m])) if 0 < y[m].sum() < m.sum() else float("nan")
        R["E4"]["tiers"].append({"lo": lo_, "hi": hi_, "n": int(m.sum()), "errors": int(y[m].sum()), "auroc_vote_dis": auc})
        print(f"  [{lo_:.2f}, {min(hi_, 1):.2f}){'':<17}{m.sum():>6}{y[m].sum():>8}{y[m].mean():>10.1%}{auc:>17.3f}")
    order = np.argsort(conf, kind="stable")
    for q, idx in enumerate(np.array_split(order, 5)):
        auc = float(roc_auc_score(y[idx], sig["vote_dis"][idx])) if 0 < y[idx].sum() < len(idx) else float("nan")
        R["E4"]["quintiles"].append({"q": q + 1, "conf_range": [float(conf[idx].min()), float(conf[idx].max())], "n": len(idx), "errors": int(y[idx].sum()), "auroc_vote_dis": auc})
        print(f"  conf quintile {q + 1} [{conf[idx].min():.3f}-{conf[idx].max():.3f}]  n={len(idx)} errors={int(y[idx].sum())} AUROC(vote_dis)={auc:.3f}")
    print("\n  error rate (errors/n) by confidence tier x disagreement tier:")
    R["E4"]["grid"] = {}
    print(f"  {'':<16}" + "".join(f"{n:>16}" for n, _, _ in DIS_TIERS))
    for lo_, hi_ in CONF_TIERS:
        cells = []
        for n, dlo, dhi in DIS_TIERS:
            m = (conf >= lo_) & (conf < hi_) & (sig["vote_dis"] >= dlo) & (sig["vote_dis"] <= dhi)
            R["E4"]["grid"][f"conf[{lo_:.2f},{min(hi_, 1):.2f})|dis{n}"] = [int(y[m].sum()), int(m.sum())]
            cells.append(f"{int(y[m].sum())}/{int(m.sum())} ({y[m].mean():.0%})" if m.sum() else "-")
        print(f"  conf[{lo_:.2f},{min(hi_, 1):.2f}) " + "".join(f"{c:>16}" for c in cells))

    print("\n== E5 what the confidence gate (conf < 0.80) misses ==")
    gate = conf < 0.80
    missed = (y == 1) & ~gate
    high_ok = (y == 0) & ~gate
    R["E5"] = {"errors_total": int(y.sum()), "errors_missed_by_gate": int(missed.sum()), "extra_gate": {}}
    print(f"  errors missed by the gate: {missed.sum()}/{y.sum()} ({missed.sum() / y.sum():.1%}); correct instances passed: {high_ok.sum()}")
    for t in [0.05, 0.10, 0.20, 0.30]:
        add = ~gate & (sig["vote_dis"] >= t)
        caught, fp = int((add & (y == 1)).sum()), int((add & (y == 0)).sum())
        # baseline for the same extra budget: lower the confidence threshold instead
        k = int(add.sum())
        cand_idx = np.where(~gate)[0]
        top = cand_idx[np.argsort(-sig["conf_unc"][cand_idx], kind="stable")[:k]]
        base_caught = int(y[top].sum())
        R["E5"]["extra_gate"][str(t)] = {"extra_flagged": k, "extra_errors_caught": caught, "extra_false_flags": fp, "same_budget_via_lower_conf_threshold": base_caught}
        print(f"  add 'disagreement >= {t:.2f}' on top of gate: +{k} flagged, +{caught} errors caught ({caught / max(k, 1):.0%} precision); lowering the conf threshold for the same +{k}: +{base_caught}")
    unanimous = (y == 1) & (conf >= 0.95) & (sig["vote_dis"] == 0)
    R["E5"]["confident_unanimous_errors"] = int(unanimous.sum())
    print(f"  errors with conf >= 0.95 AND zero vote disagreement (invisible to both): {unanimous.sum()}/{y.sum()}")

    print("\n== E6 which kind of disagreement complements confidence (dAUROC vs A, oof) ==")
    R["E6"] = {}
    for k in ["C conf+vote_dis", "D conf+mutual_info", "E conf+det_vote_dis", "G conf+conf_mc (control)", "H conf+entropy_det (control)"]:
        R["E6"][k] = R["E2"][k]["auroc"] - R["E2"]["A conf"]["auroc"]
        print(f"  {k:<32} {R['E6'][k]:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(R, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
