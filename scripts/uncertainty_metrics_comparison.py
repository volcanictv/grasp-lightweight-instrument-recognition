"""Richer uncertainty-metric comparison of the three gate signals from cached logits only
(no GPU, no training, no SAM2).

Signals (each is scored against ITS OWN ensemble's errors, and the two ensembles differ):
  arm C, the shipped deep-dropout CE ensemble:  B1 MC-Dropout vote disagreement (80 votes),
                                                B2 softmax confidence (1 - max of the weighted average softmax)
  arm E, the evidential ensemble (Dirichlet):   S1 epistemic trace, S2 aleatoric, S3 total, S4 evidence only
Both arms use the shipped member weights (0.40, 0.20, 0.20, 0.20) and the same official split; arm E is
a separately trained model (seed 42), so a signal comparison across arms also compares two models.

Metrics: AUROC, AUPRC (errors positive), AURC, risk-coverage points, error recall and precision at fixed
review budgets, selective accuracy at fixed coverage, instance-level and case-level bootstrap intervals with
paired differences, per-class and per-case AUROC, calibration (ECE, Brier, NLL, reliability bins) of four
probability readouts, and the aleatoric-vs-epistemic check on the documented closed-jaw confusion pairs.
Ties in a score are handled in expectation (tied instances share errors equally), never by arbitrary order.

Usage:
    python scripts/uncertainty_metrics_comparison.py \
        --official-e E_grasp_official_s42.npz --official-mc mc_logits_cache_deepdropout.npz \
        --fold1-e E_grasp_fold1_s42.npz --fold1-c C_grasp_fold1_s42.npz \
        --out docs/reports/uncertainty_metrics/results.json [--fig docs/reports/uncertainty_metrics/risk_coverage.png]
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from build_tracking_sample_b import plurality, vote_share
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

LABELS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.20])
CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
# documented closed-jaw / not-visible-in-frame ceiling pairs (report Limitations), either direction
CLOSED_JAW_PAIRS = [("Laparoscopic Grasper", "Suction Instrument"), ("Bipolar Forceps", "Large Needle Driver"),
                    ("Monopolar Curved Scissors", "Suction Instrument")]
BUDGETS = (525, 649, 833)          # official-test counts; scaled by share on other sets
BUDGET_REF_N = 2861
COVERAGES = (0.80, 0.90, 0.95)
RC_GRID = np.round(np.linspace(0.05, 1.0, 20), 2)
ECE_BINS = 15
PROB_EPS = 1e-6
N_BOOT = 2000


# ---------------------------------------------------------------- loading

def _npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_arms(e: dict, c: dict, e_pred: str = "mu") -> dict:
    """e: evidential extract (det_* logits). c: CE deep-dropout cache (det_*, mc_*). Returns both arms.
    e_pred 'mu': arm E predicts argmax of the ensemble mean Dirichlet (what the evidential pipeline uses, base 0.9207);
    'plurality': member-vote plurality as in evidential_analyze.py (base 0.9273)."""
    y = e["y"]
    n_classes = e["det_" + LABELS[0]].shape[1]
    idx = np.arange(len(y))

    # arm C (shipped): vote plurality with dropout-off votes as the tie-break
    v_mc = sum(w * vote_share(c["mc_" + m], n_classes) for w, m in zip(WEIGHTS, LABELS))
    v_det = sum(w * (c["det_" + m].argmax(1)[:, None] == np.arange(n_classes)) for w, m in zip(WEIGHTS, LABELS))
    pred_c = plurality(v_mc, v_det)
    u = np.round(1.0 - v_mc.max(axis=1), 9)
    p_sm = sum(w * softmax(c["det_" + m], axis=1) for w, m in zip(WEIGHTS, LABELS))
    p_mcsm = sum(w * softmax(c["mc_" + m], axis=2).mean(axis=0) for w, m in zip(WEIGHTS, LABELS))
    arm_c = {"pred": pred_c, "err": pred_c != y,
             "signals": {"B1_mc_vote_disagreement": u, "B2_softmax_confidence": 1.0 - p_sm.max(axis=1)},
             "readouts": {"softmax_avg": p_sm, "mc_vote_share": v_mc, "mc_mean_softmax": p_mcsm},
             "top_vote_share": v_mc[idx, pred_c]}

    # arm E (evidential): plurality of member argmax votes, evidence-share tie-break (as in evidential_analyze)
    alpha_m = [alpha_from_logits(e["det_" + m]) for m in LABELS]
    alpha = sum(w * a for w, a in zip(WEIGHTS, alpha_m))
    votes = sum(w * (a.argmax(1)[:, None] == np.arange(n_classes)) for w, a in zip(WEIGHTS, alpha_m))
    evidence = sum(w * (a - 1.0) for w, a in zip(WEIGHTS, alpha_m))
    pred_plur = plurality(votes, evidence / evidence.sum(axis=1, keepdims=True))
    sc = variance_scores(alpha)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    pred_e = mu.argmax(axis=1) if e_pred == "mu" else pred_plur
    p_esm = sum(w * softmax(e["det_" + m], axis=1) for w, m in zip(WEIGHTS, LABELS))
    arm_e = {"pred": pred_e, "err": pred_e != y,
             "signals": {"S1_epistemic": sc["epistemic"], "S2_aleatoric": sc["aleatoric"],
                         "S3_total": sc["total"], "S4_evidence_only": sc["evidence_only"]},
             "readouts": {"evidential_mu_bar": mu, "evidential_arm_softmax_avg": p_esm},
             "alpha0": alpha.sum(axis=1)}
    return {"y": y, "case": e["case"], "C": arm_c, "E": arm_e}


# ---------------------------------------------------------------- tie-aware ranking metrics

def _frac_err_sorted(score: np.ndarray, err: np.ndarray, descending: bool) -> np.ndarray:
    """Per-position expected error indicator when instances are ordered by score; instances tied
    on the score share the tied group's mean error (no arbitrary tie-break)."""
    key = np.round(score.astype(np.float64), 12)
    _, inv = np.unique(key, return_inverse=True)
    gmean = np.bincount(inv, weights=err.astype(float)) / np.bincount(inv)
    order = np.argsort(-key if descending else key, kind="stable")
    return gmean[inv[order]]


def aurc(score: np.ndarray, err: np.ndarray) -> float:
    fe = _frac_err_sorted(score, err, descending=False)          # keep least uncertain first
    return float((np.cumsum(fe) / np.arange(1, len(fe) + 1)).mean())


def oracle_aurc(err: np.ndarray) -> float:
    fe = np.sort(err.astype(float))                              # correct first, errors last
    return float((np.cumsum(fe) / np.arange(1, len(fe) + 1)).mean())


def risk_coverage(score: np.ndarray, err: np.ndarray, grid=RC_GRID) -> list[float]:
    fe = _frac_err_sorted(score, err, descending=False)
    cs = np.cumsum(fe)
    n = len(fe)
    return [float(cs[max(int(round(g * n)) - 1, 0)] / max(int(round(g * n)), 1)) for g in grid]


def oracle_risk_coverage(err: np.ndarray, grid=RC_GRID) -> list[float]:
    fe = np.sort(err.astype(float))
    cs = np.cumsum(fe)
    n = len(fe)
    return [float(cs[max(int(round(g * n)) - 1, 0)] / max(int(round(g * n)), 1)) for g in grid]


def budget_metrics(score: np.ndarray, err: np.ndarray, k: int) -> tuple[float, float]:
    fe = _frac_err_sorted(score, err, descending=True)
    caught = float(fe[:k].sum())
    return caught / float(err.sum()), caught / k


def selective_acc(score: np.ndarray, err: np.ndarray, coverage: float) -> float:
    fe = _frac_err_sorted(score, err, descending=False)
    m = int(round(coverage * len(fe)))
    return 1.0 - float(fe[:m].sum() / m)


def safe_auroc(err: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(err, score)) if 0 < err.sum() < len(err) else float("nan")


def signal_metrics(score: np.ndarray, err: np.ndarray, n_ref: int) -> dict:
    n = len(err)
    ks = [int(round(b / BUDGET_REF_N * n)) if n != BUDGET_REF_N else b for b in BUDGETS]
    out = {"auroc": safe_auroc(err, score), "auprc": float(average_precision_score(err, score)),
           "aurc": aurc(score, err), "risk_coverage": risk_coverage(score, err),
           "selective_accuracy": {f"{int(c * 100)}%": selective_acc(score, err, c) for c in COVERAGES},
           "at_budget": {}}
    for b, k in zip(BUDGETS, ks):
        rec, prec = budget_metrics(score, err, k)
        out["at_budget"][str(b)] = {"k_flagged": k, "error_recall": rec, "precision": prec}
    return out


# ---------------------------------------------------------------- bootstrap

def bootstrap(signals: dict[str, np.ndarray], errs: dict[str, np.ndarray], cases: np.ndarray, level: str,
              rng: np.random.Generator, diffs: list[tuple[str, str]]) -> dict:
    """signals[name] is scored against errs[name]; resample instances or cases jointly (paired)."""
    n = len(cases)
    case_ids = np.unique(cases)
    members = [np.where(cases == c)[0] for c in case_ids]
    stats = {nm: {"auroc": [], "auprc": [], "aurc": []} for nm in signals}
    for _ in range(N_BOOT):
        if level == "instance":
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(case_ids), len(case_ids))
            idx = np.concatenate([members[i] for i in pick])
        for nm, s in signals.items():
            e_b, s_b = errs[nm][idx], s[idx]
            if not 0 < e_b.sum() < len(e_b):
                for k in stats[nm]:
                    stats[nm][k].append(np.nan)
                continue
            stats[nm]["auroc"].append(roc_auc_score(e_b, s_b))
            stats[nm]["auprc"].append(average_precision_score(e_b, s_b))
            stats[nm]["aurc"].append(aurc(s_b, e_b))
    res = {"single": {}, "paired_diff": {}}
    for nm in signals:
        res["single"][nm] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                             for k, v in stats[nm].items()}
    for a, b in diffs:
        res["paired_diff"][f"{a} minus {b}"] = {}
        for k in ("auroc", "auprc", "aurc"):
            d = np.array(stats[a][k]) - np.array(stats[b][k])
            res["paired_diff"][f"{a} minus {b}"][k] = {
                "mean": float(np.nanmean(d)), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                "share_positive": float(np.nanmean(d > 0))}
    return res


# ---------------------------------------------------------------- calibration

def calibration(p: np.ndarray, y: np.ndarray) -> dict:
    n, k = p.shape
    pred, conf = p.argmax(1), p.max(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, ECE_BINS - 1)
    bins, ece = [], 0.0
    for i in range(ECE_BINS):
        m = b == i
        if m.sum() == 0:
            continue
        gap = abs(conf[m].mean() - correct[m].mean())
        ece += m.mean() * gap
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    onehot = np.eye(k)[y]
    pc = np.clip(p, PROB_EPS, 1.0)
    return {"accuracy": float(correct.mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
            "mean_confidence": float(conf.mean()), "ece_15bin": float(ece),
            "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "nll": float(-np.log(pc[np.arange(n), y]).mean()), "reliability": bins}


# ---------------------------------------------------------------- one dataset

def analyse(name: str, data: dict, seed: int, boot: bool = True) -> dict:
    y, case, C, E = data["y"], data["case"], data["C"], data["E"]
    n = len(y)
    sig_all = {**{k: (v, C["err"]) for k, v in C["signals"].items()},
               **{k: (v, E["err"]) for k, v in E["signals"].items()}}
    out = {"n": n, "cases": {c: int((case == c).sum()) for c in np.unique(case)},
           "ensemble_C": {"accuracy": float(1 - C["err"].mean()), "errors": int(C["err"].sum())},
           "ensemble_E": {"accuracy": float(1 - E["err"].mean()), "errors": int(E["err"].sum())},
           "oracle_aurc": {"C": oracle_aurc(C["err"]), "E": oracle_aurc(E["err"])},
           "oracle_risk_coverage": {"C": oracle_risk_coverage(C["err"]), "E": oracle_risk_coverage(E["err"])},
           "rc_grid": [float(g) for g in RC_GRID],
           "scored_against": {k: ("C errors" if k in C["signals"] else "E errors") for k in sig_all},
           "signals": {}, "per_class_auroc": {}, "per_case_auroc": {}}
    for nm, (s, err) in sig_all.items():
        out["signals"][nm] = signal_metrics(s, err, n)
        out["per_class_auroc"][nm] = {CLASSES[k]: safe_auroc(err[y == k], s[y == k]) if (y == k).sum() else float("nan")
                                      for k in range(len(CLASSES))}
        out["per_case_auroc"][nm] = {str(c): safe_auroc(err[case == c], s[case == c]) for c in np.unique(case)}

    rng = np.random.default_rng(seed)
    sigs = {nm: s for nm, (s, _) in sig_all.items()}
    errs = {nm: err for nm, (_, err) in sig_all.items()}
    diffs = [("S1_epistemic", "B2_softmax_confidence"), ("S1_epistemic", "B1_mc_vote_disagreement"),
             ("B2_softmax_confidence", "B1_mc_vote_disagreement"), ("S4_evidence_only", "S1_epistemic")]
    if boot:
        out["bootstrap_instance"] = bootstrap(sigs, errs, case, "instance", rng, diffs)
        out["bootstrap_case"] = bootstrap(sigs, errs, case, "case", rng, diffs)

    out["calibration"] = {}
    for nm, p in {**C["readouts"], **E["readouts"]}.items():
        out["calibration"][nm] = {"ensemble": "C" if nm in C["readouts"] else "E", **calibration(p, y)}

    # aleatoric-vs-epistemic check on the closed-jaw ceiling pairs (arm E errors; C signals for comparison)
    cj = np.zeros(n, bool)
    for a, b in CLOSED_JAW_PAIRS:
        ia, ib = CLASSES.index(a), CLASSES.index(b)
        cj |= ((y == ia) & (E["pred"] == ib)) | ((y == ib) & (E["pred"] == ia))
    cj_c = np.zeros(n, bool)
    for a, b in CLOSED_JAW_PAIRS:
        ia, ib = CLASSES.index(a), CLASSES.index(b)
        cj_c |= ((y == ia) & (C["pred"] == ib)) | ((y == ib) & (C["pred"] == ia))
    groups = {"closed_jaw_pair_errors": E["err"] & cj, "other_errors": E["err"] & ~cj, "correct": ~E["err"]}
    pct = {k: (rankdata(v) - 0.5) / n for k, v in E["signals"].items()}
    ale = {}
    for g, m in groups.items():
        ale[g] = {"n": int(m.sum()), "mean_alpha0": float(E["alpha0"][m].mean()),
                  **{f"mean_{k}": float(v[m].mean()) for k, v in E["signals"].items()},
                  **{f"mean_percentile_{k}": float(pct[k][m].mean()) for k in E["signals"]}}
    ale["note"] = ("percentile = rank of the score among all instances of this set (0 = most certain, 1 = most uncertain); "
                   "S2 aleatoric = S3 total * alpha0/(alpha0+1), so it tracks S3 and is near-collinear with it")
    out["aleatoric_epistemic"] = ale
    cg = {"closed_jaw_pair_errors": C["err"] & cj_c, "other_errors": C["err"] & ~cj_c, "correct": ~C["err"]}
    out["closed_jaw_check_arm_C"] = {
        g: {"n": int(m.sum()), "mean_B1_vote_disagreement": float(C["signals"]["B1_mc_vote_disagreement"][m].mean()),
            "mean_B2_softmax_uncertainty": float(C["signals"]["B2_softmax_confidence"][m].mean()),
            "share_unanimous_wrong (B1 == 0)": float((C["signals"]["B1_mc_vote_disagreement"][m] == 0).mean())}
        for g, m in cg.items()}
    out["closed_jaw_errors_of_all_errors"] = {"C": [int((C["err"] & cj_c).sum()), int(C["err"].sum())],
                                              "E": [int((E["err"] & cj).sum()), int(E["err"].sum())]}
    return out


# ---------------------------------------------------------------- figure

def make_figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"B1_mc_vote_disagreement": ("#0072B2", "-", "MC vote"), "B2_softmax_confidence": ("#E69F00", "--", "softmax"),
             "S1_epistemic": ("#009E73", "-", "evidential S1"), "S4_evidence_only": ("#D55E00", ":", "evidential S4")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (ds, title) in zip(axes, (("official_test", "Official test (5 cases)"), ("fold1", "Fold1 (4 cases)"))):
        r = res[ds]
        grid = np.array(r["rc_grid"])
        for nm, (col, ls, lab) in style.items():
            ax.plot(grid * 100, np.array(r["signals"][nm]["risk_coverage"]) * 100, color=col, ls=ls, lw=2, label=lab)
        ax.plot(grid * 100, np.array(r["oracle_risk_coverage"]["C"]) * 100, color="#888888", ls="-", lw=1, label="oracle (C)")
        ax.set_title(title)
        ax.set_xlabel("coverage kept, least uncertain first (%)")
        ax.set_ylabel("error rate among kept (%)")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Risk-coverage: each signal against its own ensemble's errors (MC and softmax: ensemble C; evidential: ensemble E)",
                 fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-e", type=Path, required=True)
    ap.add_argument("--official-mc", type=Path, required=True, help="deep-dropout CE cache (det_*, mc_*, y_true)")
    ap.add_argument("--fold1-e", type=Path, required=True)
    ap.add_argument("--fold1-c", type=Path, required=True, help="CE fold1 extract with det_*, mc_*, y, case")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eo, co = _npz(args.official_e), _npz(args.official_mc)
    assert (eo["y"] == co["y_true"]).all(), "official caches are not row-aligned"
    ef, cf = _npz(args.fold1_e), _npz(args.fold1_c)
    assert (ef["y"] == cf["y"]).all(), "fold1 caches are not row-aligned"

    res = {"note": ("Each signal is scored against its own ensemble's errors; ensemble C (CE deep-dropout members) supplies the MC vote "
                    "and softmax signals, ensemble E (evidential members, seed 42) supplies S1-S4. Different models, one seed each."),
           "official_test": analyse("official_test", build_arms(eo, co), args.seed),
           "fold1": analyse("fold1", build_arms(ef, cf), args.seed),
           "sensitivity_e_plurality_pred": {
               "note": "arm E scored against the member-vote plurality prediction (evidential_analyze.py rule) instead of argmax mu_bar",
               "official_test": analyse("official_test", build_arms(eo, co, "plurality"), args.seed, boot=False),
               "fold1": analyse("fold1", build_arms(ef, cf, "plurality"), args.seed, boot=False)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    if args.fig:
        make_figure(res, args.fig)
    for ds in ("official_test", "fold1"):
        r = res[ds]
        print(f"\n== {ds}: n={r['n']}, C acc {r['ensemble_C']['accuracy']:.4f} ({r['ensemble_C']['errors']} err), "
              f"E acc {r['ensemble_E']['accuracy']:.4f} ({r['ensemble_E']['errors']} err)")
        print(f"{'signal':<28}{'AUROC':>8}{'AUPRC':>8}{'AURC':>8}  recall@525/649/833")
        for nm, m in r["signals"].items():
            rec = "/".join(f"{m['at_budget'][b]['error_recall']:.3f}" for b in map(str, BUDGETS))
            print(f"{nm:<28}{m['auroc']:>8.3f}{m['auprc']:>8.3f}{m['aurc']:>8.4f}  {rec}")


if __name__ == "__main__":
    main()
