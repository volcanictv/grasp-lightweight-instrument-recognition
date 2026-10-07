"""Calibration and error-detection baselines for the flagship evidential ensemble, on the same instruments and seeds.

Methods (probabilities, then the uncertainty signals each one offers):
  softmax ensemble        the four cross-entropy members of arm C, dropout off, weighted mean of softmax        MSP, entropy, energy
  + temperature scaling   one temperature per fold-half, fitted on the other half (cross-fitted by case)       MSP, entropy
  MC dropout (80 passes)  20 dropout passes per member, mean softmax; vote disagreement as in the paper         vote, MSP, entropy, mutual information
  deep ensemble           the three independently trained seeds of arm C (12 networks)                           MSP, entropy, mutual information
  evidential (flagship)   alpha = sum_j w_j (exp(clip z_j) + 1), mu = alpha / alpha_0, one pass                   epistemic S1, MSP, entropy
  evidential, 3 seeds     alpha averaged over the three seeds (12 networks)                                      S1, MSP, entropy
Calibration of the top label (ECE with 15 equal-width bins, NLL, Brier score) and reliability bins; error detection by AUROC and by the share of all errors found when the 20% highest
scores are reviewed; risk-coverage curves. A prediction is the argmax of the method's own probabilities (the paper's single-pass label for the evidential ensemble); the MC-dropout vote
keeps the plurality vote of the paper's earlier table. Held-out fold1 (3,235 instruments) for every method; the test cases for the evidential ensemble only (no cross-entropy extract exists there).

Usage (titanxp, surgical environment, repo root):
    python scripts/calibration_baselines.py --extract-dir experiments_edl/extract --out docs/reports/evidential/calibration_baselines.json
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
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

from evidential_analyze import LABELS, WEIGHTS, arm_c, load
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

SEEDS = (42, 43, 44)
EPS = 1e-12


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + EPS)).sum(axis=-1)


def ece_bins(p: np.ndarray, y: np.ndarray, bins: int = 15):
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append([float(conf[m].mean()), float(ok[m].mean()), int(m.sum())])
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(ece), out


def calib(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    onehot = np.eye(p.shape[1])[y]
    ece, bins = ece_bins(p, y)
    return {"accuracy": float((p.argmax(axis=1) == y).mean()), "nll": float(-np.log(p[np.arange(len(y)), y]).mean()), "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
            "ece": ece, "bins": bins}


def detect(score: np.ndarray, err: np.ndarray) -> dict:
    k = int(round(0.2 * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return {"auroc": float(roc_auc_score(err.astype(int), score)), "caught20": float(err[top].sum() / max(1, err.sum()))}


def risk_coverage(score: np.ndarray, correct: np.ndarray, grid: np.ndarray) -> list[float]:
    order = np.argsort(score, kind="stable")  # most certain first
    ok = correct[order].astype(float)
    cum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    return [float(cum[max(0, int(round(c * len(ok))) - 1)]) for c in grid]


def fit_temperature(logits_list, y, idx) -> float:
    best, bt = 1e9, 1.0
    for T in np.arange(0.5, 6.0, 0.05):
        p = sum(w * softmax(z[idx] / T, axis=1) for w, z in zip(WEIGHTS, logits_list))
        nll = -np.log(np.clip(p[np.arange(len(idx)), y[idx]], EPS, 1)).mean()
        if nll < best:
            best, bt = nll, float(T)
    return bt


def load_seed(extract: Path, split: str, seed: int, arm: str):
    d = load(extract / f"{arm}_grasp_{split}_s{seed}.npz")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    grid = np.round(np.arange(0.5, 1.0001, 0.01), 2)

    # ---------- held-out fold1 ----------
    C = {s: load_seed(args.extract_dir, "fold1", s, "C") for s in SEEDS}
    E = {s: load_seed(args.extract_dir, "fold1", s, "E") for s in SEEDS}
    y = C[42]["y"]
    cases = np.array(C[42]["case"])
    uc = sorted(set(cases.tolist()))
    half = {0: np.where(np.isin(cases, uc[: len(uc) // 2]))[0], 1: np.where(np.isin(cases, uc[len(uc) // 2:]))[0]}
    for s in SEEDS:
        assert (E[s]["y"] == y).all() and (C[s]["y"] == y).all()

    per_seed: dict[str, dict[int, dict]] = {}  # method -> seed -> {"p", "scores", "pred"}
    for s in SEEDS:
        c = C[s]
        logits = [c["det_" + m] for m in LABELS]
        p_soft = sum(w * softmax(z, axis=1) for w, z in zip(WEIGHTS, logits))
        zbar = sum(w * z for w, z in zip(WEIGHTS, logits))
        per_seed.setdefault("Softmax ensemble", {})[s] = {"p": p_soft, "scores": {"max-softmax": 1 - p_soft.max(1), "entropy": entropy(p_soft), "energy": -logsumexp(zbar, axis=1)}}

        # temperature scaling, cross-fitted by case: the temperature used on one half is fitted on the other
        p_T = np.zeros_like(p_soft)
        temps = {}
        for a, b in ((0, 1), (1, 0)):
            T = fit_temperature(logits, y, half[a])
            temps[b] = T
            p_T[half[b]] = sum(w * softmax(z[half[b]] / T, axis=1) for w, z in zip(WEIGHTS, logits))
        per_seed.setdefault("Softmax + temperature scaling", {})[s] = {"p": p_T, "scores": {"max-softmax": 1 - p_T.max(1), "entropy": entropy(p_T)}, "T": temps}

        # MC dropout: 20 passes per member
        mc = [softmax(c["mc_" + m], axis=2) for m in LABELS]  # each (20, N, 7)
        p_mc = sum(w * m.mean(axis=0) for w, m in zip(WEIGHTS, mc))
        mean_h = sum(w * entropy(m).mean(axis=0) for w, m in zip(WEIGHTS, mc))
        cc = arm_c(c)
        per_seed.setdefault("MC dropout (80 passes)", {})[s] = {"p": p_mc, "scores": {"vote disagreement": cc["signals"]["B1_mc_vote_disagreement"], "max-softmax": 1 - p_mc.max(1),
                                                                                       "entropy": entropy(p_mc), "mutual information": entropy(p_mc) - mean_h},
                                                                "pred_override": {"vote disagreement": cc["pred"]}}

        e = E[s]
        alpha = sum(w * alpha_from_logits(e["det_" + m]) for w, m in zip(WEIGHTS, LABELS))
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        per_seed.setdefault("Evidential (flagship, 1 pass)", {})[s] = {"p": mu, "scores": {"epistemic S1": variance_scores(alpha)["epistemic"], "max-belief": 1 - mu.max(1), "entropy": entropy(mu)}}

    # deep ensemble of the three softmax seeds and the pooled evidential ensemble (one entry each)
    P = [per_seed["Softmax ensemble"][s]["p"] for s in SEEDS]
    p_de = sum(P) / len(P)
    mi = entropy(p_de) - sum(entropy(p) for p in P) / len(P)
    alpha_pool = sum(sum(w * alpha_from_logits(E[s]["det_" + m]) for w, m in zip(WEIGHTS, LABELS)) for s in SEEDS) / len(SEEDS)
    mu_pool = alpha_pool / alpha_pool.sum(axis=1, keepdims=True)
    pooled = {"Deep ensemble (3 seeds, 12 nets)": {"p": p_de, "scores": {"max-softmax": 1 - p_de.max(1), "entropy": entropy(p_de), "mutual information": mi}},
              "Evidential, 3 seeds (12 nets)": {"p": mu_pool, "scores": {"epistemic S1": variance_scores(alpha_pool)["epistemic"], "max-belief": 1 - mu_pool.max(1), "entropy": entropy(mu_pool)}}}

    nets = {"Softmax ensemble": 4, "Softmax + temperature scaling": 4, "MC dropout (80 passes)": 80, "Evidential (flagship, 1 pass)": 4, "Deep ensemble (3 seeds, 12 nets)": 12,
            "Evidential, 3 seeds (12 nets)": 12}

    def summarize(entries: dict[int, dict]) -> dict:
        calibs = {s: calib(v["p"], y) for s, v in entries.items()}
        sig_names = list(next(iter(entries.values()))["scores"].keys())
        sigs = {}
        for name in sig_names:
            vals = []
            for s, v in entries.items():
                pred = v.get("pred_override", {}).get(name, v["p"].argmax(axis=1))
                vals.append(detect(v["scores"][name], pred != y))
            sigs[name] = {k: {"mean": float(np.mean([x[k] for x in vals])), "sd": float(np.std([x[k] for x in vals], ddof=1)) if len(vals) > 1 else 0.0, "values": [x[k] for x in vals]} for k in ("auroc", "caught20")}
        cal = {k: {"mean": float(np.mean([c[k] for c in calibs.values()])), "sd": float(np.std([c[k] for c in calibs.values()], ddof=1)) if len(calibs) > 1 else 0.0,
                   "values": [c[k] for c in calibs.values()]} for k in ("accuracy", "nll", "brier", "ece")}
        # reliability bins pooled over seeds: 10 equal-width bins
        conf = np.concatenate([v["p"].max(1) for v in entries.values()])
        ok = np.concatenate([(v["p"].argmax(1) == y).astype(float) for v in entries.values()])
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 9)
        bins = [[float(conf[idx == b].mean()), float(ok[idx == b].mean()), int((idx == b).sum())] for b in range(10) if (idx == b).any()]
        return {"calibration": cal, "signals": sigs, "reliability": bins}

    methods = {}
    for name, entries in per_seed.items():
        methods[name] = {"seeds": len(entries), "networks": nets[name], **summarize(entries)}
        if "T" in next(iter(entries.values())):
            methods[name]["temperatures"] = {str(s): v["T"] for s, v in entries.items()}
    for name, v in pooled.items():
        methods[name] = {"seeds": 3, "networks": nets[name], **summarize({0: v})}

    # risk-coverage curves of the main signal of each method (mean over seeds where there are seeds)
    rc_signals = {"Softmax ensemble: max-softmax": ("Softmax ensemble", "max-softmax"), "Softmax + temperature scaling: max-softmax": ("Softmax + temperature scaling", "max-softmax"),
                  "MC dropout: vote disagreement": ("MC dropout (80 passes)", "vote disagreement"), "MC dropout: mutual information": ("MC dropout (80 passes)", "mutual information"),
                  "Evidential: epistemic S1": ("Evidential (flagship, 1 pass)", "epistemic S1")}
    curves = {}
    for label, (m, sig) in rc_signals.items():
        arr = []
        for s, v in per_seed[m].items():
            pred = v.get("pred_override", {}).get(sig, v["p"].argmax(axis=1))
            arr.append(risk_coverage(v["scores"][sig], pred == y, grid))
        curves[label] = np.mean(arr, axis=0).tolist()
    for label, (m, sig) in {"Deep ensemble: entropy": ("Deep ensemble (3 seeds, 12 nets)", "entropy"), "Evidential, 3 seeds: epistemic S1": ("Evidential, 3 seeds (12 nets)", "epistemic S1")}.items():
        v = pooled[m]
        curves[label] = risk_coverage(v["scores"][sig], v["p"].argmax(axis=1) == y, grid)

    out = {"split": "fold1", "instruments": int(len(y)), "errors_softmax_mean": float(np.mean([(per_seed["Softmax ensemble"][s]["p"].argmax(1) != y).sum() for s in SEEDS])),
           "methods": methods, "risk_coverage": {"coverage": grid.tolist(), "curves": curves}}

    # ---------- official test cases: the evidential ensemble only ----------
    try:
        Et = {s: load_seed(args.extract_dir, "official", s, "E") for s in SEEDS}
        yt = Et[42]["y"]
        res, alphas = {}, []
        for s in SEEDS:
            alpha = sum(w * alpha_from_logits(Et[s]["det_" + m]) for w, m in zip(WEIGHTS, LABELS))
            alphas.append(alpha)
            mu = alpha / alpha.sum(axis=1, keepdims=True)
            cal = calib(mu, yt)
            res[s] = {**{k: cal[k] for k in ("accuracy", "nll", "brier", "ece")}, **detect(variance_scores(alpha)["epistemic"], mu.argmax(1) != yt), "bins": cal["bins"]}
        out["test_evidential"] = {"instruments": int(len(yt)), "per_seed": {str(s): v for s, v in res.items()},
                                  "mean": {k: float(np.mean([v[k] for v in res.values()])) for k in ("accuracy", "nll", "brier", "ece", "auroc", "caught20")}}
    except FileNotFoundError as err:
        print("no test extract:", err)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"fold1, {out['instruments']} instruments; mean over seeds (deep ensemble and pooled evidential: one run)")
    print(f"{'method':38s} {'nets':>4s} {'acc':>6s} {'ECE':>6s} {'NLL':>6s} {'Brier':>6s}   best signal: AUROC / errors caught at 20% review")
    for name, m in methods.items():
        cal = m["calibration"]
        sigs = "  ".join(f"{k} {v['auroc']['mean']:.3f}/{100 * v['caught20']['mean']:.1f}" for k, v in m["signals"].items())
        print(f"{name:38s} {m['networks']:4d} {cal['accuracy']['mean']:6.3f} {cal['ece']['mean']:6.3f} {cal['nll']['mean']:6.3f} {cal['brier']['mean']:6.3f}   {sigs}")
    if "test_evidential" in out:
        t = out["test_evidential"]["mean"]
        print(f"test, evidential: acc {t['accuracy']:.3f} ECE {t['ece']:.3f} NLL {t['nll']:.3f} Brier {t['brier']:.3f} AUROC {t['auroc']:.3f} caught20 {100 * t['caught20']:.1f}")


if __name__ == "__main__":
    main()
