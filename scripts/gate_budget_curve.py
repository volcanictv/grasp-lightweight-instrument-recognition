"""Instrument accuracy against the number of instruments sent to tracking, when the instruments are taken in order of the evidential score S1 (highest first), for the
causal real-time run (EdgeTAM, SAM2-tiny masks) and the offline final run (SAM2-large, SAM2 + SAM3 masks). Also the "oracle gate" inside the same pool of tracked instruments
(instruments ordered by how much tracking helps them) as the upper bound for any gate. Reads saved logits and track logits only (no GPU).

Instruments outside the pool of tracked instruments have no track, so each curve stops at the largest K for which the top-K instruments by S1 are all in the pool in every seed.

Usage (titanxp, surgical environment, repo root):
    python scripts/gate_budget_curve.py --out docs/reports/realtime/gate_budget_curve.json
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

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.evaluation.evidential import variance_scores

RUNS = {"causal EdgeTAM (SAM2-tiny masks)": ("rt_tiny", "tracked_b"), "offline SAM2-large (SAM2 + SAM3 masks)": ("final", "tracked")}
SEEDS = (42, 43, 44)


def curve_for(run: str, tracks: str) -> dict:
    per_seed = []
    for s in SEEDS:
        z = np.load(REPO_ROOT / "experiments" / "gtbox_sam" / run / f"logits_s{s}.npz")
        y = z["y"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
        base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        tdir = REPO_ROOT / "experiments" / "gtbox_sam" / run / (tracks if s == 42 else f"{tracks}_s{s}")
        tracked = load_tracked(tdir)
        fused = base.copy()
        for i, frames in tracked.items():
            fused[i] = fuse(frames, CONFIGS["four"])[0]
        order = np.argsort(-s1, kind="stable")
        in_pool = np.array([int(i) in tracked for i in order])
        k_max = int(np.argmin(in_pool)) if not in_pool.all() else len(order)
        gain = (fused == y).astype(int) - (base == y).astype(int)  # +1 fixed, -1 broken, 0 unchanged
        per_seed.append({"y": y, "base": base, "gain": gain, "order": order, "k_max": k_max, "in_pool": np.array([i in tracked for i in range(len(y))])})
    k_max = min(p["k_max"] for p in per_seed)
    ks = list(range(0, k_max + 1, 10))
    acc, orc = [], []
    for p in per_seed:
        base_acc = float((p["base"] == p["y"]).mean())
        g_order = p["gain"][p["order"]]
        cum = np.concatenate([[0], np.cumsum(g_order)])
        acc.append([base_acc + cum[k] / len(p["y"]) for k in ks])
        pool_gain = np.sort(p["gain"][p["in_pool"]])[::-1]
        cum_o = np.concatenate([[0], np.cumsum(np.maximum(pool_gain, 0))])
        orc.append([base_acc + cum_o[min(k, len(pool_gain))] / len(p["y"]) for k in ks])
    acc, orc = np.array(acc), np.array(orc)
    return {"k": ks, "k_max": k_max, "accuracy_mean": acc.mean(axis=0).tolist(), "accuracy_sd": acc.std(axis=0, ddof=1).tolist(),
            "oracle_mean": orc.mean(axis=0).tolist(), "base_accuracy": float(np.mean([(p["base"] == p["y"]).mean() for p in per_seed])), "n": int(len(per_seed[0]["y"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {name: curve_for(*spec) for name, spec in RUNS.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for name, c in out.items():
        i539 = c["k"].index(530) if 530 in c["k"] else None
        print(f"{name}: base {c['base_accuracy']:.4f}, K up to {c['k_max']}, accuracy at K=530 {c['accuracy_mean'][i539] if i539 is not None else float('nan'):.4f}, "
              f"at the end {c['accuracy_mean'][-1]:.4f} (oracle {c['oracle_mean'][-1]:.4f})")


if __name__ == "__main__":
    main()
