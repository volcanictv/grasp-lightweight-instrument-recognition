"""Scores the 2026-09-29 confirmation runs (docs/DECISIONS.md, "Confirmation runs for the single-pass
evidential default"): the four-member evidential pipeline end to end at seeds 42, 43, 44, and the
three-member pipeline (resnet50_320, resnet50_224, letterbox mobilenet; weights 0.40 / 0.30 / 0.30).

For each seed and ensemble the official-test instances are ranked by S1 (epistemic trace of that
ensemble's own mixed Dirichlet), the top k (k = 525, 649, 833, nested) get the SAM2 track prediction
(frame weight = the frame's top mu_bar, final class = argmax of the weighted sum of mu_bar), all others
keep the single-pass prediction. Seed 42 uses the cached tracked frames of 2026-09-28 and exp1 wherever
they exist and the re-tracked logits otherwise; the re-tracked seed 42 logits are also compared with the
cached ones on the instances both hold.

Usage (any machine holding the npz files):
    python scripts/evidential_seeds_e2e_eval.py --extract-dir experiments_edl/extract \
        --gate-dir experiments/evidential_gate_switch --default-dir experiments/evidential_default \
        --seeds-dir experiments/evidential_seeds --out-dir docs/reports/evidential_default
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

ORDER = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
CONFIGS = {"four": {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20},
           "three": {"resnet50_320": 0.40, "resnet50_224": 0.30, "letterbox_crop": 0.30}}
BUDGETS = (525, 649, 833)
VOTE = {525: (0.9581, 0.9295), 649: (0.9595, 0.9309), 833: (0.9623, 0.9351)}


def alpha_mix(get, weights: dict) -> np.ndarray:
    return sum(w * alpha_from_logits(get(k)) for k, w in weights.items())


def track_predict(det_frames: np.ndarray, weights: dict) -> int:
    """det_frames: (frames, 4 members in ORDER, classes) logits."""
    alpha = sum(w * alpha_from_logits(det_frames[:, ORDER.index(k), :]) for k, w in weights.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return int((mu.max(axis=1)[:, None] * mu).sum(axis=0).argmax())


def load_frames(paths, prefix: str) -> dict[int, np.ndarray]:
    out = {}
    for p in paths:
        if not p.exists():
            continue
        z = np.load(p)
        for k in z.files:
            if k.startswith(prefix):
                out[int(k[len(prefix):])] = z[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "experiments_edl" / "extract")
    ap.add_argument("--gate-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_gate_switch")
    ap.add_argument("--default-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_default")
    ap.add_argument("--seeds-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_seeds")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cached42 = load_frames([args.gate_dir / f"official_frames_shard{s}.npz" for s in (0, 1)]
                           + [args.default_dir / f"exp1_frames_shard{s}.npz" for s in (0, 1)], "det_")
    new_paths = [args.seeds_dir / f"frames_shard{s}.npz" for s in (0, 1)]
    new_z = [np.load(p) for p in new_paths if p.exists()]
    new = {}  # (tag, idx) -> logits
    for z in new_z:
        for k in z.files:
            if k.startswith("det_s"):
                tag, idx = k[4:].split("_", 1)
                new[(tag, int(idx))] = z[k]
    print(f"cached seed-42 instances {len(cached42)}, re-tracked (tag, instance) arrays {len(new)}")

    # re-track reproducibility check on seed 42
    both = sorted(i for (t, i) in new if t == "s42" and i in cached42)
    agree, maxdiff = [], []
    for i in both:
        a, b = cached42[i], new[("s42", i)]
        if a.shape != b.shape:
            agree.append(None)
            continue
        maxdiff.append(float(np.abs(a - b).max()))
        agree.append(track_predict(a, CONFIGS["four"]) == track_predict(b, CONFIGS["four"]))
    check = {"instances_compared": len(both), "shape_mismatch": int(sum(x is None for x in agree)),
             "track_prediction_agreement": float(np.mean([x for x in agree if x is not None])) if agree else None,
             "max_abs_logit_diff_max": max(maxdiff) if maxdiff else None,
             "max_abs_logit_diff_median": float(np.median(maxdiff)) if maxdiff else None}
    print("re-track check", check)

    results = {"reproducibility_check": check, "runs": {}}
    for cname, weights in CONFIGS.items():
        for seed in (42, 43, 44):
            d = np.load(args.extract_dir / f"E_grasp_official_s{seed}.npz")
            y = d["y"]
            alpha = alpha_mix(lambda k: d[f"det_{k}"], weights)
            mu = alpha / alpha.sum(axis=1, keepdims=True)
            base = mu.argmax(axis=1)
            s1 = variance_scores(alpha)["epistemic"]
            order = np.argsort(-s1, kind="stable")
            err = base != y
            tag = f"s{seed}"

            def frames_for(i: int):
                if seed == 42 and i in cached42:
                    return cached42[i]
                return new.get((tag, i))

            run = {"seed": seed, "ensemble": cname, "base_accuracy": float((~err).mean()),
                   "base_macro_f1": float(f1_score(y, base, average="macro", labels=list(range(7)))),
                   "auroc_s1": float(roc_auc_score(err, s1)), "base_errors": int(err.sum()), "rows": []}
            tracked = {}
            for k in BUDGETS:
                top = order[:k]
                miss = [int(i) for i in top if frames_for(int(i)) is None]
                if miss:
                    run["rows"].append({"budget": k, "scored": False, "missing_tracking": len(miss)})
                    continue
                pred = base.copy()
                flag = np.zeros(len(y), bool)
                flag[top] = True
                for i in top:
                    i = int(i)
                    if i not in tracked:
                        tracked[i] = track_predict(frames_for(i), weights)
                    pred[i] = tracked[i]
                run["rows"].append({
                    "budget": k, "scored": True, "accuracy": float((pred == y).mean()),
                    "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(7)))),
                    "fixed": int(((base != y) & (pred == y) & flag).sum()),
                    "broken": int(((base == y) & (pred != y) & flag).sum()),
                    "share_of_base_errors_inside": float((base[top] != y[top]).sum() / max(1, (base != y).sum())),
                    "vote_pipeline_accuracy": VOTE[k][0], "vote_pipeline_macro_f1": VOTE[k][1]})
            results["runs"][f"{cname}_s{seed}"] = run
            for r in run["rows"]:
                print(cname, seed, r)

    summary = {}
    for cname in CONFIGS:
        runs = [results["runs"][f"{cname}_s{s}"] for s in (42, 43, 44)]
        summary[cname] = {"base_accuracy": _ms([r["base_accuracy"] for r in runs]),
                          "base_macro_f1": _ms([r["base_macro_f1"] for r in runs]),
                          "auroc_s1": _ms([r["auroc_s1"] for r in runs]), "by_budget": {}}
        for j, k in enumerate(BUDGETS):
            rows = [r["rows"][j] for r in runs]
            if all(r["scored"] for r in rows):
                summary[cname]["by_budget"][str(k)] = {
                    "accuracy": _ms([r["accuracy"] for r in rows]), "macro_f1": _ms([r["macro_f1"] for r in rows]),
                    "vote_pipeline_accuracy": VOTE[k][0], "vote_pipeline_macro_f1": VOTE[k][1]}
    results["summary"] = summary
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results_evidential_seeds_all.json").write_text(json.dumps(results, indent=1))
    print("wrote", args.out_dir / "results_evidential_seeds_all.json")


def _ms(v: list[float]) -> dict:
    a = np.array(v)
    return {"values": [float(x) for x in a], "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


if __name__ == "__main__":
    main()
