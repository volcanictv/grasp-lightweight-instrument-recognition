"""Scores softmax-free frame-combination rules for SAM2 tracking, and whether
tracking can rescue confident errors (docs/DECISIONS.md 2026-09-19, option B).
Rules and the primary choice were fixed in this file before any tracking ran.

Per tracked instance, each frame f has stochastic votes from every member and
pass; v_f is the weighted vote distribution over classes (weights = ensemble
member weights, split over the passes) and u_f = 1 - max(v_f) its uncertainty.

  V1  pooled vote               argmax_c sum_f v_f[c]
  V2  uncertainty-weighted vote argmax_c sum_f (1 - u_f) v_f[c]      PRIMARY
  V4  unanimity guard           keep the center frame's plurality if the center
                                frame is unanimous (u = 0), otherwise V2
  A0  average softmax           the previous rule, reproduced from the saved
                                deterministic logits; a REFERENCE ONLY, not part
                                of the softmax-free method

Ties: deterministic (dropout-off) votes, then the lower class index.

Reports (1) the gated pipeline (gate-flagged instances get the rule's tracked
prediction, everything else keeps the vote-plurality prediction), (2) what
tracking does to the gate-passed instances: rescue rate on the passed errors,
break rate on a random sample of passed correct instances, and the estimated
effect of tracking every passed instance, (3) how uncertain the tracked
instances still look after combining (track-level uncertainty = 1 - top pooled
vote share), i.e. how many remaining errors are still confidently wrong.

Usage:
    python scripts/evaluate_softmax_free_combine.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from build_tracking_sample_b import plurality, vote_share
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms

RULES = ["V1 pooled vote", "V2 uncertainty-weighted vote (primary)", "V4 unanimity guard", "A0 avg softmax (reference)"]


def track_predictions(det: np.ndarray, mc: np.ndarray, center_pos: int, weights: np.ndarray, n_classes: int) -> dict:
    """det (F, M, C), mc (F, M, S, C) logits for one tracked instance."""
    onehot = mc.argmax(-1)[..., None] == np.arange(n_classes)  # (F, M, S, C)
    v = np.einsum("m,fmc->fc", weights, onehot.mean(axis=2))  # (F, C)
    v_det = np.einsum("m,fmc->fc", weights, (det.argmax(-1)[..., None] == np.arange(n_classes)).astype(float))
    tb = v_det.sum(axis=0)[None]
    purity = v.max(axis=1)  # 1 - u_f
    pooled = v.sum(axis=0)[None]
    v1 = int(plurality(pooled, tb)[0])
    v2 = int(plurality((purity[:, None] * v).sum(axis=0)[None], tb)[0])
    center = v[center_pos]
    center_pred = int(plurality(center[None], v_det[center_pos][None])[0])
    v4 = center_pred if np.isclose(center.max(), 1.0) else v2
    p_avg = np.einsum("m,fmc->fc", weights, softmax(det, axis=-1)).mean(axis=0)
    return {RULES[0]: v1, RULES[1]: v2, RULES[2]: v4, RULES[3]: int(p_avg.argmax()),
            "track_uncertainty": float(1 - pooled[0].max() / pooled[0].sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_softmax_free")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    out_path = args.out or args.dir / "results.json"

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)

    sets = json.loads((args.dir / "sets.json").read_text())
    flagged, passed_wrong = np.array(sets["flagged"]), np.array(sets["passed_wrong"])
    sample = np.array(sets["passed_right_sample"])
    pool_all = set(sets["union"])

    tracked: dict[int, dict] = {}
    for shard in (0, 1):
        path = args.dir / f"frames_shard{shard}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        for key in data.files:
            if key.startswith("det_"):
                i = int(key[4:])
                tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    missing = pool_all - set(tracked)
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} instances have no frame data (e.g. {sorted(missing)[:5]}); use --allow-partial")
    if missing:
        print(f"WARNING dry run: {len(missing)} of {len(pool_all)} instances missing\n")

    names = GraspRegionDataset(Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")), "test",
                               transform=build_transforms(224, train=False)).class_names_ordered()
    labels = list(range(len(names)))
    N = len(y_true)

    def final(rule: str, idx_set) -> np.ndarray:
        pred = base.copy()
        for i in idx_set:
            if i in tracked:
                pred[i] = tracked[i][rule]
        return pred

    R: dict = {"n_total": N, "rows": {}}
    acc0 = float((base == y_true).mean())
    print(f"vote-plurality ensemble alone: accuracy {acc0:.4f}, macro-F1 {f1_score(y_true, base, average='macro', labels=labels):.4f}, errors {int((base != y_true).sum())}")
    print(f"\n(1) gated pipeline: {len(flagged)} flagged instances tracked")
    print(f"{'rule':<42}{'accuracy':>9}{'macro-F1':>10}{'fixed':>7}{'broken':>8}")
    gated = {}
    for rule in RULES:
        pred = final(rule, flagged)
        gated[rule] = pred
        fixed = int(((base != y_true) & (pred == y_true))[flagged].sum())
        broken = int(((base == y_true) & (pred != y_true))[flagged].sum())
        R["rows"][rule] = {"accuracy": float((pred == y_true).mean()), "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels)),
                            "fixed": fixed, "broken": broken, "per_class_f1": dict(zip(names, f1_score(y_true, pred, average=None, labels=labels).round(4).tolist()))}
        print(f"{rule:<42}{R['rows'][rule]['accuracy']:>9.4f}{R['rows'][rule]['macro_f1']:>10.4f}{fixed:>7}{broken:>8}")

    print("\n    paired comparisons on the flagged instances (exact McNemar)")
    R["pairs"] = {}
    for a, b in ((RULES[1], RULES[0]), (RULES[2], RULES[1]), (RULES[1], RULES[3])):
        ok_a, ok_b = gated[a][flagged] == y_true[flagged], gated[b][flagged] == y_true[flagged]
        only_a, only_b = int((ok_a & ~ok_b).sum()), int((ok_b & ~ok_a).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        R["pairs"][f"{a} vs {b}"] = {"only_first": only_a, "only_second": only_b, "p": float(p)}
        print(f"    {a[:30]:<32} vs {b[:30]:<32} only-first {only_a:>3}  only-second {only_b:>3}  p={p:.3f}")

    print(f"\n(2) gate-passed instances: {len(passed_wrong)} wrong (all tracked), {len(sample)} of {sets['passed_right_total']} right (sampled)")
    rng = np.random.default_rng(42)
    R["passed"] = {}
    print(f"{'rule':<42}{'rescued':>9}{'broken':>10}{'est. net on all passed':>26}")
    for rule in RULES:
        rescued = np.array([tracked[i][rule] == y_true[i] for i in passed_wrong if i in tracked])
        broke = np.array([tracked[i][rule] != y_true[i] for i in sample if i in tracked])
        if len(rescued) == 0 or len(broke) == 0:
            continue
        nets = []
        for _ in range(2000):
            r = rng.choice(rescued, len(rescued)).mean()
            b = rng.choice(broke, len(broke)).mean()
            nets.append(r * len(passed_wrong) - b * sets["passed_right_total"])
        net = rescued.mean() * len(passed_wrong) - broke.mean() * sets["passed_right_total"]
        lo, hi = np.percentile(nets, [2.5, 97.5])
        R["passed"][rule] = {"rescued": int(rescued.sum()), "n_wrong": int(len(rescued)), "broken": int(broke.sum()), "n_right_sample": int(len(broke)),
                             "est_net_instances": float(net), "est_net_ci95": [float(lo), float(hi)],
                             "est_net_accuracy_points": float(100 * net / N)}
        print(f"{rule:<42}{int(rescued.sum()):>5}/{len(rescued):<3}{int(broke.sum()):>6}/{len(broke):<4}{net:>+12.1f} [{lo:+.0f}, {hi:+.0f}] instances = {100 * net / N:+.2f} pts")

    print("\n(3) after combining: are the remaining errors still confidently wrong? (flagged instances, primary rule)")
    prim = RULES[1]
    fl = [i for i in flagged if i in tracked]
    wrong_after = np.array([tracked[i][prim] != y_true[i] for i in fl])
    u_track = np.array([tracked[i]["track_uncertainty"] for i in fl])
    if wrong_after.any() and not wrong_after.all():
        auc = float(roc_auc_score(wrong_after, u_track))
        R["post_combine"] = {"auroc_track_uncertainty": auc, "n_wrong": int(wrong_after.sum()),
                              "wrong_with_u_lt_0.20": int((u_track[wrong_after] < 0.20).sum()),
                              "wrong_with_u_lt_0.10": int((u_track[wrong_after] < 0.10).sum()),
                              "right_with_u_lt_0.10": int((u_track[~wrong_after] < 0.10).sum())}
        print(f"    track-level uncertainty flags the remaining errors with AUROC {auc:.3f}")
        print(f"    remaining errors: {int(wrong_after.sum())}; with track uncertainty < 0.20: {R['post_combine']['wrong_with_u_lt_0.20']}, < 0.10: {R['post_combine']['wrong_with_u_lt_0.10']}"
              f" (correct ones with < 0.10: {R['post_combine']['right_with_u_lt_0.10']} of {int((~wrong_after).sum())})")

    out_path.write_text(json.dumps(R, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
