"""Where do the errors left after the tracked pipeline live? Splits the final
errors of a gated tracking run into (a) never routed to tracking because the
gate passed them, (b) tracked but not fixed, (c) tracked and broken, and
describes each by the ensemble's pre-tracking confidence and MC vote
disagreement (docs/DECISIONS.md 2026-09-19).

Motivation: the frame-combination step only acts on instances the gate sent
to tracking, so it cannot touch confidently wrong instances the gate passes.
This measures how many of the remaining errors that is.

Reads the outputs of compute_tracking_gates.py / evaluate_temporal_track_ensemble.py.

Usage:
    python scripts/analyze_residual_errors.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from complementarity_analysis import build_signals
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms


def describe(name: str, idx: np.ndarray, conf: np.ndarray, dis: np.ndarray) -> None:
    if len(idx) == 0:
        print(f"  {name}: none")
        return
    c, d = conf[idx], dis[idx]
    print(f"  {name}: n={len(idx)}  ensemble confidence median {np.median(c):.3f}  "
          f"conf>=0.95: {(c >= 0.95).sum()}  0.80-0.95: {((c >= 0.80) & (c < 0.95)).sum()}  <0.80: {(c < 0.80).sum()}  "
          f"| vote disagreement == 0: {(d == 0).sum()}  <0.20: {(d < 0.20).sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate" / "residual_errors.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    det = [cache[f"det_{m['label']}"] for m in members]
    mc = [cache[f"mc_{m['label']}"] for m in members]
    p_det = sum(w * torch.softmax(torch.from_numpy(d), dim=1).numpy() for w, d in zip(weights, det))
    y_pred = p_det.argmax(axis=1)
    conf = p_det.max(axis=1)
    dis = build_signals(det, mc, weights, y_pred)["vote_dis"]

    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    names = GraspRegionDataset(data_root, "test", transform=build_transforms(224, train=False)).class_names_ordered()
    name_to_idx = {n: i for i, n in enumerate(names)}
    gates = json.loads((args.dir / "gates.json").read_text())
    tracked = {}
    for shard in (0, 1):
        for r in json.loads((args.dir / f"tracked_shard{shard}.json").read_text())["instances"]:
            tracked[r["index"]] = name_to_idx[r["avg_softmax_pred"]]

    out = {}
    for gate_name, flagged in (("confidence gate", gates["conf_flagged_indices"]), ("MC Dropout gate", gates["mc_flagged_indices"])):
        flag = np.zeros(len(y_true), bool)
        flag[flagged] = True
        final = y_pred.copy()
        for i in flagged:
            final[i] = tracked[i]
        wrong_final = final != y_true
        unrouted = np.where(wrong_final & ~flag)[0]
        never_fixed = np.where(wrong_final & flag & (y_pred != y_true))[0]
        broken = np.where(wrong_final & flag & (y_pred == y_true))[0]
        print(f"\n== {gate_name}: {int(wrong_final.sum())} errors left of {len(y_true)} (ensemble alone {int((y_pred != y_true).sum())}) ==")
        describe("(a) never tracked (gate passed them)", unrouted, conf, dis)
        describe("(b) tracked, still wrong (never fixed)", never_fixed, conf, dis)
        describe("(c) tracked and broken (were right)", broken, conf, dis)
        pairs = Counter((names[y_true[i]], names[final[i]]) for i in np.where(wrong_final)[0])
        print("  top confusions among all remaining errors:", [(f"{a} -> {b}", n) for (a, b), n in pairs.most_common(4)])
        out[gate_name] = {"errors_left": int(wrong_final.sum()), "never_tracked": int(len(unrouted)),
                          "tracked_not_fixed": int(len(never_fixed)), "tracked_broken": int(len(broken)),
                          "never_tracked_conf_ge_095": int((conf[unrouted] >= 0.95).sum()),
                          "never_tracked_conf_median": float(np.median(conf[unrouted])) if len(unrouted) else None}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
