"""Does the ensemble need all four members? Every subset of the four cached members, scored
the way the shipped pipeline scores them: weighted plurality of the 20 dropout-on votes per
member (weights from configs/region_ensemble_deepdropout.yaml, renormalised inside the
subset, or flat), dropout-off votes as the tie-break. Reports accuracy, macro-F1, the AUROC
of the vote-disagreement signal against each subset's own errors, and how many instances the
9% gate would send to tracking. No tracking is re-run: this is the ensemble-alone stage.

Usage:
    python scripts/ensemble_subset_ablation.py --cache <mc_logits_cache_deepdropout.npz> --out results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from build_tracking_sample_b import plurality, vote_share


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    ap.add_argument("--gate", type=float, default=0.09)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    labels = [m["label"] for m in members]
    base_w = {m["label"]: (w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1)) for m in members}
    c = np.load(args.cache)
    y = c["y_true"]
    n_classes = c["det_" + labels[0]].shape[1]
    mc = {l: vote_share(c["mc_" + l], n_classes) for l in labels}
    det = {l: vote_share(c["det_" + l], n_classes) for l in labels}

    def evaluate(subset: tuple[str, ...], flat: bool) -> dict:
        w = np.array([1.0 if flat else base_w[l] for l in subset])
        w = w / w.sum()
        v_mc = sum(wi * mc[l] for wi, l in zip(w, subset))
        v_det = sum(wi * det[l] for wi, l in zip(w, subset))
        pred = plurality(v_mc, v_det)
        u = np.round(1 - v_mc.max(axis=1), 9)
        err = pred != y
        return {"members": list(subset), "weights": "flat" if flat else "config",
                "accuracy": float((~err).mean()), "macro_f1": float(f1_score(y, pred, average="macro")),
                "auroc_vote": float(roc_auc_score(err, u)), "share_at_gate": float((u >= args.gate).mean())}

    rows = []
    for size in (4, 3, 2, 1):
        for subset in itertools.combinations(labels, size):
            for flat in ((False, True) if size > 1 else (False,)):
                rows.append(evaluate(subset, flat))
    print(f"{'members':<58}{'weights':>8}{'acc':>8}{'macroF1':>9}{'AUROC':>8}{'@gate':>8}")
    for r in rows:
        print(f"{'+'.join(r['members']):<58}{r['weights']:>8}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}"
              f"{r['auroc_vote']:>8.3f}{r['share_at_gate']:>8.3f}")
    if args.out:
        args.out.write_text(json.dumps({"n": int(len(y)), "gate": args.gate, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
