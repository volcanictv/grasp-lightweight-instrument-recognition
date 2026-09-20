"""Final predictions and every report number for the vote-based pipeline
(docs/DECISIONS.md 2026-09-20): prediction = plurality of the pooled MC-dropout
votes, gate = vote disagreement >= 0.09, combine = uncertainty-weighted vote over
the SAM2-tracked frames. No softmax anywhere.

Writes docs/reports/final_pipeline/metrics.json (overall, per-class, confusion
matrix, before/after tracking, look-alike pair counts) and per_instance.json (one
row per final error, for the error gallery).

Usage:
    python scripts/build_final_pipeline_outputs.py
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
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms

LOOKALIKE_PAIRS = [{"Laparoscopic Grasper", "Suction Instrument"}, {"Bipolar Forceps", "Large Needle Driver"},
                   {"Monopolar Curved Scissors", "Suction Instrument"}]


def report_block(y: np.ndarray, pred: np.ndarray, names: list[str]) -> dict:
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=labels)),
        "per_class": {n: {"support": int(s[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, n in enumerate(names)},
        "confusion_counts": cm.tolist(),
        "confusion_row_pct": (100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).round(1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dirs", type=Path, nargs="+", default=[REPO_ROOT / "docs" / "reports" / "tracking_softmax_free",
                                                                  REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep"])
    parser.add_argument("--gate", type=float, default=0.09)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "final_pipeline")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)

    ds = GraspRegionDataset(Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")), "test",
                            transform=build_transforms(224, train=False))
    names = ds.class_names_ordered()
    file_names = [fn for fn, _s, _b, _l in ds.instances]
    case_of = [fn.split("/")[0] for fn in file_names]

    tracked = {}
    for d in args.dirs:
        for shard in (0, 1):
            path = d / f"frames_shard{shard}.npz"
            if path.exists():
                data = np.load(path)
                for key in data.files:
                    if key.startswith("det_"):
                        i = int(key[4:])
                        tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    flagged = np.where(u >= args.gate)[0]
    missing = [i for i in flagged if i not in tracked]
    if missing:
        raise SystemExit(f"{len(missing)} flagged instances lack tracking data")
    prim = RULES[1]
    final = base.copy()
    for i in flagged:
        final[i] = tracked[i][prim]

    before, after = report_block(y, base, names), report_block(y, final, names)
    fixed = int(((base != y) & (final == y)).sum())
    broken = int(((base == y) & (final != y)).sum())
    pair_counts = {" / ".join(sorted(p)): int(sum(1 for i in np.where(base != y)[0] if {names[y[i]], names[base[i]]} == p)) for p in LOOKALIKE_PAIRS}
    metrics = {
        "gate": args.gate, "n_total": int(len(y)), "tracked": int(len(flagged)), "tracked_pct": float(100 * len(flagged) / len(y)),
        "fixed": fixed, "broken": broken, "errors_before": int((base != y).sum()), "errors_after": int((final != y).sum()),
        "before_tracking": before, "after_tracking": after,
        "lookalike_pair_errors_before_tracking": pair_counts, "lookalike_pair_errors_total": int(sum(pair_counts.values())),
        "top_confusions_after": [[names[a], names[b], int(n)] for (a, b), n in Counter(
            (y[i], final[i]) for i in np.where(final != y)[0]).most_common(6)],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    rows = []
    for i in np.where(final != y)[0]:
        rows.append({
            "idx": int(i), "file_name": file_names[i], "case": case_of[i], "true_class": names[y[i]], "pred_class": names[final[i]],
            "before_pred_class": names[base[i]], "tracked": bool(i in tracked and u[i] >= args.gate),
            "was_correct_before": bool(base[i] == y[i]), "base_vote_share_pred": float(v_mc[i, base[i]]), "base_vote_share_true": float(v_mc[i, y[i]]),
            "base_uncertainty": float(u[i]), "track_uncertainty": tracked[i]["track_uncertainty"] if i in tracked and u[i] >= args.gate else None,
        })
    (args.out_dir / "per_instance.json").write_text(json.dumps(rows, indent=1))

    print(f"final pipeline (gate {args.gate}, V2 weighted vote): accuracy {after['accuracy']:.4f}, macro-F1 {after['macro_f1']:.4f}; "
          f"before tracking {before['accuracy']:.4f} / {before['macro_f1']:.4f}; tracked {len(flagged)}, fixed {fixed}, broken {broken}")
    print(f"errors: before {metrics['errors_before']}, after {metrics['errors_after']}; look-alike pair errors before tracking: {pair_counts} (total {metrics['lookalike_pair_errors_total']})")
    print(f"{'class':<28}{'n':>6}{'P':>7}{'R':>7}{'F1':>7}")
    for n, v in after["per_class"].items():
        print(f"{n:<28}{v['support']:>6}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>7.3f}")
    print("top confusions after:", metrics["top_confusions_after"])
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
