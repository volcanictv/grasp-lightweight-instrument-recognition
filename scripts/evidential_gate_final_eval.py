"""Final evidential-gated pipeline accuracy on the official test set, after
SAM2 tracking of the flagged instances (docs/DECISIONS.md 2026-09-28).

Usage:
    python scripts/evidential_gate_final_eval.py --out docs/reports/evidential_pipeline_switch/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
LABELS = list(WEIGHTS)
CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def evidential_track_predict(det_frames: np.ndarray) -> int:
    w = np.array([WEIGHTS[l] for l in LABELS])
    alpha = sum(w[m] * alpha_from_logits(det_frames[:, m, :]) for m in range(len(LABELS)))
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    frame_weight = mu.max(axis=1)
    combined = (frame_weight[:, None] * mu).sum(axis=0)
    return int(combined.argmax())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--gate-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_gate_switch")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.extract)
    y = d["y"]
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in WEIGHTS.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base_pred = mu.argmax(axis=1)
    base_acc = float((base_pred == y).mean())

    tracked: dict[int, int] = {}
    for shard in (0, 1):
        p = args.gate_dir / f"official_frames_shard{shard}.npz"
        if p.exists():
            fd = np.load(p)
            for key in fd.files:
                if key.startswith("det_"):
                    tracked[int(key[4:])] = evidential_track_predict(fd[key])
    print(f"tracked instances available: {len(tracked)}")

    base_info = json.loads((args.gate_dir / "official_base.json").read_text())
    tau = base_info["tau"]
    alpha_s1 = variance_scores(alpha)["epistemic"]
    flag = alpha_s1 >= tau
    missing = [i for i in np.where(flag)[0] if i not in tracked]
    if missing:
        print(f"WARNING: {len(missing)} flagged instances still lack tracking data")

    pred = base_pred.copy()
    for i in np.where(flag)[0]:
        if i in tracked:
            pred[i] = tracked[i]

    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(7)), zero_division=0)
    _, _, f1_base, _ = precision_recall_fscore_support(y, base_pred, labels=list(range(7)), zero_division=0)
    fixed = int(((base_pred != y) & (pred == y) & flag).sum())
    broken = int(((base_pred == y) & (pred != y) & flag).sum())
    out = {
        "n": int(len(y)), "tracked": int(flag.sum()), "share_tracked": float(flag.mean()), "tau": tau,
        "base_accuracy": base_acc, "base_macro_f1": float(f1_score(y, base_pred, average="macro", labels=list(range(7)))),
        "final_accuracy": float((pred == y).mean()), "final_macro_f1": float(f1.mean()),
        "fixed": fixed, "broken": broken, "errors_left": int((pred != y).sum()),
        "per_class": {n: {"n": int(sup[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "f1_base": float(f1_base[i])} for i, n in enumerate(CLASS_NAMES)},
        "shipped_ce_pipeline_for_comparison": {"accuracy": 0.9622509612023767, "macro_f1": 0.9350740238083551},
    }
    print(f"final: accuracy={out['final_accuracy']:.4f} macroF1={out['final_macro_f1']:.4f} "
          f"(base {base_acc:.4f}/{out['base_macro_f1']:.4f}), fixed={fixed} broken={broken}, "
          f"tracked={out['tracked']} ({100*out['share_tracked']:.1f}%)")
    for n, v in out["per_class"].items():
        print(f"  {n:<28} n={v['n']:>4} F1 {v['f1_base']:.3f} -> {v['f1']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
