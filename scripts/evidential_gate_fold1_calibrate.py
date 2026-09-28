"""Fold1 calibration of the evidential tracking gate (docs/DECISIONS.md
2026-09-28, "Switching the shipped pipeline to an evidential gate").

Frame combine (new construction, pre-registered before any result exists):
each propagated frame gets its own deterministic evidential prediction
(argmax mu_bar) and a confidence weight equal to that frame's own top class
probability under mu_bar. Final tracked prediction = argmax over classes of
the sum over frames of (frame weight x that frame's mu_bar for the class).

Threshold selection rule (same as the 2026-09-20 MC-gate confirmation):
maximise fold1 accuracy; among thresholds within 0.0010 of the max, take the
highest (fewest tracked).

Usage:
    python scripts/evidential_gate_fold1_calibrate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
LABELS = list(WEIGHTS)


def evidential_track_predict(det_frames: np.ndarray) -> int:
    """det_frames: (n_frames, n_members, n_classes) raw logits. Returns the
    combined class index for one tracked instance, per the pre-registered
    frame-weighted formula."""
    w = np.array([WEIGHTS[l] for l in LABELS])
    alpha = sum(w[m] * alpha_from_logits(det_frames[:, m, :]) for m in range(len(LABELS)))  # (n_frames, C)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    frame_weight = mu.max(axis=1)  # (n_frames,)
    combined = (frame_weight[:, None] * mu).sum(axis=0)
    return int(combined.argmax())


def main() -> None:
    extract_dir = REPO_ROOT / "experiments_edl" / "extract"
    d = np.load(extract_dir / "E_grasp_fold1_s42.npz")
    y = d["y"]
    n = len(y)
    w = WEIGHTS
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in w.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base_pred = mu.argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    base_acc = float((base_pred == y).mean())
    print(f"fold1 evidential base: n={n} accuracy={base_acc:.4f} n_errors={int((base_pred != y).sum())}")

    gate_dir = REPO_ROOT / "experiments" / "evidential_gate_switch"
    tracked: dict[int, int] = {}
    for shard in (0, 1):
        frames_path = gate_dir / f"fold1_frames_shard{shard}.npz"
        if not frames_path.exists():
            raise SystemExit(f"missing {frames_path}; copy it from titanxp first")
        fd = np.load(frames_path)
        for key in fd.files:
            if key.startswith("det_"):
                idx = int(key[4:])
                tracked[idx] = evidential_track_predict(fd[key])
    print(f"tracked instances available: {len(tracked)}")

    grid = sorted({round(float(x), 6) for x in np.percentile(s1, [98, 96, 94, 92, 90, 88, 85, 80, 75, 70, 65])}, reverse=True)
    need = set(np.where(s1 >= min(grid))[0].tolist())
    missing = need - set(tracked)
    if missing:
        print(f"WARNING: {len(missing)} of {len(need)} flagged instances lack tracking data (dry run for those)")

    rows = {}
    for t in grid:
        flag = s1 >= t
        pred = base_pred.copy()
        for i in np.where(flag)[0]:
            if i in tracked:
                pred[i] = tracked[i]
        acc = float((pred == y).mean())
        macro_f1 = float(f1_score(y, pred, average="macro", labels=list(range(7))))
        fixed = int(((base_pred != y) & (pred == y) & flag).sum())
        broken = int(((base_pred == y) & (pred != y) & flag).sum())
        rows[str(t)] = {"threshold": t, "tracked": int(flag.sum()), "share": float(flag.mean()),
                        "accuracy": acc, "macro_f1": macro_f1, "fixed": fixed, "broken": broken}
        print(f"t={t:.6f} tracked={rows[str(t)]['tracked']:4d} ({100*rows[str(t)]['share']:.1f}%) "
              f"acc={acc:.4f} macroF1={macro_f1:.4f} fixed={fixed} broken={broken}")

    best_acc = max(r["accuracy"] for r in rows.values())
    chosen_t = max((r["threshold"] for r in rows.values() if r["accuracy"] >= best_acc - 0.0010), default=grid[0])
    print(f"\nchosen threshold (max accuracy {best_acc:.4f}, highest t within 0.0010): {chosen_t:.6f}")
    print(f"chosen row: {rows[str(chosen_t)]}")

    out = {"base_accuracy": base_acc, "n": n, "rows": rows, "chosen_threshold": chosen_t,
           "chosen_row": rows[str(chosen_t)], "grid": grid}
    out_path = REPO_ROOT / "docs" / "reports" / "evidential_pipeline_switch"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "fold1_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path / 'fold1_calibration.json'}")


if __name__ == "__main__":
    main()
